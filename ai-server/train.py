import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# ===== 설정 (성능 극대화 세팅) =====
CONFIG = {
    "data_dir": Path(__file__).parent.parent / "data" / "dataset_final" / "train",  # split.py 출력 구조
    "model_save_path": Path(__file__).parent / "models" / "vision_b2_imagenet.pth",
    "num_epochs": 20,             # 더 정밀한 학습을 위해 에폭 상향
    "batch_size": 32,
    "learning_rate": 1e-4,        # Backbone용 낮은 학습률
    "head_learning_rate": 1e-3,   # Classifier용 높은 학습률
    "train_ratio": 0.8,
    "image_size": 260,
    "num_workers": 8,             # RTX 5060 사양에 맞춰 상향
    "mixup_alpha": 0.2,           # Mixup 강도
    # ImageNet 전이학습 (imagenet_pretrain_epochs=0 이면 스킵)
    "imagenet_dataset_id": "ILSVRC/imagenet-1k",  # HF: imagenet-1k 동일 데이터
    "imagenet_pretrain_epochs": 0,  # 0: Phase 1 스킵, Meat만 fine-tuning
    "imagenet_batch_size": 64,
    "imagenet_max_samples": 100_000,  # None이면 전체 사용 (약 128만)
    "imagenet_model_path": Path(__file__).parent / "models" / "efficientnet_b2_imagenet.pth",
    # HuggingFace 토큰 (gated ImageNet 접근용). 코드에 넣지 말고 .env의 HF_TOKEN 사용
    "hf_token": None,
}

# ===== [핵심 1] 텍스트 및 질감 강조 증강 전략 =====
# 수정된 증강 로직 (경고 완벽 제거 버전)
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    # ShiftScaleRotate 대신 Affine 사용
    A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.3),
    # CoarseDropout 인자 수정 (최신 버전 규격)
    A.CoarseDropout(
        num_holes_range=(1, 8), 
        hole_height_range=(0.02, 0.1), 
        hole_width_range=(0.02, 0.1), 
        p=0.5
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# [핵심 2] Mixup 구현 (데이터 특징을 강제로 섞어 학습 난이도 상향)
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label


# ImageNet (HuggingFace datasets) → Albumentations 호환 래퍼
class ImageNetAlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        image = row["image"]  # PIL
        image = np.array(image.convert("RGB"))
        label = row["label"]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label

# [핵심 3] 모델 생성 및 차등 학습률 적용 준비
def create_model_b2(num_classes: int, pretrained_path=None):
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    if pretrained_path and Path(pretrained_path).exists():
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        # 1000클래스 classifier 제외, backbone만 로드
        state = {k: v for k, v in state.items() if not k.startswith("classifier")}
        model.load_state_dict(state, strict=False)
        print(f"  ✓ Backbone 로드: {pretrained_path}")
    return model


def create_model_imagenet():
    """ImageNet 1000클래스용 EfficientNet-B2 (전이학습 1단계)."""
    return models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = None

    # ----- [전이학습 1단계] ImageNet 사전학습 (선택) -----
    if CONFIG["imagenet_pretrain_epochs"] > 0:
        print("[Phase 1] ImageNet 사전학습 시작")
        # 토큰: CONFIG["hf_token"] 또는 .env HF_TOKEN. gated 이용약관 동의 필요.
        hf_token = CONFIG.get("hf_token") or os.environ.get("HF_TOKEN")
        if not hf_token:
            print("\n  ⚠ ImageNet(gated) 접근을 위해 HF 토큰이 필요합니다.")
            print("  CONFIG['hf_token'] = 'hf_xxxx' 또는 .env에 HF_TOKEN=hf_xxxx 설정 후 다시 실행.")
            print("  또는 imagenet_pretrain_epochs=0 으로 두면 ImageNet 단계 스킵.\n")
            raise RuntimeError("HF_TOKEN not set. Set CONFIG['hf_token'] or HF_TOKEN env.")
        try:
            imagenet_dataset = load_dataset(
                CONFIG["imagenet_dataset_id"],
                split="train",
                token=hf_token,
            )
        except Exception as e:
            if "gated" in str(e).lower() or "DatasetNotFoundError" in type(e).__name__:
                print("\n  ⚠ ImageNet은 gated 데이터셋입니다.")
                print("  1) https://huggingface.co/datasets/ILSVRC/imagenet-1k 접속 → 로그인 → 이용약관 동의")
                print("  2) CONFIG['hf_token'] = 'hf_xxxx' 또는 .env에 HF_TOKEN=hf_xxxx 입력")
                print("  3) 또는 imagenet_pretrain_epochs=0 으로 두면 ImageNet 단계 스킵 후 Meat만 학습\n")
            raise
        if CONFIG["imagenet_max_samples"] is not None:
            n = min(len(imagenet_dataset), CONFIG["imagenet_max_samples"])
            imagenet_dataset = imagenet_dataset.select(range(n))
            print(f"  ImageNet 서브셋 사용: {n:,} 샘플")
        ds = ImageNetAlbumentationsDataset(imagenet_dataset, val_transform)
        il = DataLoader(
            ds,
            batch_size=CONFIG["imagenet_batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
        )
        model_imagenet = create_model_imagenet().to(device)
        opt = optim.AdamW(model_imagenet.parameters(), lr=1e-4, weight_decay=1e-2)
        crit = nn.CrossEntropyLoss()

        for ep in range(CONFIG["imagenet_pretrain_epochs"]):
            model_imagenet.train()
            running = 0.0
            correct = 0
            total = 0
            for x, y in il:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model_imagenet(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                running += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
            acc = correct / total
            print(f"  ImageNet Epoch [{ep+1}/{CONFIG['imagenet_pretrain_epochs']}] Loss: {running/len(il):.4f} Acc: {acc:.4f}")

        CONFIG["imagenet_model_path"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_imagenet.state_dict(), CONFIG["imagenet_model_path"])
        print(f"  ✓ ImageNet 모델 저장: {CONFIG['imagenet_model_path']}")
        pretrained_path = CONFIG["imagenet_model_path"]
        del model_imagenet, il, ds, imagenet_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ----- [전이학습 2단계] Meat 데이터 fine-tuning -----
    print("[Phase 2] Meat 데이터 fine-tuning")
    data_dir = Path(CONFIG["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"데이터 경로가 없습니다: {data_dir}\n"
            f"  → split.py를 먼저 실행하거나, data/dataset_final/train/ 부위폴더/ 이미지 구조를 준비하세요."
        )
    full_dataset = datasets.ImageFolder(root=str(data_dir))
    num_classes = len(full_dataset.classes)

    train_size = int(CONFIG["train_ratio"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_raw, val_raw = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        AlbumentationsDataset(train_raw, train_transform),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        AlbumentationsDataset(val_raw, val_transform),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    model = create_model_b2(num_classes, pretrained_path=pretrained_path).to(device)

    # 차등 학습률 설정: Backbone은 천천히(1e-4), Head는 빠르게(1e-3)
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': CONFIG["learning_rate"]},
        {'params': model.classifier.parameters(), 'lr': CONFIG["head_learning_rate"]}
    ], weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    best_val_acc = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        # === Training Phase ===
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixup 적용
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, CONFIG["mixup_alpha"])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (lam * (outputs.argmax(1) == labels_a).float().sum() + 
                             (1 - lam) * (outputs.argmax(1) == labels_b).float().sum()).item()

        # === Validation Phase ===
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        
        t_acc, v_acc = train_correct/len(train_loader.dataset), val_correct/len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  ⭐ Best Model Saved! (Acc: {v_acc:.4f})")

if __name__ == "__main__":
    main()