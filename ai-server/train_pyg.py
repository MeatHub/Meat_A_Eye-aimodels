import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    # 폴더 구조에 맞춰 경로 수정
    "train_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\Beef_dataset\train"),
    "val_dir":   Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\Beef_dataset\val"),
    "test_dir":  Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\Beef_dataset\test"),
    
    "model_save_path": Path(__file__).parent / "models" / "models_each" / "meat_vision_b2_beef2.pth",
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "head_learning_rate": 1e-3,
    "image_size": 260,
    "num_workers": 8,
    "mixup_alpha": 0.2,
    "imagenet_dataset_id": "ILSVRC/imagenet-1k",
    "imagenet_pretrain_epochs": 0,
    "imagenet_batch_size": 64,
    "imagenet_max_samples": 100_000,
    "imagenet_model_path": Path(__file__).parent / "models" / "efficientnet_b2_imagenet.pth",
    "hf_token": None,
}

# ===== 증강 및 Mixup 로직 (기존 유지) =====
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.02, 0.1), hole_width_range=(0.02, 0.1), p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def mixup_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
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
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label

class ImageNetAlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    def __len__(self): return len(self.hf_dataset)
    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        image = np.array(row["image"].convert("RGB"))
        label = row["label"]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label

def create_model_b2(num_classes: int, pretrained_path=None):
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    if pretrained_path and Path(pretrained_path).exists():
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith("classifier")}
        model.load_state_dict(state, strict=False)
        print(f"   ✓ Backbone 로드: {pretrained_path}")
    return model

def create_model_imagenet():
    return models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = None

    # ----- [Phase 1] ImageNet (기존 유지) -----
    if CONFIG["imagenet_pretrain_epochs"] > 0:
        # (기존 ImageNet 학습 로직과 동일하므로 생략 가능하나, 구조상 포함)
        pass

    # ----- [전이학습 2단계] Meat 데이터 fine-tuning (수정된 섹션) -----
    print("[Phase 2] Meat 데이터 fine-tuning (Train/Val/Test 분리 로드)")
    
    # 1. 각 폴더 존재 여부 확인
    for d in [CONFIG["train_dir"], CONFIG["val_dir"], CONFIG["test_dir"]]:
        if not d.exists():
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {d}")

    # 2. ImageFolder를 통해 폴더별로 데이터 로드
    train_raw = datasets.ImageFolder(root=str(CONFIG["train_dir"]))
    val_raw = datasets.ImageFolder(root=str(CONFIG["val_dir"]))
    test_raw = datasets.ImageFolder(root=str(CONFIG["test_dir"]))
    
    num_classes = len(train_raw.classes)
    print(f"   ✓ 클래스 수: {num_classes} ({train_raw.classes})")

    # 3. DataLoader 설정
    train_loader = DataLoader(
        AlbumentationsDataset(train_raw, train_transform),
        batch_size=CONFIG["batch_size"], shuffle=True, 
        num_workers=CONFIG["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        AlbumentationsDataset(val_raw, val_transform),
        batch_size=CONFIG["batch_size"], shuffle=False, 
        num_workers=CONFIG["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        AlbumentationsDataset(test_raw, val_transform), # Test는 Val과 동일한 Transform 사용
        batch_size=CONFIG["batch_size"], shuffle=False, 
        num_workers=CONFIG["num_workers"], pin_memory=True,
    )

    # 4. 모델 및 학습 설정 (기존 유지)
    model = create_model_b2(num_classes, pretrained_path=pretrained_path).to(device)

    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': CONFIG["learning_rate"]},
        {'params': model.classifier.parameters(), 'lr': CONFIG["head_learning_rate"]}
    ], weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    # 5. Training Loop
    best_val_acc = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, CONFIG["mixup_alpha"])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (lam * (outputs.argmax(1) == labels_a).float().sum() + 
                              (1 - lam) * (outputs.argmax(1) == labels_b).float().sum()).item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        t_acc, v_acc = train_correct/len(train_raw), val_correct/len(val_raw)
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  ⭐ Best Model Saved! (Acc: {v_acc:.4f})")

    # 6. Final Test Phase (추가됨)
    print("\n[Phase 3] Final Test Evaluation")
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_correct += (outputs.argmax(1) == labels).sum().item()
    
    print(f"  🎯 Final Test Accuracy: {test_correct/len(test_raw):.4f}")

if __name__ == "__main__":
    main()