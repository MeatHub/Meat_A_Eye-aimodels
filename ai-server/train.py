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

# 1. 장비 설정 (맥북 MPS 우선, 없으면 CPU)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

print(f"🚀 현재 사용 중인 장비: {DEVICE}")

# ===== 설정 (맥북 에어 최적화 + 파인튜닝) =====
CONFIG = {
    "dataset_root": Path(__file__).resolve().parent.parent / "data" / "dataset_final",
    "model_save_path": Path(__file__).resolve().parent / "models" / "meat_vision_b2_pro.pth",
    "pretrained_model_path": Path(__file__).resolve().parent / "models" / "meat_vision_b2_pro.pth",  # 파인튜닝용 기존 모델 경로
    "num_epochs": 5,              # 파인튜닝용으로 줄임 (3~5 epoch 권장)
    "batch_size": 16,             # 맥북 에어 권장 (메모리 부족 시 8로 줄이세요)
    "learning_rate": 5e-6,         # 파인튜닝용 낮은 학습률 (Backbone)
    "head_learning_rate": 5e-4,   # 파인튜닝용 낮은 학습률 (Classifier)
    "train_ratio": 0.8,
    "image_size": 260,
    "num_workers": 0,             # [중요] 맥북 에어 8GB에서는 0이 가장 안전합니다.
    "mixup_alpha": 0.2,
    "fine_tune": True,            # True: 기존 모델 로드 후 파인튜닝, False: 처음부터 학습
    "class_weight_ribeye_tenderloin": 1.3,  # 등심·안심 Loss 가중치 (1.0 = 미적용, 1.2~1.5 권장)
}

# 디렉토리 자동 생성
os.makedirs(CONFIG["model_save_path"].parent, exist_ok=True)

# ===== [핵심 1] 증강 전략 =====
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.3),
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

# ===== [핵심 2] Mixup 및 데이터셋 클래스 (수정 완료) =====

# 1. Mixup 데이터 생성 함수 (장비 충돌 해결됨)
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    # [수정] 계산 전에 데이터를 DEVICE로 먼저 보냅니다.
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 2. Mixup 손실 함수 (누락 복구됨)
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 3. 커스텀 데이터셋 클래스 (누락 복구됨)
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

# ===== [핵심 3] 모델 생성 =====
def create_model_b2(num_classes: int):
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    return model

# ===== 메인 함수 =====
def main():
    # 데이터셋 로드
    print(f"📁 데이터 읽는 중... 경로: {CONFIG['dataset_root']}")
    
    # 폴더가 실제로 있는지 체크
    if not CONFIG["dataset_root"].exists():
        print(f"❌ 에러: 데이터 폴더를 찾을 수 없습니다! ({CONFIG['dataset_root']})")
        return

    train_dataset = datasets.ImageFolder(root=CONFIG["dataset_root"] / "train")
    val_dataset = datasets.ImageFolder(root=CONFIG["dataset_root"] / "val")
    test_dataset = datasets.ImageFolder(root=CONFIG["dataset_root"] / "test")

    num_classes = len(train_dataset.classes)
    print(f"✅ 클래스 개수: {num_classes}개")

    # 등심·안심 Loss 가중치 (클래스 인덱스는 ImageFolder 알파벳 순서)
    weight_val = CONFIG.get("class_weight_ribeye_tenderloin", 1.0)
    if weight_val != 1.0:
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        for name in ("Beef_Ribeye", "Beef_Tenderloin"):
            if name in train_dataset.class_to_idx:
                i = train_dataset.class_to_idx[name]
                class_weights[i] = weight_val
        class_weights = class_weights.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"   📌 등심·안심 Loss 가중치: {weight_val}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # DataLoader 설정
    train_loader = DataLoader(AlbumentationsDataset(train_dataset, train_transform), 
                              batch_size=CONFIG["batch_size"], shuffle=True, 
                              num_workers=CONFIG["num_workers"], pin_memory=False) # 맥북은 pin_memory False가 속 편함
    
    val_loader = DataLoader(AlbumentationsDataset(val_dataset, val_transform), 
                            batch_size=CONFIG["batch_size"], shuffle=False, 
                            num_workers=CONFIG["num_workers"], pin_memory=False)

    test_loader = DataLoader(AlbumentationsDataset(test_dataset, val_transform),
                             batch_size=CONFIG["batch_size"], shuffle=False, 
                             num_workers=CONFIG["num_workers"], pin_memory=False)

    # 모델 준비
    model = create_model_b2(num_classes).to(DEVICE)
    
    # === 파인튜닝: 기존 모델 로드 (D. 학습 전략) ===
    if CONFIG["fine_tune"] and os.path.exists(CONFIG["pretrained_model_path"]):
        print(f"\n📥 기존 모델 로드 중: {CONFIG['pretrained_model_path']}")
        try:
            model.load_state_dict(torch.load(CONFIG["pretrained_model_path"], map_location=DEVICE))
            print("✅ 기존 모델 로드 완료! 파인튜닝 모드로 진행합니다.")
        except Exception as e:
            print(f"⚠️ 기존 모델 로드 실패: {e}")
            print("   처음부터 학습을 시작합니다.")
    else:
        if CONFIG["fine_tune"]:
            print("⚠️ 기존 모델 파일을 찾을 수 없습니다. 처음부터 학습을 시작합니다.")
        else:
            print("🆕 처음부터 학습을 시작합니다.")

    # 옵티마이저 & 스케줄러
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': CONFIG["learning_rate"]},
        {'params': model.classifier.parameters(), 'lr': CONFIG["head_learning_rate"]}
    ], weight_decay=1e-2)

    # criterion은 위에서 class_weight 적용 여부에 따라 이미 정의됨
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    best_val_acc = 0.0
    
    mode_str = "파인튜닝" if (CONFIG["fine_tune"] and os.path.exists(CONFIG["pretrained_model_path"])) else "처음부터 학습"
    print(f"\n🔥 {mode_str} 시작! (MacBook Air가 조금 뜨거워질 수 있습니다)")
    print(f"   - Backbone 학습률: {CONFIG['learning_rate']}")
    print(f"   - Classifier 학습률: {CONFIG['head_learning_rate']}")
    print(f"   - Epoch: {CONFIG['num_epochs']}\n")
    
    for epoch in range(CONFIG["num_epochs"]):
        # === Training Phase ===
        model.train()
        train_loss, train_correct = 0, 0
        
        # 진행 상황 간단 표시
        print(f"\n[Epoch {epoch+1}/{CONFIG['num_epochs']}] 학습 진행 중...", end=" ")
        
        for inputs, labels in train_loader:
            # inputs, labels는 mixup 함수 안에서 to(DEVICE) 처리하므로 여기선 패스해도 됨
            # 하지만 안전하게 한번 더 해도 무방함
            
            # Mixup 적용 (내부에서 DEVICE로 이동)
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, CONFIG["mixup_alpha"])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 정확도 계산 (Mixup 고려)
            pred = outputs.argmax(1)
            train_correct += (lam * (pred == labels_a).float().sum() + 
                             (1 - lam) * (pred == labels_b).float().sum()).item()

        # === Validation Phase ===
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        
        # 정확도 계산
        t_acc = train_correct / len(train_loader.dataset)
        v_acc = val_correct / len(val_loader.dataset)
        avg_train_loss = train_loss / len(train_loader)
        
        # 명확한 출력 형식
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] 결과:")
        print(f"  📊 Train Accuracy:  {t_acc:.4f} ({train_correct}/{len(train_loader.dataset)})")
        print(f"  📊 Train Loss:       {avg_train_loss:.4f}")
        print(f"  📊 Val Accuracy:    {v_acc:.4f} ({val_correct}/{len(val_loader.dataset)})")
        print(f"{'='*60}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  ⭐ 최고 Validation Accuracy 갱신! 모델 저장됨 ({v_acc:.4f})\n")

    # === 최종 테스트 평가 ===
    if os.path.exists(CONFIG["model_save_path"]):
        print("\n=== 🏆 최종 테스트 세트 평가 ===")
        best_model = create_model_b2(num_classes).to(DEVICE)
        best_model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=DEVICE))
        best_model.eval()

        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = best_model(inputs)
                test_correct += (outputs.argmax(1) == labels).sum().item()

        test_acc = test_correct / len(test_loader.dataset)
        print(f"   📊 최종 Test Accuracy: {test_acc:.4f}")
    else:
        print("⚠️ 모델 파일이 저장되지 않았습니다.")

if __name__ == "__main__":
    main()