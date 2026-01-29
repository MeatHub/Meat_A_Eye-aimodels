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

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    "train_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\train"),
    "val_dir":   Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\val"),
    "test_dir":  Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\test"),
    
    "model_save_path": Path(__file__).parent / "models" / "models_b2" / "meat_vision_b2_hard.pth",
    
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "image_size": 260,
    "num_workers": 8,  # 윈도우 충돌 방지를 위해 0으로 고정
}

# ==========================================
# 2. 데이터 증강 (최신 문법 반영)
# ==========================================
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.1), rotate=(-30, 30), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.OneOf([
        A.ToGray(p=1.0),
        A.ToSepia(p=1.0),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
    ], p=0.5),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        A.MotionBlur(p=1.0),
    ], p=0.3),
    # fill_value 제거 (최신 버전 호환성)
    A.CoarseDropout(
        num_holes_range=(4, 8),
        hole_height_range=(10, 20),
        hole_width_range=(10, 20),
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

# ==========================================
# 3. 데이터셋 및 유틸리티 함수
# ==========================================
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), correct / total

# ==========================================
# 4. 메인 실행 (학습 + 최종 Test 평가)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Device: {device}")

    # 데이터 로드
    train_raw = datasets.ImageFolder(root=CONFIG["train_dir"])
    val_raw   = datasets.ImageFolder(root=CONFIG["val_dir"])
    test_raw  = datasets.ImageFolder(root=CONFIG["test_dir"])
    
    print(f"Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")

    train_loader = DataLoader(AlbumentationsDataset(train_raw, train_transform), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader   = DataLoader(AlbumentationsDataset(val_raw, val_transform), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader  = DataLoader(AlbumentationsDataset(test_raw, val_transform), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 모델 초기화
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_raw.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])

    best_acc = 0.0

    print("🚀 학습 시작...")
    for epoch in range(CONFIG["num_epochs"]):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            CONFIG["model_save_path"].parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  🔥 Best Model Saved! (Acc: {best_acc:.4f})")

    # ==========================================
    # 5. 최종 Test 성능 체크 로직
    # ==========================================
    print("\n" + "="*20 + " [FINAL TEST EVALUATION] " + "="*20)
    print(f"Loading Best Model from: {CONFIG['model_save_path'].name}")
    
    # 최고 성능 모델 다시 불러오기
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"🏆 최종 Test Set 정확도: {test_acc*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()