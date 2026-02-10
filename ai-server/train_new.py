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
# 1. 설정 (기존 스타일 유지)
# ==========================================
CONFIG = {
    "train_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\train"),
    "val_dir":   Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\val"),
    "test_dir":  Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\test"),
    
    # 원래 쓰시던 경로 형식 그대로!
    "model_save_path": Path(__file__).parent / "models" / "models_b2" / "meat_vision_b2_dataset.pth",
    
    "num_epochs": 15,        # 6500장이면 15회 정도가 적당합니다.
    "batch_size": 32,
    "learning_rate": 0.001,
    "image_size": 260,
    "num_workers": 0,        # 윈도우 에러 방지용 0
}

# ==========================================
# 2. 데이터 증강 (6500장 규모에 최적화)
# ==========================================
# 고기의 질감(Texture)과 색상(Color)을 유지하면서 각도만 다양하게 주는 전략입니다.
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ==========================================
# 3. 데이터셋 및 유틸리티
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
# 4. 메인 실행
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 시작 (Device: {device})")

    # 데이터 로드
    train_raw = datasets.ImageFolder(root=CONFIG["train_dir"])
    val_raw   = datasets.ImageFolder(root=CONFIG["val_dir"])
    test_raw  = datasets.ImageFolder(root=CONFIG["test_dir"])
    
    num_classes = len(train_raw.classes)
    print(f"📊 클래스 수: {num_classes} | 학습 데이터: {len(train_raw)}장")

    train_loader = DataLoader(AlbumentationsDataset(train_raw, train_transform), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader   = DataLoader(AlbumentationsDataset(val_raw, val_transform), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader  = DataLoader(AlbumentationsDataset(test_raw, val_transform), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 모델 설정 (17개 클래스 자동 대응)
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])

    best_acc = 0.0

    for epoch in range(CONFIG["num_epochs"]):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            CONFIG["model_save_path"].parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  🔥 Best Model Saved! ({CONFIG['model_save_path'].name})")

    # 최종 Test 평가
    print("\n" + "="*20 + " [FINAL TEST EVALUATION] " + "="*20)
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    _, test_acc = validate(model, test_loader, criterion, device)
    print(f"🏆 최종 Test Set 정확도: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()