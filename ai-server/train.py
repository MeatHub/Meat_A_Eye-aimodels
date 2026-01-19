"""
Meat-A-Eye EfficientNet-B0 학습 스크립트
소고기 부위 분류 모델 학습
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


# ===== 설정 =====
CONFIG = {
    "data_dir": Path(__file__).parent.parent / "data" / "Retail Beef Cuts Dataset" / "NA_Meat_Dataset",
    "model_save_path": Path(__file__).parent / "models" / "meat_vision_v2.pth",
    "num_epochs": 20,
    "batch_size": 64,  # GPU 활용을 위해 증가
    "learning_rate": 0.001,
    "train_ratio": 0.8,
    "image_size": 224,
    "num_workers": 4,  # GPU 데이터 로딩 병렬화
    "pin_memory": True,  # GPU 전송 최적화
}


# ===== Albumentations 증강 (포장 비닐 반사, 조명 변화 대응) =====
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomRotate90(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class AlbumentationsDataset(torch.utils.data.Dataset):
    """Albumentations 적용 커스텀 데이터셋"""

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


def create_model(num_classes: int, pretrained: bool = True):
    """EfficientNet-B0 모델 생성"""
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        model = models.efficientnet_b0(weights=None)

    # 분류 헤드 수정
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """1 에폭 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

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


def main():
    print("=" * 50)
    print("Meat-A-Eye EfficientNet-B0 학습 시작")
    print("=" * 50)

    # 디바이스 설정 (GPU 강제 사용)
    if not torch.cuda.is_available():
        print("오류: CUDA를 사용할 수 없습니다!")
        return

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # GPU 최적화
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 데이터 로드
    print(f"\n데이터 경로: {CONFIG['data_dir']}")

    if not CONFIG["data_dir"].exists():
        print(f"오류: 데이터 경로가 존재하지 않습니다: {CONFIG['data_dir']}")
        return

    # ImageFolder로 데이터셋 로드 (폴더명 = 클래스명)
    full_dataset = datasets.ImageFolder(root=CONFIG["data_dir"])

    # 클래스 정보 출력
    print(f"\n클래스 목록: {full_dataset.classes}")
    print(f"총 이미지 수: {len(full_dataset)}")

    # Train/Val 분리
    train_size = int(CONFIG["train_ratio"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Albumentations 적용
    train_dataset = AlbumentationsDataset(train_dataset, train_transform)
    val_dataset = AlbumentationsDataset(val_dataset, val_transform)

    # DataLoader (GPU 최적화)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        persistent_workers=True if CONFIG["num_workers"] > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        persistent_workers=True if CONFIG["num_workers"] > 0 else False
    )

    # 모델 생성
    num_classes = len(full_dataset.classes)
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    print(f"\n모델: EfficientNet-B0 (클래스 수: {num_classes})")

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 학습
    best_val_acc = 0.0
    print("\n학습 시작...")
    print("-" * 50)

    for epoch in range(CONFIG["num_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CONFIG["model_save_path"].parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  → 최고 모델 저장! (Val Acc: {val_acc:.4f})")

    print("-" * 50)
    print(f"\n학습 완료! 최고 Val Accuracy: {best_val_acc:.4f}")
    print(f"모델 저장 경로: {CONFIG['model_save_path']}")


if __name__ == "__main__":
    main()
