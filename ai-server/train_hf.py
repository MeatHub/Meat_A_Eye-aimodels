import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from transformers import AutoModelForImageClassification

# ==========================================
# 1. 설정 (모델명만 B2로 변경)
# ==========================================
CONFIG = {
    "train_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\train"),
    "val_dir":   Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\val"),
    "test_dir":  Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\test"),
    
    # ★ HF에 등록된 EfficientNet-B2 모델 (google 공식)
    "hf_model_name": "google/efficientnet-b2", 
    "model_save_path": Path(__file__).parent / "models" / "models_b2" / "meat_vision_b2_hf.pth",
    
    "num_epochs": 10,
    "batch_size": 32,        # B2는 ViT보다 가벼워서 배치를 더 크게 잡으셔도 됩니다.
    "learning_rate": 1e-4,   # CNN은 ViT보다 조금 더 공격적으로 배워도 됩니다.
    "image_size": 260,       # B2의 표준 해상도
    "num_workers": 0,
}

# 

# ==========================================
# 2. 데이터 증강 (B2 최적화)
# ==========================================
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.CLAHE(clip_limit=4.0, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 표준
    ToTensorV2(),
])

# (AlbumentationsDataset, evaluate 함수는 ViT 때와 동일하므로 생략 가능하나 전체 코드 위해 유지)
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform: image = self.transform(image=image)["image"]
        return image, label

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total

# ==========================================
# 3. 메인 실행
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 B2-HF Meat Vision Start | Device: {device}")

    train_raw = datasets.ImageFolder(root=CONFIG["train_dir"])
    val_raw = datasets.ImageFolder(root=CONFIG["val_dir"])
    test_raw = datasets.ImageFolder(root=CONFIG["test_dir"])
    num_classes = len(train_raw.classes)

    train_loader = DataLoader(AlbumentationsDataset(train_raw, train_transform), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(AlbumentationsDataset(val_raw, train_transform), batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(AlbumentationsDataset(test_raw, train_transform), batch_size=CONFIG["batch_size"], shuffle=False)

    # 모델 로드 (B2 모델을 HF 방식으로 불러옴)
    model = AutoModelForImageClassification.from_pretrained(
        CONFIG["hf_model_name"], 
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    if CONFIG["model_save_path"].exists():
        print("🔄 기존 B2 가중치를 이어서 학습합니다.")
        model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        v_acc = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/10] Val Acc: {v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print("  🔥 B2-HF Best Saved!")

    # 최종 결과 출력
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    test_acc = evaluate(model, test_loader, device)
    print(f"🏆 B2-HF 최종 수능 점수: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()