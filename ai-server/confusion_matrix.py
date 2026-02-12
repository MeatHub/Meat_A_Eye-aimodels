import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# ==========================================
# 1. 설정 (기존 Train 설정과 동일하게 맞춤)
# ==========================================
CONFIG = {
    "test_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\pork_final\test"),
    "model_path": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\models_each\meat_vision_b2_pork.pth"),
    "image_size": 260,
    "batch_size": 32,
}

# 검증용 변환 (학습 때와 동일)
test_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

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

def plot_confusion_matrix():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로드
    test_raw = datasets.ImageFolder(root=CONFIG["test_dir"])
    class_names = test_raw.classes
    test_loader = DataLoader(AlbumentationsDataset(test_raw, test_transform), 
                             batch_size=CONFIG["batch_size"], shuffle=False)

    # 2. 모델 로드
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model = model.to(device)
    model.eval()

    # 3. 예측 수행
    all_preds = []
    all_labels = []
    
    print("🧐 테스트 데이터 분석 중...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. 혼동 행렬 생성
    cm = confusion_matrix(all_labels, all_preds)
    
    # 5. 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted (Model)', fontsize=12)
    plt.ylabel('Actual (True)', fontsize=12)
    plt.title('Pork Cuts Classification - Confusion Matrix', fontsize=15)
    
    # 결과 저장 및 출력
    save_path = Path(CONFIG["model_path"]).parent / "confusion_matrix_pork.png"
    plt.savefig(save_path)
    print(f"📊 혼동 행렬 이미지가 저장되었습니다: {save_path}")
    
    # 상세 레포트 출력 (Precision, Recall, F1-score)
    print("\n📝 상세 분류 레포트:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix()