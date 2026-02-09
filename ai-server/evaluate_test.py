"""
저장된 모델로 dataset_final/test 정확도 측정 (전체 + 클래스별, 돼지 등심·안심 상세)
"""
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

DEVICE = torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "meat_vision_b2_pro.pth"
TEST_IMAGE_DIR = BASE_DIR.parent / "data" / "real_test"
IMAGE_SIZE = 260


def create_model_b2(num_classes):
    model = models.efficientnet_b2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model


def main():
    print(f"📁 테스트 데이터: {TEST_IMAGE_DIR}")
    print(f"📥 모델: {MODEL_PATH}\n")

    if not TEST_IMAGE_DIR.exists():
        print(f"❌ 테스트 폴더 없음: {TEST_IMAGE_DIR}")
        return
    if not MODEL_PATH.exists():
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(root=str(TEST_IMAGE_DIR), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    num_classes = len(test_dataset.classes)
    model = create_model_b2(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for i in range(labels.size(0)):
                c = labels[i].item()
                class_total[c] += 1
                if preds[i].item() == c:
                    class_correct[c] += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    print("=" * 60)
    print(f"📊 전체 Test 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   정답/전체: {(all_preds == all_labels).sum()}/{len(all_labels)}")
    print("=" * 60)

    print("\n📋 클래스별 정확도:")
    for i, class_name in enumerate(test_dataset.classes):
        total = class_total.get(i, 0)
        correct = class_correct.get(i, 0)
        acc = correct / total if total > 0 else 0
        print(f"   {class_name:25s} {acc:.4f} ({correct}/{total})")

    # 돼지 등심·안심 상세
    for name in ("Pork_Loin", "Pork_Tenderloin"):
        if name not in test_dataset.class_to_idx:
            continue
        idx = test_dataset.class_to_idx[name]
        mask = all_labels == idx
        if mask.sum() == 0:
            continue
        correct = (all_preds[mask] == idx).sum()
        total = mask.sum()
        wrong_preds = all_preds[mask][all_preds[mask] != idx]
        wrong_classes = [test_dataset.classes[p] for p in wrong_preds]
        wrong_counts = Counter(wrong_classes)
        print(f"\n🐷 {name}: 정확도 {correct}/{total} ({correct/total:.4f})")
        if wrong_counts:
            print(f"   잘못 예측된 경우: {dict(wrong_counts)}")

    print("\n✅ 평가 완료.")


if __name__ == "__main__":
    main()
