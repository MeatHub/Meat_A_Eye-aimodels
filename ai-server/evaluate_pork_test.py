"""
돼지 등심·안심 테스트 세트 정확도 및 혼동 행렬 분석
"""
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from collections import Counter

def confusion_matrix(y_true, y_pred, labels=None):
    """numpy로 구현한 혼동 행렬"""
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1
    return cm

DEVICE = torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "meat_vision_b2_pro.pth"
TEST_IMAGE_DIR = BASE_DIR.parent / "data" / "dataset_final" / "test"

CLASS_NAMES = [
    'Beef_BottomRound', 'Beef_Brisket', 'Beef_Chuck', 'Beef_Rib', 'Beef_Ribeye',
    'Beef_Round', 'Beef_Shank', 'Beef_Shoulder', 'Beef_Sirloin', 'Beef_Tenderloin',
    'Pork_Loin', 'Pork_Tenderloin'
]

def create_model_b2(num_classes):
    model = models.efficientnet_b2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    return model

def evaluate_pork_test():
    print(f"🐷 돼지 등심·안심 테스트 세트 평가 (장비: {DEVICE})\n")
    
    # 데이터셋 로드
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(root=str(TEST_IMAGE_DIR), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    num_classes = len(test_dataset.classes)
    print(f"✅ 전체 클래스 수: {num_classes}개")
    print(f"   클래스 목록: {test_dataset.classes}\n")
    
    # 모델 로드
    print("📥 모델 로드 중...")
    model = create_model_b2(num_classes).to(DEVICE)
    state_dict = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 모델 로드 완료\n")
    
    # 예측 수집
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("📊 예측 중...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 전체 정확도
    accuracy = (all_preds == all_labels).mean()
    print(f"✅ 예측 완료\n")
    
    print(f"{'='*70}")
    print(f"📊 전체 Test 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # 돼지 등심·안심 특화 분석
    pork_loin_idx = test_dataset.class_to_idx.get('Pork_Loin', -1)
    pork_tenderloin_idx = test_dataset.class_to_idx.get('Pork_Tenderloin', -1)
    
    print(f"{'='*70}")
    print(f"🐷 돼지 등심·안심 상세 분석")
    print(f"{'='*70}\n")
    
    if pork_loin_idx >= 0:
        loin_mask = all_labels == pork_loin_idx
        loin_total = loin_mask.sum()
        loin_correct = (all_preds[loin_mask] == pork_loin_idx).sum()
        loin_acc = loin_correct / loin_total if loin_total > 0 else 0
        
        # 평균 confidence
        loin_probs = all_probs[loin_mask][:, pork_loin_idx]
        avg_conf = loin_probs.mean() * 100
        
        print(f"🥩 돼지 등심 (Pork_Loin):")
        print(f"   ✅ 정확도: {loin_acc:.4f} ({loin_acc*100:.2f}%)")
        print(f"   📊 정답/전체: {loin_correct}/{loin_total}")
        print(f"   💯 평균 신뢰도: {avg_conf:.2f}%\n")
        
        # 등심이 다른 클래스로 잘못 예측된 경우
        loin_wrong = all_preds[loin_mask] != pork_loin_idx
        if loin_wrong.any():
            wrong_preds = all_preds[loin_mask][loin_wrong]
            wrong_probs = all_probs[loin_mask][loin_wrong]
            wrong_classes = [test_dataset.classes[p] for p in wrong_preds]
            wrong_confidences = [wrong_probs[i][wrong_preds[i]] * 100 for i in range(len(wrong_preds))]
            wrong_counts = Counter(wrong_classes)
            
            print(f"   ❌ 잘못 예측된 경우 ({loin_wrong.sum()}개):")
            for cls, count in wrong_counts.most_common():
                print(f"      - {cls}: {count}개")
            print()
        else:
            print(f"   ✨ 모든 등심 이미지가 정확히 분류되었습니다!\n")
    
    if pork_tenderloin_idx >= 0:
        tenderloin_mask = all_labels == pork_tenderloin_idx
        tenderloin_total = tenderloin_mask.sum()
        tenderloin_correct = (all_preds[tenderloin_mask] == pork_tenderloin_idx).sum()
        tenderloin_acc = tenderloin_correct / tenderloin_total if tenderloin_total > 0 else 0
        
        # 평균 confidence
        tenderloin_probs = all_probs[tenderloin_mask][:, pork_tenderloin_idx]
        avg_conf = tenderloin_probs.mean() * 100
        
        print(f"🥩 돼지 안심 (Pork_Tenderloin):")
        print(f"   ✅ 정확도: {tenderloin_acc:.4f} ({tenderloin_acc*100:.2f}%)")
        print(f"   📊 정답/전체: {tenderloin_correct}/{tenderloin_total}")
        print(f"   💯 평균 신뢰도: {avg_conf:.2f}%\n")
        
        # 안심이 다른 클래스로 잘못 예측된 경우
        tenderloin_wrong = all_preds[tenderloin_mask] != pork_tenderloin_idx
        if tenderloin_wrong.any():
            wrong_preds = all_preds[tenderloin_mask][tenderloin_wrong]
            wrong_probs = all_probs[tenderloin_mask][tenderloin_wrong]
            wrong_classes = [test_dataset.classes[p] for p in wrong_preds]
            wrong_confidences = [wrong_probs[i][wrong_preds[i]] * 100 for i in range(len(wrong_preds))]
            wrong_counts = Counter(wrong_classes)
            
            print(f"   ❌ 잘못 예측된 경우 ({tenderloin_wrong.sum()}개):")
            for cls, count in wrong_counts.most_common():
                print(f"      - {cls}: {count}개")
            print()
        else:
            print(f"   ✨ 모든 안심 이미지가 정확히 분류되었습니다!\n")
    
    # 혼동 행렬 (돼지 등심·안심 중심)
    print(f"{'='*70}")
    print(f"📋 돼지 등심·안심 혼동 행렬 (행=실제, 열=예측)")
    print(f"{'='*70}\n")
    
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    if pork_loin_idx >= 0:
        print(f"실제: 돼지 등심 (Pork_Loin) →")
        for j, pred_cls in enumerate(test_dataset.classes):
            count = cm[pork_loin_idx, j]
            if count > 0:
                marker = "✅" if j == pork_loin_idx else "❌"
                print(f"  {marker} 예측: {pred_cls:25s} → {count:3d}개")
        print()
    
    if pork_tenderloin_idx >= 0:
        print(f"실제: 돼지 안심 (Pork_Tenderloin) →")
        for j, pred_cls in enumerate(test_dataset.classes):
            count = cm[pork_tenderloin_idx, j]
            if count > 0:
                marker = "✅" if j == pork_tenderloin_idx else "❌"
                print(f"  {marker} 예측: {pred_cls:25s} → {count:3d}개")
        print()
    
    # 다른 클래스가 돼지 등심·안심으로 잘못 예측된 경우
    print(f"{'='*70}")
    print(f"📋 다른 클래스가 돼지 등심·안심으로 잘못 예측된 경우")
    print(f"{'='*70}\n")
    
    if pork_loin_idx >= 0:
        wrong_as_loin = (all_preds == pork_loin_idx) & (all_labels != pork_loin_idx)
        if wrong_as_loin.any():
            wrong_labels = all_labels[wrong_as_loin]
            wrong_classes = [test_dataset.classes[l] for l in wrong_labels]
            wrong_counts = Counter(wrong_classes)
            print(f"❌ 돼지 등심으로 잘못 예측된 클래스:")
            for cls, count in wrong_counts.most_common():
                print(f"   - {cls}: {count}개")
            print()
        else:
            print(f"✅ 다른 클래스가 돼지 등심으로 잘못 예측된 경우 없음\n")
    
    if pork_tenderloin_idx >= 0:
        wrong_as_tenderloin = (all_preds == pork_tenderloin_idx) & (all_labels != pork_tenderloin_idx)
        if wrong_as_tenderloin.any():
            wrong_labels = all_labels[wrong_as_tenderloin]
            wrong_classes = [test_dataset.classes[l] for l in wrong_labels]
            wrong_counts = Counter(wrong_classes)
            print(f"❌ 돼지 안심으로 잘못 예측된 클래스:")
            for cls, count in wrong_counts.most_common():
                print(f"   - {cls}: {count}개")
            print()
        else:
            print(f"✅ 다른 클래스가 돼지 안심으로 잘못 예측된 경우 없음\n")

if __name__ == "__main__":
    evaluate_pork_test()
