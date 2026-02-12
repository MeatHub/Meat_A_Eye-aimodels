import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm

# 1. 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\b2_imagenet_beef_100-v4.pth"
TEST_IMAGE_DIR = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\data\train_dataset_0\test"  # 부위별 서브폴더 구조
RESULT_DIR = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\test_results_v0"

CLASS_NAMES = ['Beef_Brisket', 'Beef_Chuck', 'Beef_Rib', 'Beef_Ribeye', 'Beef_Round', 'Beef_Shank', 'Beef_Shoulder', 'Beef_Sirloin', 'Beef_Tenderloin']
# 10클래스 모델용 병합 매핑 (9클래스 모델에서는 사용하지 않음)
CLASS_MERGE_MAP = {"Beef_BottomRound": "Beef_Round"}
IMAGE_SIZE = 260

os.makedirs(RESULT_DIR, exist_ok=True)


def collect_test_images(base_dir):
    """부위별 서브폴더에서 모든 이미지 수집 (경로, 정답 클래스)"""
    images = []
    for class_name in CLASS_NAMES:
        # Beef_BottomRound 폴더도 수집하되 정답은 Beef_Round로 병합
        mapped_name = CLASS_MERGE_MAP.get(class_name, class_name)
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            for img_path in glob.glob(os.path.join(class_dir, ext)):
                images.append((img_path, mapped_name))
    return images

# Grad-CAM 클래스 (B2 대응)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.output = None
        self.target_layer.register_forward_hook(self.save_output)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_output(self, module, input, output): self.output = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.output, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap

def load_b2_model(num_classes):
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# 모델 및 전처리 설정
model = load_b2_model(len(CLASS_NAMES))
target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_visual_test():
    # 부위별 서브폴더에서 이미지 수집
    image_list = collect_test_images(TEST_IMAGE_DIR)
    print(f"\n🚀 [분석 시작] 총 {len(image_list)}개의 이미지 검증 중...")
    print(f"📂 테스트 폴더: {TEST_IMAGE_DIR}")
    print("-" * 90)
    print(f"{'파일명':<35} | {'실제 정답':<18} | {'모델 예측':<18} | {'신뢰도':<8} | {'결과'}")
    print("-" * 90)

    correct_count = 0
    total_count = 0
    
    # 클래스별 통계
    class_stats = {name: {"correct": 0, "total": 0, "wrong_preds": []} for name in CLASS_NAMES}
    
    # 클래스별 결과 폴더 생성
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(RESULT_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "_wrong"), exist_ok=True)  # 오답 모음

    for img_path, ground_truth in image_list:
        filename = os.path.basename(img_path)
        raw_img = cv2.imread(img_path)
        if raw_img is None: continue

        # 전처리 및 추론
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(raw_img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(DEVICE)
        
        with torch.set_grad_enabled(True):
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            
            # 10클래스 모델일 때만 설도→우둔 확률 병합
            if len(CLASS_NAMES) == 10:
                prob[0, 5] += prob[0, 0]  # Beef_Round += Beef_BottomRound
                prob[0, 0] = 0
            
            conf, pred = torch.max(prob, 1)
            class_idx = pred.item()
            pred_label = CLASS_NAMES[class_idx]
            pred_label = CLASS_MERGE_MAP.get(pred_label, pred_label)  # 병합 매핑
            confidence = conf.item()

        # 정확도 계산
        is_correct = (pred_label == ground_truth)
        if is_correct: 
            correct_count += 1
            class_stats[ground_truth]["correct"] += 1
        else:
            class_stats[ground_truth]["wrong_preds"].append((filename, pred_label, confidence))
        
        class_stats[ground_truth]["total"] += 1
        total_count += 1
        result_mark = "✅" if is_correct else "❌"

        # 히트맵 생성 및 합성
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)
        heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap_img = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        
        # [원본 | 열지도 | 합성본] 가로로 붙이기
        overlaid = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
        combined_view = np.hstack((img_resized, heatmap_color, overlaid))
        
        # 결과 이미지 상단에 정보 텍스트 추가
        info_bar = np.zeros((50, combined_view.shape[1], 3), dtype=np.uint8)
        text = f"True: {ground_truth} | Pred: {pred_label} ({confidence*100:.1f}%)"
        color = (0, 255, 0) if is_correct else (255, 0, 0)
        cv2.putText(info_bar, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        final_report = np.vstack((info_bar, combined_view))

        # 저장 (클래스별 폴더 + 오답은 _wrong 폴더에도 저장)
        save_path = os.path.join(RESULT_DIR, ground_truth, f"report_{filename}")
        cv2.imwrite(save_path, cv2.cvtColor(final_report, cv2.COLOR_RGB2BGR))
        
        if not is_correct:
            wrong_path = os.path.join(RESULT_DIR, "_wrong", f"{ground_truth}_to_{pred_label}_{filename}")
            cv2.imwrite(wrong_path, cv2.cvtColor(final_report, cv2.COLOR_RGB2BGR))
        
        print(f"{filename[:35]:<35} | {ground_truth:<18} | {pred_label:<18} | {confidence*100:>6.1f}% | {result_mark}")

    # 클래스별 정확도 요약
    print("\n" + "=" * 90)
    print("📊 [클래스별 정확도]")
    print("=" * 90)
    print(f"{'클래스':<22} | {'맞춤':>6} | {'전체':>6} | {'정확도':>10} | {'주요 오분류'}")
    print("-" * 90)
    
    for name in CLASS_NAMES:
        stats = class_stats[name]
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        
        # 주요 오분류 분석
        wrong_summary = ""
        if stats["wrong_preds"]:
            wrong_classes = {}
            for _, wrong_pred, _ in stats["wrong_preds"]:
                wrong_classes[wrong_pred] = wrong_classes.get(wrong_pred, 0) + 1
            top_wrong = sorted(wrong_classes.items(), key=lambda x: -x[1])[:2]
            wrong_summary = ", ".join([f"{k}({v})" for k, v in top_wrong])
        
        acc_bar = "█" * int(acc // 10) + "░" * (10 - int(acc // 10))
        print(f"{name:<22} | {stats['correct']:>6} | {stats['total']:>6} | {acc:>6.1f}% {acc_bar} | {wrong_summary}")

    # 최종 요약
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("=" * 90)
    print(f"📊 [최종 결과] 전체: {total_count} | 맞춤: {correct_count} | 틀림: {total_count-correct_count}")
    print(f"🎯 최종 정확도(Accuracy): {accuracy:.2f}%")
    print("=" * 90)
    print(f"📂 상세 리포트 저장 위치: {RESULT_DIR}")
    print(f"   - 클래스별 폴더: 각 부위별 결과")
    print(f"   - _wrong 폴더: 오답만 모아보기")

if __name__ == "__main__":
    run_visual_test()