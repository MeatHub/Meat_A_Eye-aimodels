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
MODEL_PATH = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\vision_b2_imagenet.pth"
TEST_IMAGE_DIR = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\data\test_images"
RESULT_DIR = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\test_results"

CLASS_NAMES = ['Beef_BottomRound', 'Beef_Brisket', 'Beef_Chuck', 'Beef_Rib', 'Beef_Ribeye', 'Beef_Round', 'Beef_Shank', 'Beef_Shoulder', 'Beef_Sirloin', 'Beef_Tenderloin']
IMAGE_SIZE = 260

os.makedirs(RESULT_DIR, exist_ok=True)

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
    image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.*"))
    print(f"\n🚀 [분석 시작] 총 {len(image_files)}개의 이미지 검증 중...")
    print("-" * 80)
    print(f"{'파일명':<30} | {'실제 정답':<15} | {'모델 예측':<15} | {'신뢰도':<8} | {'결과'}")
    print("-" * 80)

    correct_count = 0
    total_count = 0

    for img_path in image_files:
        filename = os.path.basename(img_path)
        raw_img = cv2.imread(img_path)
        if raw_img is None: continue
        
        # 파일명에서 정답 추출 (이전 병합 로직 기반)
        ground_truth = "Unknown"
        for name in CLASS_NAMES:
            if name in filename:
                ground_truth = name
                break

        # 전처리 및 추론
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(raw_img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(DEVICE)
        
        with torch.set_grad_enabled(True):
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
            class_idx = pred.item()
            pred_label = CLASS_NAMES[class_idx]
            confidence = conf.item()

        # 정확도 계산
        is_correct = (pred_label == ground_truth)
        if is_correct: correct_count += 1
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

        # 저장 및 출력
        cv2.imwrite(os.path.join(RESULT_DIR, f"report_{filename}"), cv2.cvtColor(final_report, cv2.COLOR_RGB2BGR))
        print(f"{filename[:30]:<30} | {ground_truth:<15} | {pred_label:<15} | {confidence*100:>6.1f}% | {result_mark}")

    # 최종 요약
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("-" * 80)
    print(f"📊 [최종 결과] 전체: {total_count} | 맞춤: {correct_count} | 틀림: {total_count-correct_count}")
    print(f"🎯 최종 정확도(Accuracy): {accuracy:.2f}%")
    print("-" * 80)
    print(f"📂 상세 리포트 저장 위치: {RESULT_DIR}")

if __name__ == "__main__":
    run_visual_test()