import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path

# 1. 설정 (B2 최적화)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "meat_vision_b2_pro.pth"
# dataset_final/test 전체(클래스별 폴더 포함)를 시각화 대상으로 사용
TEST_IMAGE_DIR = BASE_DIR.parent / "data" / "dataset_final" / "test"
RESULT_DIR = BASE_DIR / "test_results"

# 학습 시 사용한 폴더 순서와 반드시 일치해야 합니다.
CLASS_NAMES = ['Beef_BottomRound', 'Beef_Brisket', 'Beef_Chuck', 'Beef_Rib', 'Beef_Ribeye', 'Beef_Round', 'Beef_Shank', 'Beef_Shoulder', 'Beef_Sirloin', 'Beef_Tenderloin']
IMAGE_SIZE = 260  # EfficientNet-B2 권장 입력 사이즈

os.makedirs(RESULT_DIR, exist_ok=True)

# 2. Grad-CAM 클래스 (B2 대응)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.output = None

        # Hook 등록
        self.target_layer.register_forward_hook(self.save_output)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_output(self, module, input, output):
        self.output = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # 타겟 클래스에 대한 역전파
        loss = output[0, class_idx]
        loss.backward()

        # 그래디언트 평균 계산 (Global Average Pooling)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 가중치와 특성 맵 결합
        heatmap = torch.sum(weights * self.output, dim=1).squeeze()
        
        # ReLU 적용 및 정규화
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap

# 3. B2 모델 로드 및 설정
def load_b2_model(num_classes):
    model = models.efficientnet_b2(weights=None)
    # 분류 헤드 수정 (B0와 인덱스 구조 동일)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # 가중치 로드
    state_dict = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

model = load_b2_model(len(CLASS_NAMES))

# B2의 마지막 컨볼루션 레이어 선택 (features의 마지막 블록)
target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

# B2 규격 전처리
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_visual_test():
    # test/ 하위의 모든 이미지(클래스별 폴더 포함)를 대상으로 함
    image_files = glob.glob(str(TEST_IMAGE_DIR / "**" / "*.*"), recursive=True)
    print(f"🚀 테스트 시작: {len(image_files)}개의 이미지 감지됨 (B2 엔진)")
    
    for img_path in image_files:
        # 이미지 로드 및 전처리
        raw_img = cv2.imread(img_path)
        if raw_img is None: continue
        
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(raw_img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        
        input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(DEVICE)
        
        # 1. 추론 (Inference)
        with torch.set_grad_enabled(True): # Grad-CAM을 위해 grad 활성화
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
            class_idx = pred.item()
        
        # 2. 히트맵 생성
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)
        heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 3. 원본 이미지(260px)와 히트맵 합성
        result_img = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
        
        # 결과 저장 로직
        filename = os.path.basename(img_path)
        save_path = os.path.join(RESULT_DIR, f"res_b2_{filename}")
        
        # 정보 텍스트 삽입
        label_text = CLASS_NAMES[class_idx]
        confidence_text = f"{conf.item()*100:.1f}%"
        display_text = f"{label_text} ({confidence_text})"
        
        # 가독성을 위한 텍스트 배경 처리
        cv2.rectangle(result_img, (0, 0), (260, 30), (0, 0, 0), -1)
        cv2.putText(result_img, display_text, (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # RGB에서 BGR로 다시 변경하여 저장
        cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"✅ 분석 완료 ({label_text}): {save_path}")

        del input_tensor, output, heatmap, result_img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_visual_test()