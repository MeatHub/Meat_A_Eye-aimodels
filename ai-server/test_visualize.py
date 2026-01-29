import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    # ★ 실제 테스트할 이미지가 들어있는 폴더
    "test_images_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\test_images"),
    
    # ★ 방금 학습 완료된 모델 가중치 경로
    "model_path": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\models_b2\meat_vision_b2_hard.pth"),
    
    # ★ 결과 이미지가 저장될 폴더
    "result_save_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\test_results"),
    
    # 학습된 클래스 리스트 (순서가 정확해야 합니다)
    "class_names": [
        'Beef_BottomRound', 'Beef_Brisket', 'Beef_Chuck', 'Beef_Rib', 
        'Beef_Round', 'Beef_Shank', 'Beef_Shoulder', 'Beef_Sirloin', 'Beef_Tenderloin'
    ],
    
    "image_size": 260,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 결과 저장 폴더 생성
os.makedirs(CONFIG["result_save_dir"], exist_ok=True)

# ==========================================
# 2. Grad-CAM 클래스
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.output = None
        
        # 훅(Hook) 등록
        target_layer.register_forward_hook(self.save_output)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_output(self, module, input, output):
        self.output = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # 특정 클래스에 대한 역전파
        loss = output[0, class_idx]
        loss.backward()
        
        # 가중치 계산 및 히트맵 생성
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.output, dim=1).squeeze()
        
        # ReLU 적용 및 정규화
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-7)
        return heatmap

# ==========================================
# 3. 테스트 실행 로직
# ==========================================
def run_visualization():
    print(f"🚀 시각화 테스트 시작 (Device: {CONFIG['device']})")
    
    # 1. 모델 로드
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CONFIG["class_names"]))
    
    if not CONFIG["model_path"].exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {CONFIG['model_path']}")
        return
        
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.to(CONFIG["device"]).eval()

    # 2. Grad-CAM 준비 (EfficientNet-B2의 마지막 특징 맵 추출 층)
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    # 3. 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. 이미지 파일 목록 가져오기
    image_paths = [p for p in CONFIG["test_images_dir"].glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    print(f"🔎 {len(image_paths)}개의 이미지를 발견했습니다.")

    for img_path in image_paths:
        # 이미지 읽기
        raw_img = cv2.imread(str(img_path))
        if raw_img is None: continue
        
        # 전처리
        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(CONFIG["device"])
        
        # 추론
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probabilities, 1)
        
        pred_label = CONFIG["class_names"][pred_idx.item()]
        confidence = conf.item() * 100
        
        # 히트맵 생성
        heatmap = grad_cam.generate(input_tensor, pred_idx.item())
        
        # --- 시각화 가공 ---
        # 1. 원본을 정사각형으로 리사이즈 (모델이 본 시점과 일치시킴)
        view_img = cv2.resize(raw_img, (CONFIG["image_size"], CONFIG["image_size"]))
        
        # 2. 히트맵을 이미지 크기에 맞게 키우고 색상 입히기
        heatmap_resize = cv2.resize(heatmap, (CONFIG["image_size"], CONFIG["image_size"]))
        heatmap_uint8 = np.uint8(255 * heatmap_resize)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # 3. 원본과 히트맵 합성 (원본 60%, 히트맵 40%)
        blended = cv2.addWeighted(view_img, 0.6, heatmap_color, 0.4, 0)
        
        # 4. 상단 여백 추가 (100px 검은색 바) - 텍스트 잘림 방지
        header_h = 100
        result_img = cv2.copyMakeBorder(blended, header_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # 5. 텍스트 2줄로 나누어 작성 (가로 잘림 방지)
        txt_color = (0, 255, 0) # 초록색
        text_1 = f"PRED: {pred_label}"
        text_2 = f"CONF: {confidence:.2f}%"
        
        # 첫 번째 줄
        cv2.putText(result_img, text_1, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        # 두 번째 줄
        cv2.putText(result_img, text_2, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2, cv2.LINE_AA)

        # 6. 결과 저장
        save_path = CONFIG["result_save_dir"] / f"vis_{img_path.name}"
        cv2.imwrite(str(save_path), result_img)
        print(f"✅ 저장 완료: {save_path.name} ({pred_label})")

    print(f"\n✨ 모든 결과가 '{CONFIG['result_save_dir']}'에 저장되었습니다.")

if __name__ == "__main__":
    run_visualization()