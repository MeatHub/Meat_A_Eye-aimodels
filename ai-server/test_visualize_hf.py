import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification
from torchvision import transforms
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    # 테스트할 이미지가 담긴 폴더
    "test_images_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\0205dataset"),
    # 방금 학습한 B2-HF 가중치 경로
    "model_path": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\models_b2\meat_vision_b2_hf.pth"),
    # 결과가 저장될 폴더
    "result_save_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\test0205"),
    
    "hf_model_name": "google/efficientnet-b2",
    
    "class_names": [
        'Beef_BottomRound', 'Beef_Brisket', 'Beef_Chuck', 'Beef_Rib', 'Beef_Ribeye',
        'Beef_Round', 'Beef_Shank', 'Beef_Shoulder', 'Beef_Sirloin', 'Beef_Tenderloin'
    ],
    
    "image_size": 260, # B2 표준 크기
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

os.makedirs(CONFIG["result_save_dir"], exist_ok=True)

# ==========================================
# 2. Grad-CAM 클래스 (B2-HF 전용)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.output = None
        
        # 훅(Hook) 등록
        self.target_layer.register_forward_hook(self.save_output)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_output(self, module, input, output):
        self.output = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        outputs = self.model(input_tensor).logits
        loss = outputs[0, class_idx]
        loss.backward()

        # 가중치 계산 (GAP)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # 히트맵 생성
        cam = torch.sum(weights * self.output, dim=1).squeeze()
        
        # ReLU 적용 및 정규화
        cam = np.maximum(cam.detach().cpu().numpy(), 0)
        cam = cam / (np.max(cam) + 1e-7)
        return cam

# ==========================================
# 3. 시각화 실행 (최종 수정 버전)
# ==========================================
def run_visualization():
    print(f"🚀 B2-HF 시각화 테스트 시작 (Device: {CONFIG['device']})")
    
    # 1. 모델 로드
    model = AutoModelForImageClassification.from_pretrained(
        CONFIG["hf_model_name"], 
        num_labels=len(CONFIG["class_names"]),
        ignore_mismatched_sizes=True
    )
    
    # 학습된 가중치 로드
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.to(CONFIG["device"]).eval()

    # ---------------------------------------------------------
    # ★ [수정 포인트] Hugging Face EfficientNet의 최종 레이어 경로
    # ---------------------------------------------------------
    # model (EfficientNetForImageClassification)
    #  └─ efficientnet (EfficientNetModel)
    #      └─ encoder (EfficientNetEncoder)
    #          └─ blocks (nn.ModuleList) <- 이 안의 마지막 블록을 선택합니다.
    try:
        target_layer = model.efficientnet.encoder.blocks[-1]
        print("✅ 타겟 레이어(encoder.blocks[-1]) 설정 완료!")
    except AttributeError:
        # 혹시라도 구조가 다를 경우를 대비해 하위 모듈을 출력해봅니다.
        print("❌ 레이어 구조가 예상과 다릅니다. 아래 구조를 확인하세요:")
        print(model)
        return
    # ---------------------------------------------------------
    
    grad_cam = GradCAM(model, target_layer)

    # ... (이후 전처리 및 시각화 로직은 기존과 동일)

    # ... (이하 전처리 및 루프 로직은 동일)

    # 3. 전처리 (학습 때와 동일하게)
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_paths = [p for p in CONFIG["test_images_dir"].glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]

    for img_path in image_paths:
        raw_img = cv2.imread(str(img_path))
        if raw_img is None: continue
        
        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(CONFIG["device"])
        
        # 추론 (그래디언트 활성화 필요)
        with torch.enable_grad():
            logits = model(input_tensor).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        pred_label = CONFIG["class_names"][pred_idx.item()]
        confidence = conf.item() * 100
        
        # 히트맵 생성
        heatmap = grad_cam.generate(input_tensor, pred_idx.item())
        
        # 시각화 가공
        view_img = cv2.resize(raw_img, (CONFIG["image_size"], CONFIG["image_size"]))
        heatmap_resize = cv2.resize(heatmap, (CONFIG["image_size"], CONFIG["image_size"]))
        heatmap_uint8 = np.uint8(255 * heatmap_resize)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        blended = cv2.addWeighted(view_img, 0.6, heatmap_color, 0.4, 0)
        
        # 상단 여백 추가 (텍스트용)
        header_h = 100
        result_img = cv2.copyMakeBorder(blended, header_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # 텍스트 출력
        text_1 = f"B2-HF PRED: {pred_label}"
        text_2 = f"CONF: {confidence:.2f}%"
        cv2.putText(result_img, text_1, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(result_img, text_2, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        save_path = CONFIG["result_save_dir"] / f"b2hf_vis_{img_path.name}"
        cv2.imwrite(str(save_path), result_img)
        print(f"✅ [B2-HF] {img_path.name} -> {pred_label}")

    print(f"\n✨ 시각화 완료! 결과 확인: {CONFIG['result_save_dir']}")

if __name__ == "__main__":
    run_visualization()