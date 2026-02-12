import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

# ==========================================
# 0. 한글 경로 지원을 위한 헬퍼 함수
# ==========================================
def imread_kor(path):
    """한글 경로의 이미지를 읽어옵니다."""
    try:
        img_array = np.fromfile(str(path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ 이미지 읽기 실패: {path} ({e})")
        return None

def imwrite_kor(path, img):
    """이미지를 한글 경로에 저장합니다."""
    try:
        ext = os.path.splitext(path)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            with open(path, mode='w+b') as f:
                n.tofile(f)
    except Exception as e:
        print(f"❌ 이미지 저장 실패: {path} ({e})")

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    # [중요] 테스트할 이미지가 들어있는 최상위 폴더 (pork_final/test 경로로 지정)
    "test_root_dirs": [
        Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\Pork_Test2")
    ],
    
    # 학습된 모델 경로
    "model_path": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\models_each\meat_vision_b2_pork.pth"),
    
    # 결과가 저장될 폴더
    "result_save_dir": Path(r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\pork_results"),
    
    # 클래스 이름 (폴더 이름과 정확히 일치해야 함)
    "class_names": sorted([
        'Pork_Belly', 'Pork_Ham', 'Pork_Loin', 'Pork_Neck', 'Pork_PicnicShoulder',
        'Pork_Ribs', 'Pork_Tenderloin'
    ]),
    
    "image_size": 260,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

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
        self.target_layer.register_forward_hook(self.save_output)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_output(self, module, input, output): self.output = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.output, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        
        # heatmap 정규화 (0~1)
        heatmap = heatmap / (np.max(heatmap) + 1e-7)
        return heatmap

# ==========================================
# 3. 테스트 실행 로직
# ==========================================
def run_integrated_visualization():
    print(f"🚀 [폴더명 기준] 시각화 테스트 시작 (Device: {CONFIG['device']})")
    
    # 1. 모델 로드 및 초기화
    try:
        model = models.efficientnet_b2(weights=None)
        # 마지막 레이어 수정 (클래스 개수에 맞게)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CONFIG["class_names"]))
        
        if not CONFIG["model_path"].exists():
            print(f"❌ 모델 파일을 찾을 수 없습니다: {CONFIG['model_path']}")
            return

        model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
        model.to(CONFIG["device"]).eval()
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 중 에러 발생: {e}")
        return

    # Grad-CAM 설정 (EfficientNet의 마지막 컨볼루션 레이어)
    grad_cam = GradCAM(model, model.features[-1])
    
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 이미지 경로 수집
    image_paths = []
    for root in CONFIG["test_root_dirs"]:
        if root.exists():
            # 모든 하위 폴더를 뒤져서 이미지 파일 찾기
            found = list(root.rglob("*"))
            image_paths.extend([p for p in found if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp", ".webp"]])
    
    print(f"🔎 총 {len(image_paths)}개의 테스트 이미지를 발견했습니다.")
    
    if not image_paths:
        print("⚠️ 처리할 이미지가 없습니다. 경로를 확인해주세요.")
        return

    # 3. 개별 이미지 처리
    success_count = 0
    
    for img_path in image_paths:
        # 🔥 [핵심 수정] 정답(Ground Truth)을 "상위 폴더 이름"에서 가져옵니다.
        # 예: .../test/Pork_Belly/image.jpg -> folder_name = "Pork_Belly"
        folder_name = img_path.parent.name
        
        # 폴더 이름이 우리가 아는 클래스 목록에 있는지 확인
        if folder_name in CONFIG["class_names"]:
            ground_truth = folder_name
        else:
            # 폴더명이 클래스 목록에 없다면 (예: test 폴더 바로 아래에 파일이 있는 경우 등)
            print(f"⚠️ 경고: '{img_path.name}'의 폴더명({folder_name})이 클래스 목록에 없습니다. 건너뜁니다.")
            continue

        # 이미지 읽기
        raw_img = imread_kor(img_path)
        if raw_img is None:
            continue
        
        # 이미지 전처리
        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(CONFIG["device"])
        
        # 추론 (Prediction)
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probabilities, 1)
        
        pred_label = CONFIG["class_names"][pred_idx.item()]
        confidence = conf.item() * 100
        
        # Grad-CAM Heatmap 생성
        heatmap = grad_cam.generate(input_tensor, pred_idx.item())
        
        # 시각화 합성
        view_img = cv2.resize(raw_img, (CONFIG["image_size"], CONFIG["image_size"]))
        heatmap_resize = cv2.resize(heatmap, (CONFIG["image_size"], CONFIG["image_size"]))
        
        # 히트맵 컬러 입히기 (파란색 -> 빨간색)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resize), cv2.COLORMAP_JET)
        
        # 원본 이미지와 겹치기 (투명도 조절)
        blended = cv2.addWeighted(view_img, 0.6, heatmap_color, 0.4, 0)
        
        # 결과 이미지 생성 (하단에 검은색 바 추가)
        result_img = cv2.copyMakeBorder(blended, 0, 120, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # 정답 여부에 따른 텍스트 색상 결정
        is_correct = (pred_label == ground_truth)
        status_color = (0, 255, 0) if is_correct else (0, 0, 255) # 맞으면 초록, 틀리면 빨강
        
        # 텍스트 정보 입력
        # (좌표는 이미지 크기에 따라 유동적일 수 있으나 여기선 고정값 사용)
        base_y = CONFIG["image_size"] + 25
        cv2.putText(result_img, f"GT   : {ground_truth}", (10, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result_img, f"PRED : {pred_label}", (10, base_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(result_img, f"CONF : {confidence:.2f}%", (10, base_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # 파일명 한글 깨짐 방지를 위해 영어/숫자로만 저장하거나 안전한 이름 사용
        # 예: O_Pork_Belly_image123.jpg
        safe_filename = img_path.name
        save_name = f"{'O' if is_correct else 'X'}_{ground_truth}_{safe_filename}"
        save_path = str(CONFIG["result_save_dir"] / save_name)
        
        imwrite_kor(save_path, result_img)
        success_count += 1

    print(f"\n✨ 분석 완료! 총 {success_count}장의 결과 이미지가 저장되었습니다.")
    print(f"📍 결과 위치: {CONFIG['result_save_dir']}")

if __name__ == "__main__":
    run_integrated_visualization()