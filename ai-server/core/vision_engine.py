"""
Meat-A-Eye EfficientNet-B2 Vision Engine
17클래스 고기 부위 분류 + Grad-CAM 히트맵
백엔드 PART_TO_CODES와 17개 영문 클래스명 1:1 대응
"""
import base64
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Dict, List, Optional
from pathlib import Path
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[VisionEngine] 경고: opencv 미설치. 히트맵 비활성화")

# ---------------------------------------------------------------------------
# B2 17클래스 (dataset_final 폴더명 알파벳순 = ImageFolder 순서)
# ---------------------------------------------------------------------------
B2_17_CLASSES: List[str] = [
    "Beef_BottomRound", "Beef_Brisket", "Beef_Chuck", "Beef_Rib",
    "Beef_Ribeye", "Beef_Round", "Beef_Shank", "Beef_Shoulder",
    "Beef_Sirloin", "Beef_Tenderloin",
    "Pork_Belly", "Pork_Ham", "Pork_Loin", "Pork_Neck",
    "Pork_PicnicShoulder", "Pork_Ribs", "Pork_Tenderloin",
]

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
MODEL_INPUT_SIZE = 260
CONFIDENCE_THRESHOLD = 0.5

MODEL_DIR = Path(__file__).parent.parent / "models" / "models_b2"
MODEL_CANDIDATES = [
    "meat_vision_b2_dataset.pth",   # train_new.py 17클래스 학습 결과 (우선)
    "meat_vision_b2_new.pth",
    "meat_vision_b2_hard.pth",
    "meat_vision_b2_hard_test.pth",
]

# 추론 전처리 (학습 시 val_transform과 동일)
_transform = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()


# ---------------------------------------------------------------------------
# 모델 래퍼
# ---------------------------------------------------------------------------
class MeatVisionB2:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes: List[str] = []
        self.model: Optional[nn.Module] = None
        self.grad_cam: Optional[GradCAM] = None
        self._load()

    def _find_model_path(self) -> Optional[Path]:
        for name in MODEL_CANDIDATES:
            p = MODEL_DIR / name
            if p.exists():
                return p
        return None

    def _load(self):
        model_path = self._find_model_path()

        if model_path is None:
            print("[VisionEngine] 경고: B2 모델 파일 없음. pretrained 가중치 사용")
            self.classes = B2_17_CLASSES
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.classes))
            self.model = model.to(self.device).eval()
            self.grad_cam = GradCAM(self.model, self.model.features[-1])
            return

        # 가중치에서 클래스 수 자동 감지
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=self.device)

        num_classes = state_dict["classifier.1.weight"].shape[0]
        print(f"[VisionEngine] 모델 로드: {model_path.name} ({num_classes}클래스, device={self.device})")

        # 클래스 리스트 결정
        if num_classes == len(B2_17_CLASSES):
            self.classes = B2_17_CLASSES
        elif num_classes <= len(B2_17_CLASSES):
            self.classes = B2_17_CLASSES[:num_classes]
        else:
            self.classes = [f"class_{i}" for i in range(num_classes)]

        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()
        self.grad_cam = GradCAM(self.model, self.model.features[-1])

    def predict(self, pil_image: Image.Image, generate_heatmap: bool = True) -> Dict:
        """
        고기 부위 예측 + (선택) Grad-CAM 히트맵

        Returns:
            {status, class_name, confidence, heatmap_image}
        """
        input_tensor = _transform(pil_image).unsqueeze(0).to(self.device)

        if generate_heatmap and self.grad_cam and CV2_AVAILABLE:
            # Grad-CAM: gradient 필요하므로 no_grad 사용 안 함
            self.model.zero_grad()
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
            confidence = conf.item()
            class_idx = idx.item()

            # backward로 Grad-CAM 생성
            output[0, class_idx].backward()

            weights = self.grad_cam.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.grad_cam.activations).sum(dim=1).squeeze().cpu().numpy()
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()

            heatmap_b64 = self._heatmap_to_base64(pil_image, cam)
        else:
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, idx = torch.max(probs, dim=1)
                confidence = conf.item()
                class_idx = idx.item()
            heatmap_b64 = None

        class_name = self.classes[class_idx] if class_idx < len(self.classes) else "unknown"

        return {
            "status": "success",
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "heatmap_image": heatmap_b64,
        }

    def _heatmap_to_base64(self, pil_image: Image.Image, heatmap: np.ndarray) -> Optional[str]:
        try:
            img_resized = pil_image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
            img_arr = np.array(img_resized)[:, :, ::-1]  # RGB → BGR

            heatmap_resized = cv2.resize(heatmap, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            blended = cv2.addWeighted(img_arr, 0.6, heatmap_color, 0.4, 0)

            _, buffer = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print(f"[VisionEngine] 히트맵 base64 변환 실패: {e}")
            return None


# ---------------------------------------------------------------------------
# 싱글톤 & API 함수
# ---------------------------------------------------------------------------
_instance: Optional[MeatVisionB2] = None


def get_model() -> MeatVisionB2:
    global _instance
    if _instance is None:
        _instance = MeatVisionB2()
    return _instance


def predict_part(img_array: np.ndarray) -> Dict:
    """
    기존 /ai/analyze vision 호환용

    Returns:
        {label, label_en, animal, score, is_valid}
    """
    pil_image = Image.fromarray(img_array)
    model = get_model()
    result = model.predict(pil_image, generate_heatmap=False)
    backend_name = result["class_name"]
    return {
        "label": backend_name,
        "label_en": backend_name,
        "animal": "beef" if backend_name.startswith("Beef") else "pork",
        "score": result["confidence"],
        "is_valid": result["confidence"] >= CONFIDENCE_THRESHOLD,
    }


def predict_for_api(img_array: np.ndarray) -> Dict:
    """
    POST /predict 엔드포인트용

    Returns:
        {status, class_name, confidence, heatmap_image}
    """
    pil_image = Image.fromarray(img_array)
    model = get_model()
    return model.predict(pil_image, generate_heatmap=True)
