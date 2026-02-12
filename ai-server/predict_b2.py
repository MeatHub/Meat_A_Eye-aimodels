"""
EfficientNet-B2 기반 소고기 부위 분류 + Grad-CAM 히트맵 엔진.
train.py에서 학습한 모델(vision_b2_imagenet.pth)을 로드하여 추론.
"""
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image

# ── 설정 ──────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "b2_imagenet_beef_100-v3.pth"
IMAGE_SIZE = 260

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# datasets.ImageFolder는 폴더명을 알파벳 순으로 정렬하여 클래스 인덱스를 부여
# ⚠️ 기존 10클래스 모델 호환용 (재학습 후 9클래스로 변경 필요)
MODEL_CLASS_NAMES: List[str] = [
    "Beef_BottomRound",   # 0 — 설도 (→ Beef_Round로 병합)
    "Beef_Brisket",       # 1 — 양지
    "Beef_Chuck",         # 2 — 목심
    "Beef_Rib",           # 3 — 갈비
    "Beef_Ribeye",        # 4 — 등심
    "Beef_Round",         # 5 — 우둔
    "Beef_Shank",         # 6 — 사태
    "Beef_Shoulder",      # 7 — 앞다리
    "Beef_Sirloin",       # 8 — 채끝
    "Beef_Tenderloin",    # 9 — 안심
]
NUM_CLASSES = len(MODEL_CLASS_NAMES)

# 병합 후 실제 사용하는 9클래스 (설도+우둔 → Beef_Round)
CLASS_NAMES: List[str] = [
    "Beef_Brisket",       # 양지
    "Beef_Chuck",         # 목심
    "Beef_Rib",           # 갈비
    "Beef_Ribeye",        # 등심
    "Beef_Round",         # 우둔 (설도 병합)
    "Beef_Shank",         # 사태
    "Beef_Shoulder",      # 앞다리
    "Beef_Sirloin",       # 채끝
    "Beef_Tenderloin",    # 안심
]

# 병합 매핑 (기존 10클래스 모델 호환)
CLASS_MERGE_MAP = {
    "Beef_BottomRound": "Beef_Round",  # 설도 → 우둔 병합
}


def _build_model() -> nn.Module:
    """train.py의 create_model_b2와 동일 구조로 모델 생성."""
    model = models.efficientnet_b2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, NUM_CLASSES),
    )
    return model


class GradCAM:
    """EfficientNet 마지막 conv 레이어의 Grad-CAM 히트맵 생성."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        return (cam * 255).astype(np.uint8)


class PredictEngine:
    """싱글톤 추론 엔진."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_model()
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()
        self.grad_cam = GradCAM(self.model)

    def _load_weights(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
                f"train.py를 먼저 실행하여 모델을 학습해 주세요."
            )
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        print(f"✓ 모델 로드 완료: {MODEL_PATH.name} ({NUM_CLASSES} classes)")

    def _preprocess(self, contents: bytes) -> tuple:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)
        img_resized = cv2.resize(img_np, (IMAGE_SIZE, IMAGE_SIZE))

        img_float = img_resized.astype(np.float32) / 255.0
        for c in range(3):
            img_float[:, :, c] = (img_float[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device), img_resized

    def _generate_heatmap_overlay(self, cam: np.ndarray, original: np.ndarray) -> str:
        heatmap_color = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original, 0.55, heatmap_color, 0.45, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def predict(self, contents: bytes) -> Dict[str, Any]:
        input_tensor, original_img = self._preprocess(contents)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            
            # 설도(idx=0) 확률을 우둔(idx=5)에 합산 (클래스 병합)
            probs[0, 5] += probs[0, 0]   # Beef_Round += Beef_BottomRound
            probs[0, 0] = 0               # Beef_BottomRound 제거
            
            confidence, pred_idx = probs.max(dim=1)

        class_name = MODEL_CLASS_NAMES[pred_idx.item()]
        class_name = CLASS_MERGE_MAP.get(class_name, class_name)  # 병합 매핑 적용
        conf_value = confidence.item()

        try:
            input_tensor.requires_grad_(True)
            cam = self.grad_cam.generate(input_tensor, pred_idx.item())
            heatmap_b64 = self._generate_heatmap_overlay(cam, original_img)
        except Exception as e:
            print(f"⚠ Grad-CAM 생성 실패 (추론 결과는 정상): {e}")
            heatmap_b64 = None

        return {
            "class_name": class_name,
            "confidence": round(conf_value, 4),
            "heatmap_image": heatmap_b64,
        }


_engine_instance: Optional[PredictEngine] = None


def get_predict_engine() -> PredictEngine:
    """싱글톤 PredictEngine 반환. 최초 호출 시 모델 로드."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PredictEngine()
    return _engine_instance
