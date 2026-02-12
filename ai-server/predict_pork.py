"""
EfficientNet-B2 기반 돼지 부위 분류 + Grad-CAM 히트맵 엔진.
소 predict_b2.py와 동일 구조, 돼지 7부위 전용 모델 사용.
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
MODEL_PATH = MODEL_DIR / "b2_imagenet_pork_50-v4.pth"
IMAGE_SIZE = 260

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 돼지 7부위 (알파벳순 — ImageFolder 기준)
CLASS_NAMES: List[str] = [
    "Pork_Belly",           # 0 — 삼겹살
    "Pork_Ham",             # 1 — 뒷다리
    "Pork_Loin",            # 2 — 등심
    "Pork_Neck",            # 3 — 목살
    "Pork_PicnicShoulder",  # 4 — 앞다리
    "Pork_Ribs",            # 5 — 갈비
    "Pork_Tenderloin",      # 6 — 안심
]
NUM_CLASSES = len(CLASS_NAMES)


def _build_model() -> nn.Module:
    """EfficientNet-B2 돼지 7부위 분류 모델 생성."""
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


class PorkPredictEngine:
    """돼지 부위 싱글톤 추론 엔진."""

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
                f"돼지 모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
                f"돼지 부위 모델을 먼저 학습해 주세요."
            )
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        print(f"✓ 돼지 모델 로드 완료: {MODEL_PATH.name} ({NUM_CLASSES} classes)")

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
            confidence, pred_idx = probs.max(dim=1)

        class_name = CLASS_NAMES[pred_idx.item()]
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


_pork_engine_instance: Optional[PorkPredictEngine] = None


def get_pork_predict_engine() -> PorkPredictEngine:
    """싱글톤 PorkPredictEngine 반환. 최초 호출 시 모델 로드."""
    global _pork_engine_instance
    if _pork_engine_instance is None:
        _pork_engine_instance = PorkPredictEngine()
    return _pork_engine_instance
