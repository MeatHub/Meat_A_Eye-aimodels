"""
EfficientNetV2-L 소고기 부위 분류 + Grad-CAM 추론 엔진.
predict_b2.py와 동일한 인터페이스 (predict_b2.py 대체용).
"""
import io
import base64
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image

# ── 설정 ──
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "v2l_beef_100-v1.pth"
IMAGE_SIZE = 480

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES: List[str] = [
    "Beef_Brisket", "Beef_Chuck", "Beef_Rib", "Beef_Ribeye", "Beef_Round",
    "Beef_Shank", "Beef_Shoulder", "Beef_Sirloin", "Beef_Tenderloin",
]
NUM_CLASSES = len(CLASS_NAMES)


class _GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._gradients = None
        self._activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self._activations = o
    def _bwd(self, m, gi, go): self._gradients = go[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[0, class_idx].backward()
        w = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * self._activations).sum(dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)
        return cam


class BeefPredictor:
    """EfficientNetV2-L 기반 소고기 부위 분류기."""

    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(model_path) if model_path else MODEL_PATH

        model = models.efficientnet_v2_l(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, NUM_CLASSES),
        )
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.to(self.device).eval()
        self.model = model

        self.grad_cam = _GradCAM(model, model.features[-1])

    def _preprocess(self, contents: bytes):
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)
        img_resized = cv2.resize(img_np, (IMAGE_SIZE, IMAGE_SIZE))
        blob = img_resized.astype(np.float32) / 255.0
        blob = (blob - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(blob.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        return tensor, img_resized

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
        except Exception:
            heatmap_b64 = None

        top5 = probs[0].topk(min(5, NUM_CLASSES))
        top5_results = [{"class": CLASS_NAMES[i], "confidence": round(c, 4)}
                        for c, i in zip(top5.values.tolist(), top5.indices.tolist())]

        return {
            "class_name": class_name,
            "confidence": round(conf_value, 4),
            "top5": top5_results,
            "heatmap": heatmap_b64,
            "model": "EfficientNetV2-L",
        }

    @staticmethod
    def _generate_heatmap_overlay(cam, original_img):
        h, w = original_img.shape[:2]
        heatmap = cv2.resize(cam, (w, h))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")
