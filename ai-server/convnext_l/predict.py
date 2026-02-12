"""
ConvNeXt-Large/XL 소고기 부위 분류 + Grad-CAM 추론 엔진.
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

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_VARIANT = "large"  # "large" 또는 "xlarge"
MODEL_PATH = MODEL_DIR / f"convnext_{MODEL_VARIANT}_beef-v1.pth"
IMAGE_SIZE = 224

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
        self._g = self._a = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, '_a', o))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, '_g', go[0]))

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[0, class_idx].backward()
        w = self._g.mean(dim=(2, 3), keepdim=True)
        cam = (w * self._a).sum(1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0); cam /= (cam.max() + 1e-8)
        return cam


class BeefPredictor:
    def __init__(self, model_path=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(model_path) if model_path else MODEL_PATH

        if MODEL_VARIANT == "xlarge":
            model = models.convnext_xlarge(weights=None)
        else:
            model = models.convnext_large(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.to(self.device).eval()
        self.model = model
        self.grad_cam = _GradCAM(model, model.features[-1][-1])

    def _preprocess(self, contents):
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)
        img_resized = cv2.resize(img_np, (IMAGE_SIZE, IMAGE_SIZE))
        blob = img_resized.astype(np.float32) / 255.0
        blob = (blob - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(blob.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device), img_resized

    def predict(self, contents: bytes) -> Dict[str, Any]:
        inp, orig = self._preprocess(contents)
        with torch.no_grad():
            probs = F.softmax(self.model(inp), dim=1)
            conf, idx = probs.max(dim=1)

        class_name = CLASS_NAMES[idx.item()]
        try:
            inp.requires_grad_(True)
            cam = self.grad_cam.generate(inp, idx.item())
            heatmap_b64 = self._overlay(cam, orig)
        except Exception:
            heatmap_b64 = None

        top5 = probs[0].topk(min(5, NUM_CLASSES))
        return {
            "class_name": class_name,
            "confidence": round(conf.item(), 4),
            "top5": [{"class": CLASS_NAMES[i], "confidence": round(c, 4)}
                     for c, i in zip(top5.values.tolist(), top5.indices.tolist())],
            "heatmap": heatmap_b64,
            "model": f"ConvNeXt-{MODEL_VARIANT.upper()}",
        }

    @staticmethod
    def _overlay(cam, orig):
        h, w = orig.shape[:2]
        hm = cv2.resize(cam, (w, h))
        hmc = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        ov = cv2.addWeighted(orig, 0.6, hmc, 0.4, 0)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")
