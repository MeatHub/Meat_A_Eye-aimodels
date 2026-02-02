"""
EfficientNet-B2 + Grad-CAM 기반 고기 부위 추론 엔진
vision_b2_imagenet.pth 가중치 사용
"""
import base64
import io
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import Dict, Optional

# train.py와 동일한 클래스 (Beef 10종)
CLASS_NAMES = [
    "Beef_BottomRound", "Beef_Brisket", "Beef_Chuck", "Beef_Rib",
    "Beef_Ribeye", "Beef_Round", "Beef_Shank", "Beef_Shoulder",
    "Beef_Sirloin", "Beef_Tenderloin",
]
IMAGE_SIZE = 260
MODEL_PATH = Path(__file__).parent.parent / "models" / "vision_b2_imagenet.pth"


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.output = None
        self.target_layer.register_forward_hook(self._save_output)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_output(self, module, input, output):
        self.output = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(input_tensor)
        loss = out[0, class_idx]
        loss.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.output, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap


class PredictB2Engine:
    """EfficientNet-B2 + Grad-CAM 추론 엔진"""

    def __init__(self, model_path: Optional[Path] = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_path or MODEL_PATH
        self.model = models.efficientnet_b2(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, len(CLASS_NAMES)
        )
        self._load_weights()
        self.model.to(self.device).eval()

        self.target_layer = self.model.features[-1]
        self.grad_cam = GradCAM(self.model, self.target_layer)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_weights(self):
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(f"모델 가중치 없음: {self.model_path}")

    def predict(self, image_bytes: bytes) -> Dict:
        """
        이미지 바이트로부터 부위 예측 + Grad-CAM 히트맵 반환

        Returns:
            {
                "class_name": str,
                "confidence": float (0~1),
                "heatmap_image": str (base64 JPEG),
            }
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        input_tensor = self.transform(Image.fromarray(img_array)).unsqueeze(0).to(self.device)

        with torch.set_grad_enabled(True):
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
            class_idx = pred.item()
            confidence = conf.item()
            class_name = CLASS_NAMES[class_idx]

            heatmap = self.grad_cam.generate(input_tensor, class_idx)
            heatmap_resized = np.uint8(255 * np.clip(heatmap, 0, 1))
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            heatmap_pil = Image.fromarray(heatmap_rgb)
            buf = io.BytesIO()
            heatmap_pil.save(buf, format="JPEG", quality=90)
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "class_name": class_name,
            "confidence": float(confidence),
            "heatmap_image": f"data:image/jpeg;base64,{heatmap_b64}",
        }


_engine: Optional[PredictB2Engine] = None


def get_predict_engine() -> PredictB2Engine:
    global _engine
    if _engine is None:
        _engine = PredictB2Engine()
    return _engine
