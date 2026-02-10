"""
Meat-A-Eye 웹 부위 추론 엔진
- 우선: B2 17클래스 (meat_vision_b2_pro.pth) — 학습 시와 동일 전처리·클래스
- fallback: B0 10클래스 (meat_vision_v2.pth)
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional
from pathlib import Path

from .web_processor import preprocess_for_model, preprocess_for_model_b2

# ---------- B2 17클래스 (학습 dataset_final 폴더 순서와 동일) ----------
CLASS_NAMES_B2 = [
    "Beef_BottomRound", "Beef_Brisket", "Beef_Chuck", "Beef_Rib", "Beef_Ribeye",
    "Beef_Round", "Beef_Shank", "Beef_Shoulder", "Beef_Sirloin", "Beef_Tenderloin",
    "Pork_Belly", "Pork_Ham", "Pork_Loin", "Pork_Neck", "Pork_PicnicShoulder",
    "Pork_Ribs", "Pork_Tenderloin",
]
MODEL_PATH_B2 = Path(__file__).parent.parent / "models" / "meat_vision_b2_pro.pth"
IMAGE_SIZE_B2 = 260

# ---------- B0 10클래스 (fallback) ----------
MEAT_CLASSES_B0 = {
    0: {"name": "삼겹살", "name_en": "pork_belly", "animal": "pork"},
    1: {"name": "목살", "name_en": "pork_shoulder", "animal": "pork"},
    2: {"name": "등심", "name_en": "sirloin", "animal": "beef"},
    3: {"name": "안심", "name_en": "tenderloin", "animal": "beef"},
    4: {"name": "갈비", "name_en": "ribs", "animal": "beef"},
    5: {"name": "채끝", "name_en": "striploin", "animal": "beef"},
    6: {"name": "앞다리", "name_en": "front_leg", "animal": "pork"},
    7: {"name": "뒷다리", "name_en": "rear_leg", "animal": "pork"},
    8: {"name": "차돌박이", "name_en": "brisket", "animal": "beef"},
    9: {"name": "항정살", "name_en": "pork_jowl", "animal": "pork"},
}
MODEL_PATH_B0 = Path(__file__).parent.parent / "models" / "meat_vision_v2.pth"
CONFIDENCE_THRESHOLD = 0.75


def _create_model_b2(num_classes: int):
    model = models.efficientnet_b2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model


class MeatVisionModelB2:
    """EfficientNet-B2 17클래스. 학습 시와 동일 전처리(260x260)."""

    def __init__(self, model_path: Optional[Path] = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device(
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        path = model_path or MODEL_PATH_B2
        self.model = _create_model_b2(len(CLASS_NAMES_B2)).to(self.device)
        if path.exists():
            state = torch.load(str(path), map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"[Vision Engine] B2 17클래스 모델 로드: {path}")
        else:
            raise FileNotFoundError(f"B2 가중치 없음: {path}")

    @torch.no_grad()
    def predict(self, img_array: np.ndarray) -> Dict:
        processed = preprocess_for_model_b2(img_array)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)
        score, idx = torch.max(probs, dim=1)
        score, idx = score.item(), idx.item()
        class_name = CLASS_NAMES_B2[idx]
        return {
            "label": class_name,
            "label_en": class_name,
            "animal": "beef" if class_name.startswith("Beef_") else "pork",
            "class_idx": idx,
            "score": score,
            "is_valid": score >= CONFIDENCE_THRESHOLD,
        }


class MeatVisionModelB0:
    """EfficientNet-B0 10클래스 (fallback)."""

    def __init__(self, model_path: Optional[Path] = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device(
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            len(MEAT_CLASSES_B0),
        )
        path = model_path or MODEL_PATH_B0
        path_resolved = Path(path).resolve()
        if path.exists():
            state = torch.load(str(path), map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[Vision Engine] B0 10클래스 모델 로드 (fallback): {path_resolved}")
        else:
            pretrained = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            state = {k: v for k, v in pretrained.state_dict().items() if "classifier" not in k}
            self.model.load_state_dict(state, strict=False)
            print(f"[Vision Engine] B0 가중치 없음 (찾는 경로: {path_resolved}), classifier 랜덤 초기화")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, img_array: np.ndarray) -> Dict:
        processed = preprocess_for_model(img_array)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)
        score, idx = torch.max(probs, dim=1)
        score, idx = score.item(), idx.item()
        info = MEAT_CLASSES_B0.get(idx, {"name": "알 수 없음", "name_en": "unknown", "animal": "unknown"})
        return {
            "label": info["name"],
            "label_en": info["name_en"],
            "animal": info["animal"],
            "class_idx": idx,
            "score": score,
            "is_valid": score >= CONFIDENCE_THRESHOLD,
        }


_model_instance: Optional[object] = None


def get_model():
    """B2 가중치 있으면 B2 17클래스, 없으면 B0 10클래스."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    path_b2 = MODEL_PATH_B2.resolve()
    if MODEL_PATH_B2.exists():
        _model_instance = MeatVisionModelB2()
    else:
        print(f"[Vision Engine] B2 가중치 없음 (찾는 경로: {path_b2}) → B0 사용")
        _model_instance = MeatVisionModelB0()
    return _model_instance


def predict_part(img_array: np.ndarray) -> Dict:
    """고기 부위 판별 (API용)."""
    return get_model().predict(img_array)
