"""
Meat-A-Eye 웹 브라우저 업로드 이미지 기반 부위 추론 엔진
EfficientNet-B0 기반, 필요시 B2~B3 확장 가능
목표 추론 속도: 200ms 이내
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional
from pathlib import Path

from .web_processor import preprocess_for_model


# 소/돼지 주요 부위 7~10종 클래스 정의
MEAT_CLASSES = {
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

# 신뢰도 임계값 (75% 미만 시 경고)
CONFIDENCE_THRESHOLD = 0.75

# 모델 경로
MODEL_PATH = Path(__file__).parent.parent / "models" / "meat_vision_v2.pth"


class MeatVisionModel:
    """EfficientNet-B0 기반 고기 부위 분류 모델"""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        모델 초기화

        Args:
            model_path: 학습된 가중치 파일 경로 (.pth)
            device: 추론 디바이스 ("auto", "cuda", "cpu")
        """
        # 디바이스 설정 (Mac M2: MPS 우선, 없으면 CUDA → CPU)
        if device == "auto":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 모델 구조 생성 (EfficientNet-B0)
        self.model = models.efficientnet_b0(weights=None)

        # 분류 헤드 수정 (클래스 수에 맞게)
        num_classes = len(MEAT_CLASSES)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            num_classes
        )

        # 가중치 로드
        self.model_path = model_path or MODEL_PATH
        self._load_weights()

        # 평가 모드로 전환
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        """학습된 가중치 로드"""
        if Path(self.model_path).exists():
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[Vision Engine] 모델 가중치 로드 완료: {self.model_path}")
        else:
            print(f"[Vision Engine] 경고: 가중치 파일 없음. 사전학습 모델 사용")
            # ImageNet 사전학습 가중치로 초기화
            pretrained = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            # classifier를 제외한 가중치만 복사
            state_dict = pretrained.state_dict()
            # classifier 레이어 제외
            state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
            self.model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def predict(self, img_array: np.ndarray) -> Dict:
        """
        이미지에서 고기 부위 예측

        Args:
            img_array: 전처리된 이미지 numpy 배열

        Returns:
            예측 결과 딕셔너리
        """
        # 모델 입력용 전처리
        processed = preprocess_for_model(img_array)

        # numpy -> torch tensor 변환 (B, C, H, W)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        # 추론
        outputs = self.model(tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # 최고 확률 클래스
        score, predicted_idx = torch.max(probabilities, dim=1)
        score = score.item()
        predicted_idx = predicted_idx.item()

        # 결과 구성
        meat_info = MEAT_CLASSES.get(predicted_idx, {"name": "알 수 없음", "name_en": "unknown", "animal": "unknown"})

        return {
            "label": meat_info["name"],
            "label_en": meat_info["name_en"],
            "animal": meat_info["animal"],
            "class_idx": predicted_idx,
            "score": score,
            "is_valid": score >= CONFIDENCE_THRESHOLD
        }


# 싱글톤 모델 인스턴스 (서버 시작 시 1회 로드)
_model_instance: Optional[MeatVisionModel] = None


def get_model() -> MeatVisionModel:
    """모델 싱글톤 인스턴스 반환"""
    global _model_instance
    if _model_instance is None:
        _model_instance = MeatVisionModel()
    return _model_instance


def predict_part(img_array: np.ndarray) -> Dict:
    """
    고기 부위 판별 (API 엔드포인트용)

    Args:
        img_array: process_web_image로 전처리된 이미지

    Returns:
        {
            "label": "삼겹살",
            "label_en": "pork_belly",
            "animal": "pork",
            "score": 0.92,
            "is_valid": True
        }
    """
    model = get_model()
    return model.predict(img_array)
