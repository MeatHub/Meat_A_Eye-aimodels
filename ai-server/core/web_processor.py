"""
Meat-A-Eye 브라우저 업로드 이미지 규격화 및 전처리 모듈
다양한 브라우저 포맷 대응 및 Alpha 채널 제거 등
"""
import io
import numpy as np
from PIL import Image
from typing import Union


# 모델 입력 크기 (EfficientNet-B0 기준)
MODEL_INPUT_SIZE = (224, 224)
# 서버 부하 방지를 위한 최대 이미지 크기
MAX_IMAGE_SIZE = 1920


def process_web_image(image_bytes: bytes) -> np.ndarray:
    """
    웹 업로드 이미지를 모델 추론용으로 전처리

    Args:
        image_bytes: 웹에서 업로드된 이미지 바이트

    Returns:
        전처리된 numpy 배열 (H, W, C) RGB 포맷
    """
    # 바이트를 PIL Image로 변환
    image = Image.open(io.BytesIO(image_bytes))

    # 1. Alpha 채널 제거 (PNG 등 투명 이미지 대응)
    if image.mode == 'RGBA':
        # 흰색 배경으로 합성
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # 2. 고해상도 이미지 리사이즈 (서버 부하 방지)
    if max(image.size) > MAX_IMAGE_SIZE:
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)

    # 3. numpy 배열로 변환
    img_array = np.array(image)

    return img_array


def preprocess_for_model(img_array: np.ndarray) -> np.ndarray:
    """
    모델 추론을 위한 최종 전처리 (Center Crop + Normalize)

    Args:
        img_array: RGB numpy 배열

    Returns:
        정규화된 numpy 배열 (224, 224, 3)
    """
    image = Image.fromarray(img_array)

    # 1. Center Crop을 위한 정사각형 크롭
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    image = image.crop((left, top, right, bottom))

    # 2. 모델 입력 크기로 리사이즈
    image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)

    # 3. numpy 배열로 변환 및 정규화 (0-1 범위)
    img_array = np.array(image).astype(np.float32) / 255.0

    # 4. ImageNet 정규화 (EfficientNet 표준)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    return img_array


def validate_image(image_bytes: bytes) -> bool:
    """
    이미지 유효성 검사

    Args:
        image_bytes: 업로드된 이미지 바이트

    Returns:
        유효한 이미지 여부
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        return True
    except Exception:
        return False
