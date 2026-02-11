"""
Web Image Processor — 업로드된 이미지 바이트를 numpy RGB 배열로 변환.
main.py의 /ai/analyze 엔드포인트에서 사용.
"""
import cv2
import numpy as np


def process_web_image(contents: bytes) -> np.ndarray:
    """
    웹에서 업로드된 이미지 바이트를 RGB numpy 배열로 변환.

    Args:
        contents: 이미지 파일의 raw bytes (JPEG, PNG 등)

    Returns:
        RGB numpy 배열 (H, W, 3), dtype=uint8

    Raises:
        ValueError: 이미지 디코딩 실패 시
    """
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("이미지 디코딩에 실패했습니다. 유효한 이미지 파일인지 확인해주세요.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
