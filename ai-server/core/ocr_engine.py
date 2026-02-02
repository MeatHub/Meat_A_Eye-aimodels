"""
Meat-A-Eye 영수증/라벨지 텍스트 추출 로직
이력번호 추출을 위한 OCR 엔진
"""
import re
import numpy as np
from typing import Dict, Optional

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[OCR Engine] 경고: easyocr 미설치. OCR 기능 비활성화")


# 이력번호 패턴 (소: 12자리, 돼지: 12자리)
TRACE_NUMBER_PATTERNS = [
    r'\d{12}',           # 기본 12자리 숫자
    r'\d{4}-\d{4}-\d{4}', # 하이픈 포함
    r'\d{3}-\d{3}-\d{6}', # 다른 형식
]

# OCR 리더 인스턴스 (싱글톤)
_ocr_reader: Optional['easyocr.Reader'] = None


def get_ocr_reader():
    """EasyOCR 리더 싱글톤 반환"""
    global _ocr_reader
    if _ocr_reader is None and EASYOCR_AVAILABLE:
        # 한국어 + 영어 지원
        _ocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)
        print("[OCR Engine] EasyOCR 리더 초기화 완료")
    return _ocr_reader


def extract_trace_number(text: str) -> Optional[str]:
    """
    텍스트에서 이력번호 추출

    Args:
        text: OCR로 추출된 전체 텍스트

    Returns:
        이력번호 문자열 또는 None
    """
    # 공백 및 특수문자 정리
    cleaned = text.replace(' ', '').replace('-', '')

    for pattern in TRACE_NUMBER_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            return match.group()

    return None


def extract_text(img_array: np.ndarray) -> Dict:
    """
    이미지에서 텍스트 추출 (API 엔드포인트용)

    Args:
        img_array: 전처리된 이미지 numpy 배열

    Returns:
        {
            "text": "123456789012",  # 추출된 이력번호
            "raw": ["전체", "텍스트", "목록"],
            "success": True
        }
    """
    if not EASYOCR_AVAILABLE:
        return {
            "text": None,
            "raw": [],
            "success": False,
            "error": "EasyOCR not installed"
        }

    reader = get_ocr_reader()
    if reader is None:
        return {
            "text": None,
            "raw": [],
            "success": False,
            "error": "OCR reader initialization failed"
        }

    try:
        # OCR 수행
        results = reader.readtext(img_array)

        # 텍스트 추출
        raw_texts = [result[1] for result in results]
        full_text = ' '.join(raw_texts)

        # 이력번호 추출 시도
        trace_number = extract_trace_number(full_text)

        return {
            "text": trace_number,
            "raw": raw_texts,
            "success": True
        }

    except Exception as e:
        return {
            "text": None,
            "raw": [],
            "success": False,
            "error": str(e)
        }
