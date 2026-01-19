"""
Meat-A-Eye 웹 프론트엔드(Next.js) 맞춤형 응답 규격 정의
"""
from typing import Any, Optional


def create_response(data: Any, message: str = "Success") -> dict:
    """
    성공 응답 생성 (Next.js 프론트엔드 규격)

    Args:
        data: 응답 데이터
        message: 응답 메시지

    Returns:
        표준화된 응답 딕셔너리
    """
    return {
        "status": "success",
        "message": message,
        "data": data
    }


def create_error_response(
    error_code: str,
    message: str,
    detail: Optional[str] = None
) -> dict:
    """
    에러 응답 생성

    Args:
        error_code: 에러 코드 (예: LOW_CONFIDENCE, INVALID_IMAGE)
        message: 사용자 표시용 메시지
        detail: 개발자용 상세 정보

    Returns:
        표준화된 에러 응답 딕셔너리
    """
    response = {
        "status": "error",
        "error_code": error_code,
        "message": message
    }
    if detail:
        response["detail"] = detail
    return response


# 에러 코드 상수
class ErrorCodes:
    """API 에러 코드 정의"""
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # 신뢰도 75% 미만
    INVALID_IMAGE = "INVALID_IMAGE"    # 이미지 파일 아님
    NOT_MEAT = "NOT_MEAT"              # 축산물 인식 불가
    OCR_FAILED = "OCR_FAILED"          # OCR 추출 실패
    SERVER_ERROR = "SERVER_ERROR"      # 서버 내부 오류
