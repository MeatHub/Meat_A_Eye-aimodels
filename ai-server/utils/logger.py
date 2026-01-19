"""
Meat-A-Eye 서버 로그 관리 모듈
웹 트래픽 모니터링 및 추론 로그 기록
"""
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "meat-a-eye", log_level: str = "INFO") -> logging.Logger:
    """
    서버 로거 초기화

    Args:
        name: 로거 이름
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)

    Returns:
        설정된 Logger 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # 포맷 설정
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # 중복 핸들러 방지
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "meat-a-eye") -> logging.Logger:
    """기존 로거 반환"""
    return logging.getLogger(name)


# 기본 로거 인스턴스
logger = setup_logger()
