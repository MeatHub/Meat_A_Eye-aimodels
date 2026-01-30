# Meat-A-Eye AI Core Module
from .vision_engine import predict_part
from .web_processor import process_web_image
from .ocr_engine import extract_text

__all__ = ['predict_part', 'process_web_image', 'extract_text']
