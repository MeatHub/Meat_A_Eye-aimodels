"""
Vision Engine — 레거시 호환용.
main.py에서 import되지만 실제 사용되지 않음 (predict_b2.get_predict_engine으로 대체됨).
"""
from predict_b2 import get_predict_engine
from typing import Dict, Any


def predict_part(contents: bytes) -> Dict[str, Any]:
    """레거시 호환용. 내부적으로 predict_b2 엔진을 사용."""
    engine = get_predict_engine()
    return engine.predict(contents)
