# main.py (Web-Optimized Version)
# Meat-A-Eye AI Web Server
# FastAPI 메인 서버 (Web API 엔드포인트)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 웹 필수: CORS 설정

from core.web_processor import process_web_image
from core.vision_engine import predict_part
from core.ocr_engine import extract_text
from core.predict_b2 import get_predict_engine

app = FastAPI(title="Meat-A-Eye AI Web Server")

# 웹 프론트엔드(Next.js)와의 통신을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서는 전체 허용, 운영 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """서버 상태 확인"""
    return {"status": "running", "service": "Meat-A-Eye AI Server"}


@app.post("/predict")
async def predict_meat_part(file: UploadFile = File(...)):
    """
    EfficientNet-B2 + Grad-CAM 기반 고기 부위 추론
    
    Returns:
        class_name, confidence, heatmap_image(base64)
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    contents = await file.read()
    try:
        engine = get_predict_engine()
        result = engine.predict(contents)
        return {
            "status": "success",
            "class_name": result["class_name"],
            "confidence": result["confidence"],
            "heatmap_image": result["heatmap_image"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"모델 로드 실패: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"추론 실패: {str(e)}")


@app.post("/ai/analyze")
async def analyze_meat(
    mode: str = Form(...),  # vision 또는 ocr
    file: UploadFile = File(...)
):
    """
    고기 이미지 분석 API

    Args:
        mode: "vision" (부위 판별) 또는 "ocr" (이력번호 추출)
        file: 업로드된 이미지 파일

    Returns:
        분석 결과 JSON
    """
    # 1. 파일 검증 (웹 브라우저 업로드 특성 반영)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    contents = await file.read()

    processed_img = process_web_image(contents)

    if mode == "vision":
        # 고기 부위 판별
        result = predict_part(processed_img)

        # 신뢰도 75% 미만 시 상세 가이드 반환
        if not result['is_valid']:
            return {
                "status": "warning",
                "error_code": "LOW_CONFIDENCE",
                "message": "이미지가 흐립니다. 다시 업로드해 주세요.",
                "data": {
                    "category": result['label'],
                    "probability": round(result['score'] * 100, 2),
                    "is_valid": False
                }
            }

        return {
            "status": "success",
            "data": {
                "category": result['label'],
                "category_en": result['label_en'],
                "animal": result['animal'],
                "probability": round(result['score'] * 100, 2),
                "is_valid": result['is_valid']
            }
        }

    elif mode == "ocr":
        # 이력번호 및 텍스트 추출
        ocr_result = extract_text(processed_img)

        if not ocr_result['success']:
            return {
                "status": "error",
                "error_code": "OCR_FAILED",
                "message": "텍스트 추출에 실패했습니다.",
                "detail": ocr_result.get('error')
            }

        return {
            "status": "success",
            "data": {
                "trace_number": ocr_result['text'],
                "raw_output": ocr_result['raw']
            }
        }

    return {"status": "error", "message": "Invalid mode. Use 'vision' or 'ocr'."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
