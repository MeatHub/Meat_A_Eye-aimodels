# main.py (Web-Optimized Version)
# Meat-A-Eye AI Web Server
# FastAPI 메인 서버 (Web API 엔드포인트)
# 연동: 백엔드 Vision → POST /predict, OCR → POST /ai/analyze

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 웹 필수: CORS 설정

from core.web_processor import process_web_image
from core.vision_engine import predict_part
from core.ocr_engine import extract_text

app = FastAPI(title="Meat-A-Eye AI Web Server")

# 웹 프론트엔드(Next.js)와의 통신을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서는 전체 허용, 운영 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _label_en_to_class_name(label_en: str) -> str:
    """label_en(snake_case) → 백엔드 partName용 class_name (PascalCase). 예: pork_belly → Pork_Belly"""
    if not label_en:
        return "Unknown"
    return "".join(w.capitalize() for w in label_en.split("_"))


@app.get("/")
async def root():
    """서버 상태 확인"""
    return {"status": "running", "service": "Meat-A-Eye AI Server"}


# ---------- 백엔드 연동: Vision 모드 (백엔드가 POST /predict 호출) ----------
@app.post("/predict")
async def predict_vision(file: UploadFile = File(..., alias="file")):
    """
    고기 부위 인식 (Vision). 백엔드 연동용.
    요청: multipart/form-data, 키 `file`에 이미지 (jpeg/png/webp, 최대 5MB).
    응답: status, class_name(부위 코드), confidence.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기 초과 (최대 5MB).")
    processed_img = process_web_image(contents)
    result = predict_part(processed_img)
    class_name = _label_en_to_class_name(result.get("label_en", ""))
    confidence = float(result.get("score", 0.0))
    return {
        "status": "success",
        "class_name": class_name,
        "confidence": confidence,
    }


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
                "historyNo": ocr_result['text'],
                "raw_output": ocr_result['raw']
            }
        }

    return {"status": "error", "message": "Invalid mode. Use 'vision' or 'ocr'."}


if __name__ == "__main__":
    import uvicorn
    # 백엔드가 8000 사용 → AI 서버는 8001 (백엔드 .env AI_SERVER_URL=http://localhost:8001)
    uvicorn.run(app, host="0.0.0.0", port=8001)
