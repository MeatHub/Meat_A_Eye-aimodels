# main.py (Web-Optimized Version)
# Meat-A-Eye AI Web Server
# 연동: 백엔드 Vision → POST /predict (file), OCR → POST /ai/analyze
# 부위명: 백엔드 PART_TO_CODES 키와 맞춰 404 방지

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.web_processor import process_web_image
from core.vision_engine import predict_part
from core.ocr_engine import extract_text

app = FastAPI(title="Meat-A-Eye AI Web Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# B2 17클래스 → 백엔드 PART_TO_CODES 키 (404 방지)
B2_TO_BACKEND = {
    "Pork_Ribs": "Pork_Rib",
    "Pork_PicnicShoulder": "Pork_Shoulder",
    "Pork_Ham": "Pork_Shoulder",
    "Pork_Neck": "Pork_Loin",
    "Pork_Tenderloin": "Pork_Loin",
}
# B0 10클래스 PascalCase → 백엔드 키
B0_TO_BACKEND = {
    "FrontLeg": "Pork_Shoulder",
    "RearLeg": "Pork_Shoulder",
    "PorkBelly": "Pork_Belly",
    "PorkShoulder": "Pork_Loin",
    "Sirloin": "Beef_Sirloin",
    "Tenderloin": "Beef_Tenderloin",
    "Ribs": "Beef_Rib",
    "Striploin": "Beef_Ribeye",
    "Brisket": "Beef_Brisket",
    "PorkJowl": "Pork_Loin",
    "Unknown": "Pork_Belly",
}


def _to_backend_class_name(result: dict) -> str:
    """AI 추론 결과 → 백엔드 PART_TO_CODES에 있는 부위명."""
    label_en = result.get("label_en", "")
    if not label_en:
        return "Pork_Belly"
    # B2: 이미 Beef_* / Pork_* 형태
    if label_en.startswith("Beef_") or label_en.startswith("Pork_"):
        return B2_TO_BACKEND.get(label_en, label_en)
    # B0: snake_case → PascalCase → 매핑
    pascal = "".join(w.capitalize() for w in label_en.split("_"))
    return B0_TO_BACKEND.get(pascal, pascal)


@app.get("/")
async def root():
    return {"status": "running", "service": "Meat-A-Eye AI Server"}


@app.post("/predict")
async def predict_vision(file: UploadFile = File(..., alias="file")):
    """
    Vision 연동. multipart/form-data, 필드 `file` (이미지).
    응답: status, class_name(PART_TO_CODES 호환), confidence, heatmap_image(null 가능).
    실패 시 status != "success", message → 백엔드 422.
    """
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            return {"status": "error", "message": "이미지 파일만 업로드 가능합니다."}
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            return {"status": "error", "message": "파일 크기 초과 (최대 5MB)."}
        processed_img = process_web_image(contents)
        result = predict_part(processed_img)
        class_name = _to_backend_class_name(result)
        confidence = float(result.get("score", 0.0))
        return {
            "status": "success",
            "class_name": class_name,
            "confidence": confidence,
            "heatmap_image": None,
        }
    except FileNotFoundError as e:
        return {"status": "error", "message": "모델 가중치를 찾을 수 없습니다. (B2: meat_vision_b2_pro.pth, B0: meat_vision_v2.pth)"}
    except Exception as e:
        return {"status": "error", "message": f"인식 중 오류가 발생했습니다. 다시 촬영해 주세요. ({str(e)})"}


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
