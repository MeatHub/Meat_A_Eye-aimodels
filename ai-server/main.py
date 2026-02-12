# main.py
# Meat-A-Eye AI Server (백엔드 연동용)
# POST /predict  → 고기 부위 판별 (백엔드 vision 모드)
# POST /ai/analyze → 기존 호환 (vision + ocr)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.web_processor import process_web_image
from core.vision_engine import predict_part, predict_for_api
from core.ocr_engine import extract_text

app = FastAPI(title="Meat-A-Eye AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """서버 상태 확인"""
    return {"status": "running", "service": "Meat-A-Eye AI Server", "port": 8001}


# ---------------------------------------------------------------
# 백엔드 연동 전용 엔드포인트
# ---------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    고기 부위 판별 API (백엔드 vision 모드 전용)

    Request:
        Content-Type: multipart/form-data
        필드: file (이미지 파일)

    Response (JSON):
        성공: {status:"success", class_name, confidence, heatmap_image}
        실패: {status:"error", message}
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "status": "error",
            "message": "이미지 파일만 업로드 가능합니다.",
        }

    try:
        contents = await file.read()
        processed_img = process_web_image(contents)
        result = predict_for_api(processed_img)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"추론 실패: {str(e)}",
        }


# ---------------------------------------------------------------
# 기존 호환 엔드포인트 (vision + ocr)
# ---------------------------------------------------------------
@app.post("/ai/analyze")
async def analyze_meat(
    mode: str = Form(...),
    file: UploadFile = File(...)
):
    """
    고기 분석 API (vision/ocr 겸용)

    Args:
        mode: "vision" (부위 판별) 또는 "ocr" (이력번호 추출)
        file: 업로드된 이미지 파일
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    contents = await file.read()
    processed_img = process_web_image(contents)

    if mode == "vision":
        result = predict_for_api(processed_img)
        return result

    elif mode == "ocr":
        ocr_result = extract_text(processed_img)

        if not ocr_result["success"]:
            return {
                "status": "error",
                "error_code": "OCR_FAILED",
                "message": "텍스트 추출에 실패했습니다.",
                "detail": ocr_result.get("error"),
            }

        return {
            "status": "success",
            "data": {
                "trace_number": ocr_result["text"],
                "raw_output": ocr_result["raw"],
            },
        }

    return {"status": "error", "message": "Invalid mode. Use 'vision' or 'ocr'."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
