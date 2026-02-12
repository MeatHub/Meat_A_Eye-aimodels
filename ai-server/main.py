# main.py (Web-Optimized Version)
# Meat-A-Eye AI Web Server
# FastAPI 메인 서버 (Web API 엔드포인트)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 웹 필수: CORS 설정

from web_processor import process_web_image
from vision_engine import predict_part
from ocr_engine import extract_text
from predict_b2 import get_predict_engine
from predict_pork import get_pork_predict_engine

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
    file: UploadFile = File(...),
    mode: str = Form(default="beef"),  # beef(소) | pork(돼지) | ocr
):
    """
    고기 이미지 분석 API (소 버전 | 돼지 버전 | OCR 버전)

    Args:
        file: 업로드된 이미지 파일
        mode: "beef" (소 9부위) | "pork" (돼지 7부위) | "ocr" (이력번호 추출)

    Returns:
        beef/pork: class_name, confidence, heatmap_image
        ocr: data.trace_number
    """
    # 레거시 vision → beef 호환
    mode = (mode or "beef").strip().lower()
    if mode == "vision":
        mode = "beef"

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    contents = await file.read()
    processed_img = process_web_image(contents)

    # ----- 소 버전 (beef): EfficientNet-B2 소 9부위 -----
    if mode == "beef":
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

    # ----- 돼지 버전 (pork): EfficientNet-B2 돼지 7부위 -----
    if mode == "pork":
        try:
            pork_engine = get_pork_predict_engine()
            result = pork_engine.predict(contents)
            return {
                "status": "success",
                "class_name": result["class_name"],
                "confidence": result["confidence"],
                "heatmap_image": result["heatmap_image"],
            }
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=f"돼지 모델 로드 실패: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"돼지 추론 실패: {str(e)}")

    # ----- OCR 버전: 이력번호 추출 -----
    if mode == "ocr":
        ocr_result = extract_text(processed_img)
        if not ocr_result["success"]:
            return {
                "status": "error",
                "message": "텍스트 추출에 실패했습니다.",
                "error_code": "OCR_FAILED",
                "detail": ocr_result.get("error"),
            }
        return {
            "status": "success",
            "data": {
                "trace_number": ocr_result["text"],
                "raw_output": ocr_result["raw"],
            },
        }

    return {"status": "error", "message": "Invalid mode. Use 'beef', 'pork', or 'ocr'."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
