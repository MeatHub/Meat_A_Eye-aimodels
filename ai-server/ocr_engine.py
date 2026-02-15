"""
ocr_engine.py — 축산물 이력번호 OCR 엔진 (EasyOCR det + PaddleOCR rec 하이브리드)
================================================================================
main.py 에서 `from ocr_engine import extract_text` 로 사용.

핵심 파이프라인:
  1. EasyOCR  → 텍스트 영역(bbox) 탐지 + 수평 박스 병합
  2. PaddleOCR rec(학습 모델) → 크롭 이미지 문자 인식
  3. 이력번호 패턴 필터링 → 최종 결과 반환

학습 모델 경로:
  ai-server/OCR/rec_output/best_accuracy.pdparams  (en_PP-OCRv4 fine-tuned)
  ai-server/OCR/PaddleOCR/ppocr/utils/en_dict.txt   (95-char 딕셔너리)
"""

import cv2
import math
import numpy as np
import re
import os
import sys
import glob
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional

# ──────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent                    # ai-server/
_OCR_DIR = _SCRIPT_DIR / "OCR"                                  # ai-server/OCR/
_REC_MODEL_DIR = _OCR_DIR / "rec_output"                        # .pdparams 위치
_PADDLE_OCR_DIR = _OCR_DIR / "PaddleOCR"                        # 클론된 PaddleOCR 리포
_EN_DICT_PATH = _PADDLE_OCR_DIR / "ppocr" / "utils" / "en_dict.txt"


# ──────────────────────────────────────────────────────────────
# PaddleOCR rec 직접 로드 (paddleocr pip 패키지 사용 안 함)
# ──────────────────────────────────────────────────────────────

class PaddleRecInfer:
    """paddle 모델(.pdparams) 직접 로드를 이용한 경량 rec 추론기"""

    def __init__(self, model_dir: str, dict_path: str, use_gpu: bool = True):
        import paddle
        # PaddleOCR 리포를 sys.path 에 추가
        paddle_repo = str(_PADDLE_OCR_DIR)
        if paddle_repo not in sys.path:
            sys.path.insert(0, paddle_repo)
        from ppocr.modeling.architectures import build_model

        # 딕셔너리 로드
        self.character = ["blank"]
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                ch = line.strip("\n").strip("\r\n")
                if ch:
                    self.character.append(ch)
        self.character.append(" ")          # use_space_char
        char_num = len(self.character)

        # en_PP-OCRv4 아키텍처 (학습과 동일)
        arch_config = {
            "model_type": "rec",
            "algorithm": "SVTR_LCNet",
            "Transform": None,
            "Backbone": {"name": "PPLCNetV3", "scale": 0.95},
            "Head": {
                "name": "MultiHead",
                "head_list": [
                    {"CTCHead": {
                        "Neck": {"name": "svtr", "dims": 120, "depth": 2,
                                 "hidden_dims": 120, "kernel_size": [1, 3],
                                 "use_guide": True},
                        "Head": {"fc_decay": 0.00001}}},
                    {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
                ],
                "out_channels_list": {
                    "CTCLabelDecode": char_num,
                    "SARLabelDecode": char_num + 2,
                    "NRTRLabelDecode": char_num + 3,
                },
            },
        }

        if use_gpu and paddle.is_compiled_with_cuda():
            paddle.set_device("gpu:0")
        else:
            paddle.set_device("cpu")

        self.model = build_model(arch_config)

        # best_accuracy.pdparams → latest.pdparams 순서로 로드
        params_path = os.path.join(model_dir, "best_accuracy.pdparams")
        if not os.path.exists(params_path):
            params_path = os.path.join(model_dir, "latest.pdparams")
        state = paddle.load(params_path)
        self.model.set_state_dict(state)
        self.model.eval()

        self.rec_image_shape = [3, 48, 320]
        print(f"[PaddleRecInfer] 모델 로드 완료: {params_path}")

    # ── resize + normalize ──
    def _resize_norm(self, img: np.ndarray) -> np.ndarray:
        c, h, w = self.rec_image_shape
        ratio = w / float(h)
        ih, iw = img.shape[:2]
        wh_ratio = iw / float(ih)
        resized_w = w if wh_ratio > ratio else max(1, int(math.ceil(h * wh_ratio)))
        resized = cv2.resize(img, (resized_w, h))
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
        resized = resized.astype("float32").transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5
        padding = np.zeros((c, h, w), dtype=np.float32)
        padding[:, :, :resized_w] = resized
        return padding

    # ── 크롭 이미지 → (텍스트, 신뢰도) ──
    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        import paddle

        if img is None or img.size == 0:
            return "", 0.0
        norm = self._resize_norm(img)
        inp = paddle.to_tensor(norm[np.newaxis, :])

        with paddle.no_grad():
            preds = self.model(inp)

        if isinstance(preds, paddle.Tensor):
            output = preds.numpy()
        elif isinstance(preds, dict) and "ctc" in preds:
            output = preds["ctc"].numpy()
        else:
            return "", 0.0

        # CTC greedy decode
        preds_idx = output.argmax(axis=2)[0]
        preds_prob = output.max(axis=2)[0]
        text, conf_list, prev = "", [], 0
        for idx, prob in zip(preds_idx, preds_prob):
            if idx != 0 and idx != prev:
                if idx < len(self.character):
                    text += self.character[idx]
                    conf_list.append(prob)
            prev = idx
        confidence = float(np.mean(conf_list)) if conf_list else 0.0
        return text, confidence


# ──────────────────────────────────────────────────────────────
# 싱글톤 OCR 인스턴스
# ──────────────────────────────────────────────────────────────
_ocr_instance: Optional["MeatTraceabilityOCR"] = None


def extract_text(img_array: np.ndarray) -> Dict[str, Any]:
    """
    numpy 이미지 배열에서 이력번호 추출 (main.py /ai/analyze ocr 모드용).

    Args:
        img_array: RGB numpy 배열 (H, W, 3)

    Returns:
        {"success": bool, "text": str, "raw": str}
    """
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = MeatTraceabilityOCR()

    try:
        # RGB → BGR 변환 후 numpy 배열을 직접 전달 (JPEG 재압축 없음)
        bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        result = _ocr_instance.extract(img=bgr_img)
        text = result.get("text", "")
        return {
            "success": result.get("success", False),
            "text": text,
            "raw": text if text else "Not Found",
        }
    except Exception as e:
        return {"success": False, "text": "", "raw": str(e), "error": str(e)}


# ──────────────────────────────────────────────────────────────
# 메인 OCR 클래스
# ──────────────────────────────────────────────────────────────

class MeatTraceabilityOCR:
    """
    EasyOCR (det) + PaddleOCR rec (학습 모델) 하이브리드 OCR.

    - EasyOCR: 두 가지 전처리(bilateral, adaptive) 앙상블 → 텍스트 영역 탐지
    - PaddleOCR rec: bbox 크롭 → 문자 인식 (학습된 모델)
    - 이력번호 패턴 필터링 → 최종 선택
    - GPU 기본 사용
    """

    def __init__(self, use_gpu: bool = True):
        import easyocr

        self.use_gpu = use_gpu
        print(f"[MeatTraceabilityOCR] 초기화 (GPU={'ON' if use_gpu else 'OFF'})")

        # EasyOCR — 텍스트 탐지용
        self.reader = easyocr.Reader(['ko', 'en'], gpu=use_gpu)

        # PaddleOCR rec 모델 로드
        self.paddle_rec: Optional[PaddleRecInfer] = None
        model_dir = str(_REC_MODEL_DIR)
        dict_path = str(_EN_DICT_PATH)

        best = os.path.join(model_dir, "best_accuracy.pdparams")
        latest = os.path.join(model_dir, "latest.pdparams")
        if os.path.exists(best) or os.path.exists(latest):
            try:
                self.paddle_rec = PaddleRecInfer(model_dir, dict_path, use_gpu=use_gpu)
            except Exception as e:
                print(f"[MeatTraceabilityOCR] PaddleOCR 로드 실패 (EasyOCR 단독): {e}")
        else:
            print(f"[MeatTraceabilityOCR] PaddleOCR 모델 없음 → EasyOCR 단독 모드")

    # ── 전처리 (원본 해상도 유지 — 2x 확대 제거) ──
    def _preprocess_bilateral(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    def _preprocess_adaptive(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    # ── 텍스트 정제 ──
    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = text.replace(' ', '').replace('\t', '').replace('\n', '')
        return re.sub(r'[^AL0-9]', '', cleaned.upper())

    # ── 이력번호 패턴 매칭 ──
    @staticmethod
    def _is_trace_pattern(text: str) -> bool:
        if re.match(r'^\d{12}$', text):
            return True
        if re.match(r'^L\d{14}$', text):
            return True
        if re.match(r'^A\d{19,29}$', text):
            return True
        return False

    # ── 수평 박스 병합 ──
    def _merge_horizontal_boxes(self, results: list) -> list:
        if not results:
            return []
        sorted_r = sorted(results, key=lambda x: min(p[0] for p in x[0]))
        merged = []
        i = 0
        while i < len(sorted_r):
            bbox, text, prob = sorted_r[i]
            m_text = text
            m_prob = prob
            m_bbox = [list(p) for p in bbox]
            cnt = 1
            j = i + 1
            while j < len(sorted_r):
                nb, nt, np_ = sorted_r[j]
                ys1 = [p[1] for p in m_bbox]
                ys2 = [p[1] for p in nb]
                h_avg = ((max(ys1) - min(ys1)) + (max(ys2) - min(ys2))) / 2
                if h_avg == 0:
                    j += 1
                    continue
                cy_diff = abs(np.mean(ys1) - np.mean(ys2))
                x_gap = min(p[0] for p in nb) - max(p[0] for p in m_bbox)
                if cy_diff <= h_avg * 0.5 and x_gap < h_avg * 2:
                    m_text += nt.replace(" ", "")
                    m_prob = (m_prob * cnt + np_) / (cnt + 1)
                    cnt += 1
                    xs = [p[0] for p in m_bbox] + [p[0] for p in nb]
                    ys_all = [p[1] for p in m_bbox] + [p[1] for p in nb]
                    m_bbox = [
                        [min(xs), min(ys_all)],
                        [max(xs), min(ys_all)],
                        [max(xs), max(ys_all)],
                        [min(xs), max(ys_all)],
                    ]
                    j += 1
                else:
                    break
            merged.append((m_bbox, m_text, m_prob))
            i = j
        return merged

    # ── 이미지 로드 (한글 경로 지원) ──
    @staticmethod
    def _load_image(image_path: str) -> Optional[np.ndarray]:
        return cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)

    # ── 메인 추출 함수 ──
    def extract(self, image_path: str = None, img: np.ndarray = None) -> Dict[str, Any]:
        """
        이력번호 추출 메인 파이프라인.

        Args:
            image_path: 이미지 파일 경로 (CLI 테스트용)
            img: BGR numpy 배열 (API 호출용 — JPEG 재압축 없이 직접 전달)
            → 둘 중 하나만 제공하면 됨. img 가 있으면 image_path 무시.

        Returns:
            {
                "success": bool,
                "text": str,          # 최종 이력번호
                "method": str,        # "paddle_rec" | "easyocr" | "not_found"
                "confidence": float,
                "all_candidates": [...],
            }
        """
        if img is not None:
            pass  # numpy 배열 직접 사용
        elif image_path is not None:
            img = self._load_image(image_path)
        else:
            return {"success": False, "text": "", "method": "error", "confidence": 0}

        if img is None:
            return {"success": False, "text": "", "method": "error", "confidence": 0}

        h, w = img.shape[:2]

        # 1) 두 가지 전처리 앙상블 → EasyOCR 탐지
        all_results = []
        try:
            pre1 = self._preprocess_bilateral(img)
            all_results.extend(self.reader.readtext(pre1, detail=1))
        except Exception:
            pass
        try:
            pre2 = self._preprocess_adaptive(img)
            all_results.extend(self.reader.readtext(pre2, detail=1))
        except Exception:
            pass

        if not all_results:
            return {"success": False, "text": "", "method": "not_found", "confidence": 0}

        # 2) 수평 병합
        merged = self._merge_horizontal_boxes(all_results)

        # 3) 이력번호 후보 영역 필터
        det_boxes = []
        anchors = []
        keyword_patterns = ["이력", "번호", "묶음", "축산물"]

        for bbox, text, prob in merged:
            if any(k in text for k in keyword_patterns):
                anchors.append(bbox)

            ct = self._clean_text(text)
            if len(ct) >= 10 and re.match(r'^[AL0-9]{10,30}$', ct):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                box = [max(0, int(min(xs))), max(0, int(min(ys))),
                       min(w, int(max(xs))), min(h, int(max(ys)))]
                det_boxes.append({
                    "bbox": box,
                    "easyocr_text": ct,
                    "easyocr_conf": prob,
                })

        if not det_boxes:
            return {"success": False, "text": "", "method": "not_found", "confidence": 0}

        # 4) PaddleOCR rec 로 각 영역 재인식
        results = []
        for det in det_boxes:
            x1, y1, x2, y2 = det["bbox"]
            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * 0.05), int(bh * 0.1)
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
            crop = img[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            paddle_text, paddle_conf = "", 0.0
            if self.paddle_rec is not None:
                try:
                    pt, pc = self.paddle_rec.predict(crop)
                    paddle_text = self._clean_text(pt)
                    paddle_conf = pc
                except Exception:
                    pass

            easyocr_text = det["easyocr_text"]
            easyocr_conf = det["easyocr_conf"]

            # PaddleOCR 결과 우선, fallback → EasyOCR
            if self._is_trace_pattern(paddle_text) and paddle_conf > 0.5:
                results.append({
                    "text": paddle_text, "confidence": paddle_conf,
                    "method": "paddle_rec", "bbox": det["bbox"],
                })
            elif self._is_trace_pattern(easyocr_text):
                results.append({
                    "text": easyocr_text, "confidence": easyocr_conf,
                    "method": "easyocr", "bbox": det["bbox"],
                })
            elif paddle_text and len(paddle_text) >= 12:
                results.append({
                    "text": paddle_text, "confidence": paddle_conf,
                    "method": "paddle_rec_partial", "bbox": det["bbox"],
                })
            elif len(easyocr_text) >= 12:
                results.append({
                    "text": easyocr_text, "confidence": easyocr_conf,
                    "method": "easyocr_partial", "bbox": det["bbox"],
                })

        if not results:
            return {"success": False, "text": "", "method": "not_found", "confidence": 0}

        # 5) 앵커 거리 + 패턴 정확도 기반 최적 선택
        def sort_key(r):
            exact = self._is_trace_pattern(r["text"])
            is_paddle = "paddle" in r["method"]
            anchor_bonus = 0.0
            if anchors:
                center = [(r["bbox"][0] + r["bbox"][2]) / 2,
                          (r["bbox"][1] + r["bbox"][3]) / 2]
                min_dist = min(
                    np.linalg.norm(np.array(center) - np.mean(a, axis=0))
                    for a in anchors
                )
                anchor_bonus = 1.0 / (np.log(min_dist + 2))
            return (exact, is_paddle, r["confidence"] + anchor_bonus)

        results.sort(key=sort_key, reverse=True)
        best = results[0]

        return {
            "success": True,
            "text": best["text"],
            "method": best["method"],
            "confidence": best["confidence"],
            "all_candidates": results,
        }

    # ── 파일명 기반 정답 추출 (테스트용) ──
    @staticmethod
    def _extract_ground_truth(file_path: str) -> Optional[str]:
        name = os.path.splitext(os.path.basename(file_path))[0]
        name = re.sub(r'\s*\([^)]*\)\s*', '', name).replace(' ', '')
        cleaned = re.sub(r'[^AL0-9]', '', name.upper())
        if re.match(r'^\d{12}$', cleaned):
            return cleaned
        if re.match(r'^L\d{14}$', cleaned):
            return cleaned
        if re.match(r'^A\d{19,29}$', cleaned):
            return cleaned
        if 12 <= len(cleaned) <= 30 and re.match(r'^[AL0-9]+$', cleaned):
            return cleaned
        return None


# ──────────────────────────────────────────────────────────────
# CLI 테스트
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_DIR = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\data\OCR_test"

    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(TEST_DIR, ext)))

    if not image_files:
        print(f"경로를 다시 확인해주세요: {TEST_DIR}")
    else:
        print(f"[{len(image_files)}개 파일 테스트 — EasyOCR det + PaddleOCR rec 하이브리드]")
        print("=" * 100)

        extractor = MeatTraceabilityOCR()

        correct_count = 0
        total_count = 0
        error_log = []

        for img_path in image_files:
            file_name = os.path.basename(img_path)
            try:
                result = extractor.extract(image_path=img_path)
                pred = result.get("text", "")
                method = result.get("method", "")
                conf = result.get("confidence", 0)
                ground_truth = extractor._extract_ground_truth(img_path)

                if ground_truth:
                    total_count += 1
                    is_correct = (pred == ground_truth)
                    if is_correct:
                        correct_count += 1
                        status = "✓"
                    else:
                        status = "✗"
                        error_log.append({
                            'file': file_name,
                            'ground_truth': ground_truth,
                            'predicted': pred,
                        })
                    print(f"  {status} {file_name:<35} 실제: {ground_truth:<25} 예측: {pred:<25} [{method}, {conf:.2f}]")
                else:
                    print(f"  ? {file_name:<35} 예측: {pred:<25} [{method}] (검증불가)")

            except KeyboardInterrupt:
                print("\n중단됨.")
                break
            except Exception as e:
                print(f"  ! {file_name:<35} 에러: {e}")

        print("=" * 100)
        if total_count > 0:
            accuracy = correct_count / total_count * 100
            print(f"\n정확도: {correct_count}/{total_count} = {accuracy:.2f}%")
            if error_log:
                print(f"\n[오류 목록]")
                for err in error_log:
                    print(f"  {err['file']}: 실제={err['ground_truth']} / 예측={err['predicted']}")
        else:
            print("\n검증 가능한 파일이 없습니다.")
