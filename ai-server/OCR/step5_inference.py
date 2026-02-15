"""
Step 5: 추론 파이프라인 — EasyOCR det + PaddleOCR rec
====================================================
최종 목표 파이프라인:
  1. EasyOCR 로 이미지에서 텍스트 영역(bbox) 탐지
  2. bbox 크롭 → 학습된 PaddleOCR rec 모델로 문자 인식
  3. 이력번호 패턴 필터링 후 최종 결과 반환

사용법:
  # 단일 이미지
  python step5_inference.py --image path/to/image.jpg

  # 폴더 전체 테스트
  python step5_inference.py --test-dir C:\\path\\to\\test_images

  # 학습 데이터로 정확도 확인
  python step5_inference.py --eval
"""

import os
import sys
import re
import math
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # Meat_A_Eye-aimodels
DATA_DIR = PROJECT_ROOT / "data" / "Paddle_train_images"
REC_OUTPUT_DIR = SCRIPT_DIR / "rec_output"          # OCR/ 안
REC_DIR = SCRIPT_DIR / "rec_dataset"                 # OCR/ 안
EN_DICT_PATH = SCRIPT_DIR / "PaddleOCR" / "ppocr" / "utils" / "en_dict.txt"


class PaddleRecInfer:
    """paddle 모델 직접 로드를 이용한 경량 rec 추론기"""

    def __init__(self, model_dir: str, dict_path: str, use_gpu: bool = False):
        import paddle
        sys.path.insert(0, str(SCRIPT_DIR / "PaddleOCR"))
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process

        # 딕셔너리 로드
        self.character = ["blank"]
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                ch = line.strip("\n").strip("\r\n")
                if ch:
                    self.character.append(ch)
        self.character.append(" ")  # use_space_char
        char_num = len(self.character)

        # 모델 빌드
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

        # best_accuracy.pdparams 로드
        params_path = os.path.join(model_dir, "best_accuracy.pdparams")
        if not os.path.exists(params_path):
            params_path = os.path.join(model_dir, "latest.pdparams")
        state = paddle.load(params_path)
        self.model.set_state_dict(state)
        self.model.eval()

        self.rec_image_shape = [3, 48, 320]

    def _resize_norm(self, img: np.ndarray) -> np.ndarray:
        """RecResizeImg: 비율 유지하며 [3, 48, 320]으로 resize"""
        c, h, w = self.rec_image_shape
        ratio = w / float(h)
        ih, iw = img.shape[:2]
        wh_ratio = iw / float(ih)
        if wh_ratio > ratio:
            resized_w = w
        else:
            resized_w = max(1, int(math.ceil(h * wh_ratio)))
        resized = cv2.resize(img, (resized_w, h))
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
        resized = resized.astype("float32")
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized -= 0.5
        resized /= 0.5
        padding = np.zeros((c, h, w), dtype=np.float32)
        padding[:, :, :resized_w] = resized
        return padding

    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        """크롭 이미지 → (텍스트, 신뢰도)"""
        import paddle

        if img is None or img.size == 0:
            return "", 0.0
        norm = self._resize_norm(img)
        inp = paddle.to_tensor(norm[np.newaxis, :])

        with paddle.no_grad():
            preds = self.model(inp)

        # eval 모드에서 MultiHead 는 CTC 출력만 반환
        if isinstance(preds, paddle.Tensor):
            output = preds.numpy()
        elif isinstance(preds, dict) and "ctc" in preds:
            output = preds["ctc"].numpy()
        else:
            return "", 0.0

        # CTC decode
        preds_idx = output.argmax(axis=2)[0]
        preds_prob = output.max(axis=2)[0]
        text = ""
        conf_list = []
        prev = 0
        for idx, prob in zip(preds_idx, preds_prob):
            if idx != 0 and idx != prev:
                if idx < len(self.character):
                    text += self.character[idx]
                    conf_list.append(prob)
            prev = idx

        confidence = float(np.mean(conf_list)) if conf_list else 0.0
        return text, confidence


class HybridOCR:
    """EasyOCR (det) + PaddleOCR (rec) 하이브리드 파이프라인"""

    def __init__(self, paddle_model_dir: Optional[str] = None, use_gpu: bool = True):
        import easyocr

        print(f"[HybridOCR] EasyOCR 로딩 중... (GPU={'ON' if use_gpu else 'OFF'})")
        self.reader = easyocr.Reader(["ko", "en"], gpu=use_gpu)

        # PaddleOCR rec 모델 로드
        self.paddle_rec = None
        model_dir = paddle_model_dir or str(REC_OUTPUT_DIR)

        best_params = os.path.join(model_dir, "best_accuracy.pdparams")
        latest_params = os.path.join(model_dir, "latest.pdparams")
        if os.path.exists(best_params) or os.path.exists(latest_params):
            try:
                dict_path = str(EN_DICT_PATH)
                print(f"[HybridOCR] PaddleOCR rec 모델 로딩: {model_dir}")
                self.paddle_rec = PaddleRecInfer(model_dir, dict_path, use_gpu=use_gpu)
                print("[HybridOCR] PaddleOCR rec 모델 로드 완료")
            except Exception as e:
                print(f"[HybridOCR] PaddleOCR 로드 실패 (EasyOCR 단독 모드): {e}")
        else:
            print(f"[HybridOCR] PaddleOCR 모델 없음 ({model_dir}) — EasyOCR 단독 모드")

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """bilateral + adaptive 두 가지 전처리 (원본 해상도 유지)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return bilateral, adaptive

    def _clean(self, text: str) -> str:
        cleaned = text.replace(" ", "").replace("\t", "").replace("\n", "")
        return re.sub(r"[^AL0-9]", "", cleaned.upper())

    def _is_trace_pattern(self, text: str) -> bool:
        if re.match(r"^\d{12}$", text):
            return True
        if re.match(r"^L\d{14}$", text):
            return True
        if re.match(r"^A\d{19,29}$", text):
            return True
        return False

    def _merge_horizontal(self, results: list) -> list:
        """수평 인접 박스 병합"""
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
                # y 중심 차이 및 x 간격 체크
                ys1 = [p[1] for p in m_bbox]
                ys2 = [p[1] for p in nb]
                h_avg = ((max(ys1) - min(ys1)) + (max(ys2) - min(ys2))) / 2
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

    def detect(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """EasyOCR 로 텍스트 영역 탐지 (앙상블 + 병합)"""
        h, w = img.shape[:2]
        bilateral, adaptive = self._preprocess(img)

        results1 = self.reader.readtext(bilateral, detail=1)
        results2 = self.reader.readtext(adaptive, detail=1)
        all_results = results1 + results2
        merged = self._merge_horizontal(all_results)

        candidates = []
        for bbox, text, prob in merged:
            ct = self._clean(text)
            if len(ct) >= 10 and re.match(r"^[AL0-9]{10,30}$", ct):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                box = [
                    max(0, int(min(xs))),
                    max(0, int(min(ys))),
                    min(w, int(max(xs))),
                    min(h, int(max(ys))),
                ]
                candidates.append({
                    "bbox": box,
                    "easyocr_text": ct,
                    "easyocr_conf": prob,
                })
        return candidates

    def recognize_crop(self, crop_img: np.ndarray) -> Tuple[str, float]:
        """PaddleOCR rec 로 크롭 이미지 인식"""
        if self.paddle_rec is None:
            return "", 0.0

        try:
            text, conf = self.paddle_rec.predict(crop_img)
            return self._clean(text), conf
        except Exception as e:
            pass
        return "", 0.0

    def extract(self, image_path: str) -> Dict[str, Any]:
        """
        메인 추론 함수

        Returns:
            {
                "success": bool,
                "text": str,         # 최종 이력번호
                "method": str,       # "paddle_rec" | "easyocr_det" | "not_found"
                "confidence": float,
                "all_candidates": [...],
            }
        """
        img = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "text": "", "method": "error", "confidence": 0}

        h, w = img.shape[:2]
        det_boxes = self.detect(img)

        if not det_boxes:
            return {"success": False, "text": "", "method": "not_found", "confidence": 0}

        results = []

        for det in det_boxes:
            x1, y1, x2, y2 = det["bbox"]
            # 약간 패딩
            bw = x2 - x1
            bh = y2 - y1
            px, py = int(bw * 0.05), int(bh * 0.1)
            cx1 = max(0, x1 - px)
            cy1 = max(0, y1 - py)
            cx2 = min(w, x2 + px)
            cy2 = min(h, y2 + py)
            crop = img[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            # PaddleOCR rec
            paddle_text, paddle_conf = self.recognize_crop(crop)

            # EasyOCR det 결과도 후보에 포함
            easyocr_text = det["easyocr_text"]
            easyocr_conf = det["easyocr_conf"]

            # 최종 선택: PaddleOCR 가 유효한 패턴이면 paddle, 아니면 easyocr
            if self._is_trace_pattern(paddle_text) and paddle_conf > 0.5:
                results.append({
                    "text": paddle_text,
                    "confidence": paddle_conf,
                    "method": "paddle_rec",
                    "bbox": det["bbox"],
                })
            elif self._is_trace_pattern(easyocr_text):
                results.append({
                    "text": easyocr_text,
                    "confidence": easyocr_conf,
                    "method": "easyocr_det",
                    "bbox": det["bbox"],
                })
            elif paddle_text and len(paddle_text) >= 12:
                results.append({
                    "text": paddle_text,
                    "confidence": paddle_conf,
                    "method": "paddle_rec_partial",
                    "bbox": det["bbox"],
                })
            elif len(easyocr_text) >= 12:
                results.append({
                    "text": easyocr_text,
                    "confidence": easyocr_conf,
                    "method": "easyocr_det_partial",
                    "bbox": det["bbox"],
                })

        if not results:
            return {"success": False, "text": "", "method": "not_found", "confidence": 0}

        # 우선순위 정렬: 정확 패턴 > 신뢰도
        def sort_key(r):
            is_exact = self._is_trace_pattern(r["text"])
            is_paddle = "paddle" in r["method"]
            return (is_exact, is_paddle, r["confidence"])

        results.sort(key=sort_key, reverse=True)
        best = results[0]

        return {
            "success": True,
            "text": best["text"],
            "method": best["method"],
            "confidence": best["confidence"],
            "all_candidates": results,
        }


def run_eval(ocr: HybridOCR):
    """annotation bbox 크롭 → PaddleOCR rec 직접 테스트 (기본 모드)"""
    test_dir = DATA_DIR / "relabeled" / "test"
    if not test_dir.exists():
        test_dir = DATA_DIR / "test"
    ann_dir = test_dir / "annotations"
    img_dir = test_dir / "images"

    if not (ann_dir.exists() and img_dir.exists()):
        print(f"[ERROR] 평가 데이터가 없습니다: {ann_dir}")
        return

    import json
    total = correct = 0
    print(f"\n[평가] annotation bbox 크롭 → PaddleOCR rec 인식")
    print(f"  이미지: {img_dir}")
    print(f"  어노테이션: {ann_dir}")
    print("-" * 80)

    for ann_path in sorted(ann_dir.glob("*.json")):
        stem = ann_path.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        gt = ann["objects"][0]["text"]
        bbox = ann["objects"][0]["bbox"]  # [x1, y1, x2, y2]

        img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        # 패딩 추가
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * 0.03), int(bh * 0.05)
        crop = img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]

        if crop.size == 0:
            continue

        pred_text, pred_conf = ocr.recognize_crop(crop)
        total += 1
        is_correct = pred_text == gt
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        print(f"  {status} {stem}: GT={gt:<25} PRED={pred_text:<25} (conf={pred_conf:.2f})")

    if total > 0:
        print(f"\nbbox 크롭 인식 정확도: {correct}/{total} = {correct / total * 100:.1f}%")


def run_full_eval(ocr: HybridOCR):
    """실제 원본 이미지로 EasyOCR det + PaddleOCR rec 전체 파이프라인 평가"""
    test_dir = DATA_DIR / "relabeled" / "test"
    if not test_dir.exists():
        test_dir = DATA_DIR / "test"
    ann_dir = test_dir / "annotations"
    img_dir = test_dir / "images"

    if not (ann_dir.exists() and img_dir.exists()):
        print(f"[ERROR] 평가 데이터가 없습니다: {ann_dir}")
        return

    import json
    total = correct = 0
    print(f"\n[전체 파이프라인 테스트] EasyOCR det + PaddleOCR rec")
    print(f"  이미지: {img_dir}")
    print(f"  어노테이션: {ann_dir}")
    print("-" * 80)

    for ann_path in sorted(ann_dir.glob("*.json")):
        stem = ann_path.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        gt = ann["objects"][0]["text"]

        result = ocr.extract(str(img_path))
        pred = result["text"]
        method = result["method"]
        conf = result["confidence"]
        total += 1
        is_correct = pred == gt
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        print(f"  {status} {stem}: GT={gt:<25} PRED={pred:<25} [{method}, {conf:.2f}]")

    if total > 0:
        print(f"\n전체 파이프라인 정확도: {correct}/{total} = {correct / total * 100:.1f}%")


def run_crop_eval(ocr: HybridOCR):
    """rec_test.txt 기반 크롭 이미지 평가 (fallback)"""
    test_label = REC_DIR / "rec_test.txt"
    if not test_label.exists():
        print(f"[ERROR] rec_test.txt 없음: {test_label}")
        return

    print(f"\n[크롭 이미지 테스트] PaddleOCR rec 단독 (rec_test.txt)")
    print("-" * 80)
    with open(test_label, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    total = correct = 0
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        crop_path = REC_DIR / parts[0]
        gt = parts[1]

        if not crop_path.exists():
            continue

        crop = cv2.imdecode(np.fromfile(str(crop_path), np.uint8), cv2.IMREAD_COLOR)
        if crop is None:
            continue

        pred_text, pred_conf = ocr.recognize_crop(crop)
        total += 1
        is_correct = pred_text == gt
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        print(f"  {status} GT={gt:<25} PRED={pred_text:<25} (conf={pred_conf:.2f})")

    if total > 0:
        print(f"\n크롭 인식 정확도: {correct}/{total} = {correct / total * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="EasyOCR det + PaddleOCR rec 추론")
    parser.add_argument("--image", type=str, help="단일 이미지 경로")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="테스트 이미지 폴더")
    parser.add_argument("--eval", action="store_true", help="annotation bbox 크롭 → PaddleOCR rec 정확도 평가")
    parser.add_argument("--full-eval", action="store_true", help="EasyOCR det + PaddleOCR rec 전체 파이프라인 평가")
    parser.add_argument("--crop-eval", action="store_true", help="rec_test.txt 기반 크롭 이미지 평가")
    parser.add_argument("--model-dir", type=str, default=None, help="PaddleOCR rec 모델 디렉터리")
    parser.add_argument("--cpu", action="store_true", help="CPU 사용 (기본: GPU)")
    args = parser.parse_args()

    use_gpu = not args.cpu

    print("=" * 70)
    print("EasyOCR det + PaddleOCR rec 하이브리드 추론")
    print(f"  데이터 경로: {DATA_DIR}")
    print(f"  디바이스: {'GPU' if use_gpu else 'CPU'}")
    print("=" * 70)

    ocr = HybridOCR(paddle_model_dir=args.model_dir, use_gpu=use_gpu)

    # 인자 없이 실행 시 기본으로 --eval 모드
    no_args = not args.image and not args.test_dir and not args.full_eval and not args.crop_eval
    if args.full_eval:
        run_full_eval(ocr)
    elif args.crop_eval:
        run_crop_eval(ocr)
    elif args.eval or no_args:
        run_eval(ocr)
    elif args.image:
        result = ocr.extract(args.image)
        print(f"\n이미지: {args.image}")
        print(f"결과:   {result['text']}")
        print(f"방법:   {result['method']}")
        print(f"신뢰도: {result['confidence']:.3f}")
        if result.get("all_candidates"):
            print(f"\n전체 후보:")
            for c in result["all_candidates"]:
                print(f"  [{c['method']}] {c['text']} (conf={c['confidence']:.3f})")
    elif args.test_dir:
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            print(f"[ERROR] 테스트 폴더가 없습니다: {test_dir}")
            return
        from glob import glob
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            files.extend(glob(str(test_dir / ext)))

        print(f"\n{len(files)}개 이미지 테스트 ({test_dir})")
        print("-" * 70)

        for img_path in sorted(files):
            fname = os.path.basename(img_path)
            result = ocr.extract(img_path)
            status = "✓" if result["success"] else "✗"
            print(f"  {status} {fname:<35} → {result['text']:<25} ({result['method']})")
    else:
        print("\n사용법:")
        print("  python step5_inference.py --image <이미지>")
        print("  python step5_inference.py --test-dir <폴더>")
        print("  python step5_inference.py --eval")


if __name__ == "__main__":
    main()
