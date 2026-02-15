"""
test_ocr_engine.py — ocr_engine.py 성능 테스트
================================================
test/images 안의 모든 이미지를 대상으로 성능 측정.

정답(Ground Truth) 소스:
  1. trace_XXXX.jpg → annotations/trace_XXXX.json 에서 text 필드
  2. 이름이 이력번호인 파일 (002188519524.jpg, L02601245978085.jpg 등)
     → 파일명이 정답. (2) 같은 중복 표시는 제거.

사용법:
  python test_ocr_engine.py                 # 기본: GPU
  python test_ocr_engine.py --cpu           # CPU 모드
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List

# ai-server 를 import 경로에 추가
SCRIPT_DIR = Path(__file__).resolve().parent           # OCR/
AI_SERVER_DIR = SCRIPT_DIR.parent                      # ai-server/
if str(AI_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVER_DIR))

from ocr_engine import MeatTraceabilityOCR

# 데이터 경로
DATA_DIR = AI_SERVER_DIR.parent / "data" / "Paddle_train_images"
TEST_IMG_DIR = DATA_DIR / "relabeled" / "test" / "images"
TEST_ANN_DIR = DATA_DIR / "relabeled" / "test" / "annotations"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_ground_truth(img_path: Path) -> Optional[str]:
    """
    이미지의 정답(Ground Truth) 반환.

    1. annotation JSON 이 있으면 → JSON 의 text
    2. 파일명이 이력번호 패턴이면 → 파일명에서 추출
    3. 어느 쪽도 아니면 → None
    """
    stem = img_path.stem  # e.g. "trace_0041", "L02601245978085(2)"

    # 1) annotation JSON 확인
    ann_path = TEST_ANN_DIR / f"{stem}.json"
    if ann_path.exists():
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            objects = ann.get("objects", [])
            if objects:
                return objects[0].get("text", "").strip()
        except Exception:
            pass

    # 2) 파일명 기반 (괄호/공백 제거)
    name = re.sub(r'\s*\([^)]*\)\s*', '', stem).replace(' ', '')
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


def main():
    parser = argparse.ArgumentParser(description="ocr_engine.py 성능 테스트")
    parser.add_argument("--cpu", action="store_true", help="CPU 모드 (기본: GPU)")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="테스트 이미지 폴더 (기본: relabeled/test/images)")
    args = parser.parse_args()

    use_gpu = not args.cpu
    img_dir = Path(args.test_dir) if args.test_dir else TEST_IMG_DIR

    if not img_dir.exists():
        print(f"[ERROR] 테스트 이미지 폴더가 없습니다: {img_dir}")
        return

    # 이미지 파일 수집
    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in VALID_EXTS
    )

    if not image_files:
        print(f"[ERROR] 이미지가 없습니다: {img_dir}")
        return

    print("=" * 90)
    print(f"  ocr_engine.py 성능 테스트 (EasyOCR det + PaddleOCR rec)")
    print(f"  이미지 폴더: {img_dir}")
    print(f"  이미지 수:   {len(image_files)}")
    print(f"  디바이스:    {'GPU' if use_gpu else 'CPU'}")
    print("=" * 90)

    # OCR 엔진 초기화
    t0 = time.time()
    ocr = MeatTraceabilityOCR(use_gpu=use_gpu)
    init_time = time.time() - t0
    print(f"  모델 로드 시간: {init_time:.1f}s")
    print("-" * 90)

    total = 0
    correct = 0
    errors: List[Dict] = []
    skipped = 0
    times: List[float] = []

    for img_path in image_files:
        fname = img_path.name
        gt = get_ground_truth(img_path)

        if gt is None:
            skipped += 1
            continue

        total += 1
        t1 = time.time()
        result = ocr.extract(str(img_path))
        elapsed = time.time() - t1
        times.append(elapsed)

        pred = result.get("text", "")
        method = result.get("method", "")
        conf = result.get("confidence", 0)

        is_correct = (pred == gt)
        if is_correct:
            correct += 1
            status = "O"
        else:
            status = "X"
            errors.append({
                "file": fname,
                "gt": gt,
                "pred": pred,
                "method": method,
                "conf": conf,
            })

        print(f"  {status} {fname:<40} GT={gt:<25} PRED={pred:<25} [{method}, {conf:.2f}, {elapsed:.2f}s]")

    print("=" * 90)

    # ── 리포트 ──
    if total > 0:
        accuracy = correct / total * 100
        avg_time = sum(times) / len(times) if times else 0

        print(f"\n  [결과 요약]")
        print(f"  총 테스트:   {total}개 (스킵: {skipped}개)")
        print(f"  정확:        {correct}개")
        print(f"  오류:        {len(errors)}개")
        print(f"  정확도:      {accuracy:.1f}%")
        print(f"  평균 시간:   {avg_time:.2f}s / 이미지")
        print(f"  총 소요:     {sum(times):.1f}s")

        if errors:
            print(f"\n  [오류 상세]")
            for e in errors:
                print(f"    {e['file']}")
                print(f"      GT:   {e['gt']}")
                print(f"      PRED: {e['pred']}  [{e['method']}, conf={e['conf']:.2f}]")
    else:
        print("\n  검증 가능한 이미지가 없습니다.")

    print()


if __name__ == "__main__":
    main()
