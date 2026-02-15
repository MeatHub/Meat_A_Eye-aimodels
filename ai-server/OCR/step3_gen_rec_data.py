"""
Step 3: PaddleOCR rec 학습 데이터 생성
======================================
relabeled/ 의 bbox 좌표로 이미지를 크롭하여
PaddleOCR recognition 학습에 필요한 형식으로 변환합니다.

PaddleOCR rec 학습 형식:
  rec_dataset/
    crop_images/
      trace_0001_0.jpg   ← bbox 로 잘라낸 글자 영역
      trace_0002_0.jpg
      ...
    rec_train.txt        ← "crop_images/trace_0001_0.jpg\tL02602045978929"
    rec_val.txt
    rec_test.txt
    dict.txt             ← 인식 가능한 문자 목록

사용법:
  python step3_gen_rec_data.py
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Set


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # Meat_A_Eye-aimodels
DATA_DIR = PROJECT_ROOT / "data" / "Paddle_train_images"
RELABELED_DIR = DATA_DIR / "relabeled"
REC_DIR = SCRIPT_DIR / "rec_dataset"               # OCR/ 폴더 안에 출력
SPLITS = ["train", "val", "test"]

# 크롭 이미지 타겟 높이 (폭은 비율 유지)
TARGET_HEIGHT = 48


def crop_and_save(img: np.ndarray, bbox: list, out_path: Path, pad_ratio: float = 0.05) -> bool:
    """bbox 영역을 크롭하여 저장. 약간의 패딩 추가."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # 패딩
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    # 높이를 TARGET_HEIGHT 로 리사이즈 (비율 유지)
    ch, cw = crop.shape[:2]
    scale = TARGET_HEIGHT / ch
    new_w = max(1, int(cw * scale))
    resized = cv2.resize(crop, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

    cv2.imencode(".jpg", resized)[1].tofile(str(out_path))
    return True


def build_dict(texts: Set[str]) -> str:
    """모든 학습 텍스트에서 고유 문자 추출 → dict.txt"""
    chars: Set[str] = set()
    for t in texts:
        chars.update(t)
    # 정렬: 숫자 → A → L
    sorted_chars = sorted(chars)
    return "\n".join(sorted_chars)


def main():
    print("=" * 70)
    print("Step 3: PaddleOCR rec 학습 데이터 생성")
    print(f"  데이터 경로: {DATA_DIR}")
    print("=" * 70)

    crop_dir = REC_DIR / "crop_images"
    crop_dir.mkdir(parents=True, exist_ok=True)

    all_texts: Set[str] = set()
    label_lines = {s: [] for s in SPLITS}
    stats = {s: {"total": 0, "success": 0, "fail": 0} for s in SPLITS}

    for split in SPLITS:
        ann_dir = RELABELED_DIR / split / "annotations"
        img_dir = RELABELED_DIR / split / "images"

        if not ann_dir.exists():
            print(f"  [{split}] 디렉터리 없음 — 건너뜀")
            continue

        for ann_path in sorted(ann_dir.glob("*.json")):
            stem = ann_path.stem

            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)

            img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue

            img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            for idx, obj in enumerate(ann["objects"]):
                stats[split]["total"] += 1
                bbox = obj["bbox"]
                text = obj["text"]
                all_texts.add(text)

                crop_name = f"{stem}_{idx}.jpg"
                crop_path = crop_dir / crop_name

                if crop_and_save(img, bbox, crop_path):
                    # PaddleOCR 형식: 상대경로\t라벨
                    label_lines[split].append(f"crop_images/{crop_name}\t{text}")
                    stats[split]["success"] += 1
                else:
                    stats[split]["fail"] += 1

    # 라벨 파일 저장
    for split in SPLITS:
        label_path = REC_DIR / f"rec_{split}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines[split]))
        print(f"  [{split}] {stats[split]['success']}/{stats[split]['total']} 크롭 성공 → {label_path.name}")

    # dict.txt 생성
    dict_content = build_dict(all_texts)
    dict_path = REC_DIR / "dict.txt"
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(dict_content)
    print(f"\n  dict.txt 생성: {len(dict_content.splitlines())} 문자")

    # 통계
    total = sum(s["success"] for s in stats.values())
    print(f"\n총 크롭 이미지: {total}장")
    print(f"결과 저장 위치: {REC_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
