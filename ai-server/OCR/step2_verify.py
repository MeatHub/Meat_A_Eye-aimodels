"""
Step 2: 릴레이블 결과 검증 — 시각화 & 통계
==========================================
- step1 에서 생성된 relabeled/ 의 bbox 를 이미지 위에 그려서 확인합니다.
- 원본(train/) vs 교정(relabeled/) bbox 를 나란히 비교합니다.
- 결과는 relabeled/preview/ 에 저장됩니다.

사용법:
  python step2_verify.py                  # 전체
  python step2_verify.py --limit 20       # 앞 20장만
  python step2_verify.py --split val      # val 만
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # Meat_A_Eye-aimodels
DATA_DIR = PROJECT_ROOT / "data" / "Paddle_train_images"
RELABELED_DIR = DATA_DIR / "relabeled"
PREVIEW_DIR = RELABELED_DIR / "preview"


def draw_bbox(img: np.ndarray, bbox: List[int], color: tuple, label: str, thickness: int = 3):
    """bbox [x1,y1,x2,y2] 를 이미지에 그리기"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # 라벨 텍스트
    font_scale = max(0.5, min(img.shape[1], img.shape[0]) / 2000)
    t_thick = max(1, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, t_thick)
    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), t_thick)


def process_one(split: str, stem: str):
    """원본 vs 교정 bbox 비교 이미지 생성"""
    # 교정된 이미지 & 라벨
    new_img_path = RELABELED_DIR / split / "images" / f"{stem}.jpg"
    new_ann_path = RELABELED_DIR / split / "annotations" / f"{stem}.json"
    if not new_img_path.exists() or not new_ann_path.exists():
        return None

    img = cv2.imdecode(np.fromfile(str(new_img_path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    with open(new_ann_path, "r", encoding="utf-8") as f:
        new_ann = json.load(f)

    # 원본 라벨 (있으면)
    old_ann_path = DATA_DIR / split / "annotations" / f"{stem}.json"
    old_ann = None
    if old_ann_path.exists():
        with open(old_ann_path, "r", encoding="utf-8") as f:
            old_ann = json.load(f)

    # 비교 이미지 생성
    canvas = img.copy()

    # 원본 bbox (파란색 점선)
    if old_ann:
        for obj in old_ann["objects"]:
            draw_bbox(canvas, obj["bbox"], (255, 100, 0), f"OLD: {obj['text'][:15]}", 2)

    # 교정 bbox (녹색 실선)
    for obj in new_ann["objects"]:
        draw_bbox(canvas, obj["bbox"], (0, 200, 0), f"NEW: {obj['text'][:15]}", 3)

    # 축소 (미리보기용)
    max_dim = 1200
    h, w = canvas.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)))

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None, help="특정 split 만 (train/val/test)")
    parser.add_argument("--limit", type=int, default=0, help="최대 처리 장수")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["train", "val", "test"]
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"데이터 경로: {DATA_DIR}")
    print(f"미리보기 저장: {PREVIEW_DIR}")

    count = 0
    changed = 0

    for split in splits:
        ann_dir = RELABELED_DIR / split / "annotations"
        if not ann_dir.exists():
            continue
        for ann_path in sorted(ann_dir.glob("*.json")):
            stem = ann_path.stem
            canvas = process_one(split, stem)
            if canvas is None:
                continue

            # 변경 여부 확인
            old_path = DATA_DIR / split / "annotations" / f"{stem}.json"
            if old_path.exists():
                with open(old_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                with open(ann_path, "r", encoding="utf-8") as f:
                    new = json.load(f)
                if any(
                    o["bbox"] != n["bbox"]
                    for o, n in zip(old["objects"], new["objects"])
                ):
                    changed += 1
                    # 변경된 것만 저장
                    out_path = PREVIEW_DIR / f"{split}_{stem}_compare.jpg"
                    cv2.imencode(".jpg", canvas)[1].tofile(str(out_path))

            count += 1
            if args.limit and count >= args.limit:
                break
        if args.limit and count >= args.limit:
            break

    print(f"검증 완료: 총 {count}건 확인, bbox 변경 {changed}건 (미리보기: {PREVIEW_DIR})")


if __name__ == "__main__":
    main()
