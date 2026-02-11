# -*- coding: utf-8 -*-
"""
EasyOCR + PaddleOCR v4 기반 학습/데이터 파이프라인.

- data/test2 (이미지 + 동일명.json)를 train/val/test(70/15/15)로 분할
- 분할된 경로를 output/test2_split/ 에 저장 (train/val/test 폴더)
- CRNN 등 추가 모델 학습 시 이 경로를 데이터로 사용 가능

실행: python train_easy_paddle.py
"""

import json
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "test2"
SPLIT_ROOT = PROJECT_ROOT / "output" / "test2_split"
SEED = 42


def find_image_json_pairs(root: Path):
    """이미지와 같은 이름의 .json 쌍 목록 반환."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []
    if not root.exists():
        return pairs
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in exts:
            continue
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        pairs.append((img_path, json_path))
    return pairs


def load_gt_from_json(json_path: Path) -> str | None:
    """JSON에서 GT 텍스트만 반환."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = (data.get("text") or data.get("label") or data.get("gt") or "").strip()
        return text if text else None
    except Exception:
        return None


def main():
    print(f"[INFO] 데이터 루트: {DATA_ROOT}")
    pairs = find_image_json_pairs(DATA_ROOT)
    print(f"[INFO] 이미지+JSON 쌍: {len(pairs)}개")
    if not pairs:
        print("[ERROR] data/test2 에 이미지와 동일명 .json 이 없습니다.")
        return

    random.seed(SEED)
    random.shuffle(pairs)
    n = len(pairs)
    t = int(n * 0.70)
    v = int(n * 0.15)
    train_pairs = pairs[:t]
    val_pairs = pairs[t : t + v]
    test_pairs = pairs[t + v :]

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        out_dir = SPLIT_ROOT / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path, json_path in split_pairs:
            shutil.copy2(img_path, out_dir / img_path.name)
            shutil.copy2(json_path, out_dir / json_path.name)
        print(f"[INFO] {split_name}: {len(split_pairs)}개 -> {out_dir}")

    # split 정보 저장 (학습 스크립트에서 참조)
    split_info = {
        "data_root": str(DATA_ROOT),
        "split_root": str(SPLIT_ROOT),
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
        "seed": SEED,
    }
    with open(SPLIT_ROOT / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    print(f"[INFO] split_info.json 저장: {SPLIT_ROOT / 'split_info.json'}")
    print("[INFO] 학습/평가 시 output/test2_split/train, val, test 경로를 사용하세요.")


if __name__ == "__main__":
    main()
