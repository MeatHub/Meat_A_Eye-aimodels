# -*- coding: utf-8 -*-
"""
EasyOCR + PaddleOCR v4 하이브리드 엔진 성능 평가.

- data/test2 의 이미지+JSON으로 정확도 측정
- 결과: output/easy_paddle_eval/report.json, results.json, 시각화(선택)

실행: python evaluate_easy_paddle.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "test2"
OUT_ROOT = PROJECT_ROOT / "output" / "easy_paddle_eval"
REPORT_JSON = OUT_ROOT / "report.json"
RESULTS_JSON = OUT_ROOT / "results.json"


def find_image_json_pairs(root: Path):
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


def load_gt(json_path: Path) -> str | None:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = (data.get("text") or data.get("label") or data.get("gt") or "").strip()
        return text if text else None
    except Exception:
        return None


def clean_text(text: str) -> str:
    import re
    if not text:
        return ""
    s = text.replace(" ", "").replace("\t", "").replace("\n", "")
    return re.sub(r"[^AL0-9]", "", s.upper())


def main():
    print(f"[INFO] 데이터: {DATA_ROOT}")
    pairs = find_image_json_pairs(DATA_ROOT)
    print(f"[INFO] 샘플 수: {len(pairs)}")
    if not pairs:
        print("[ERROR] data/test2 에 이미지+JSON 쌍이 없습니다.")
        return

    from easy_paddle import MeatEyeHybridEngine
    engine = MeatEyeHybridEngine()

    results = []
    correct = 0
    for img_path, json_path in pairs:
        gt = load_gt(json_path)
        if gt is None:
            continue
        pred = engine.process(str(img_path))
        if pred in ("미인식", "Not Found", ""):
            pred = ""
        gt_n = clean_text(gt)
        pred_n = clean_text(pred)
        ok = gt_n == pred_n
        if ok:
            correct += 1
        results.append({
            "image": str(img_path.name),
            "gt": gt,
            "pred": pred,
            "correct": ok,
        })

    total = len(results)
    accuracy = (correct / total * 100.0) if total else 0.0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "accuracy_percent": round(accuracy, 2),
            "engine": "EasyOCR + PaddleOCR v4 (MeatEyeHybridEngine)",
        }, f, ensure_ascii=False, indent=2)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 전체: {total}, 정답: {correct}, 정확도: {accuracy:.2f}%")
    print(f"[INFO] 리포트: {REPORT_JSON}")
    print(f"[INFO] 상세 결과: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
