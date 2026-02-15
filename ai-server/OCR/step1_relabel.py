"""
Step 1: EasyOCR 기반 라벨 좌표 자동 재생성
=========================================
- 기존 JSON 의 text(이력번호 정답)는 유지하면서,
  EasyOCR det 결과로 bbox 좌표를 교정합니다.
- 삭제된(이미지 없는) annotation 은 자동 스킵합니다.
- 결과는 relabeled/ 에 저장되며 원본은 건드리지 않습니다.

사용법:
  python step1_relabel.py
"""

import os
import sys
import json
import re
import shutil
import cv2
import numpy as np
import easyocr
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any


# ── 경로 설정 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # Meat_A_Eye-aimodels
DATA_DIR = PROJECT_ROOT / "data" / "Paddle_train_images"
SPLITS = ["train", "val", "test"]

# 결과 출력 디렉터리
OUTPUT_DIR = DATA_DIR / "relabeled"


# ── 유틸리티 ───────────────────────────────────────────────
def clean_text(text: str) -> str:
    """OCR 텍스트에서 A/L/숫자만 추출"""
    cleaned = text.replace(" ", "").replace("\t", "").replace("\n", "")
    cleaned = re.sub(r"[^AL0-9]", "", cleaned.upper())
    return cleaned


def is_trace_number(text: str) -> bool:
    """축산물 이력번호 패턴인지 확인"""
    c = clean_text(text)
    if re.match(r"^\d{12}$", c):
        return True
    if re.match(r"^L\d{14}$", c):
        return True
    if re.match(r"^A\d{19,29}$", c):
        return True
    return False


def poly_to_xywh(poly: List[List[float]]) -> Tuple[int, int, int, int]:
    """EasyOCR polygon [[x,y],...] → [x_min, y_min, x_max, y_max]"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


# ── 좌표 변환 (해상도/회전 변경 대응) ────────────────────────
def scale_bbox(bbox: List[int], old_w: int, old_h: int, new_w: int, new_h: int) -> List[int]:
    """이미지 리사이즈에 맞춰 bbox 좌표를 비례 변환"""
    sx = new_w / old_w
    sy = new_h / old_h
    return [int(bbox[0] * sx), int(bbox[1] * sy), int(bbox[2] * sx), int(bbox[3] * sy)]


def rotate_bbox_cw(bbox: List[int], old_w: int, old_h: int) -> List[int]:
    """90° 시계방향 회전: (x,y)→(old_h-1-y, x). 새 이미지는 old_H × old_W"""
    x1, y1, x2, y2 = bbox
    # 네 꼭짓점 변환 후 min/max
    new_x1 = old_h - 1 - y2
    new_y1 = x1
    new_x2 = old_h - 1 - y1
    new_y2 = x2
    return [min(new_x1, new_x2), min(new_y1, new_y2),
            max(new_x1, new_x2), max(new_y1, new_y2)]


def rotate_bbox_ccw(bbox: List[int], old_w: int, old_h: int) -> List[int]:
    """90° 반시계방향 회전: (x,y)→(y, old_w-1-x). 새 이미지는 old_H × old_W"""
    x1, y1, x2, y2 = bbox
    new_x1 = y1
    new_y1 = old_w - 1 - x2
    new_x2 = y2
    new_y2 = old_w - 1 - x1
    return [min(new_x1, new_x2), min(new_y1, new_y2),
            max(new_x1, new_x2), max(new_y1, new_y2)]


def transform_old_bbox(old_bbox: List[int], ann_w: int, ann_h: int,
                       actual_w: int, actual_h: int) -> List[Tuple[List[int], str]]:
    """
    어노테이션 좌표를 현재 이미지 좌표로 변환.
    
    Returns:
        [(변환된_bbox, 변환명), ...]
        - 회전이 필요한 경우 CW/CCW 두 옵션 반환 (호출측에서 IoU로 선택)
    """
    if ann_w == actual_w and ann_h == actual_h:
        return [(old_bbox, "none")]

    # 방향이 같으면 (portrait→portrait 또는 landscape→landscape) → 단순 스케일
    same_orientation = (ann_w >= ann_h) == (actual_w >= actual_h)
    if same_orientation:
        return [(scale_bbox(old_bbox, ann_w, ann_h, actual_w, actual_h), "scale")]

    # 방향이 다르면 (portrait↔landscape) → 회전 + 스케일
    cw = rotate_bbox_cw(old_bbox, ann_w, ann_h)
    ccw = rotate_bbox_ccw(old_bbox, ann_w, ann_h)

    # 회전 후 이미지 크기: [ann_h, ann_w]
    rot_w, rot_h = ann_h, ann_w
    if rot_w != actual_w or rot_h != actual_h:
        cw = scale_bbox(cw, rot_w, rot_h, actual_w, actual_h)
        ccw = scale_bbox(ccw, rot_w, rot_h, actual_w, actual_h)

    return [(cw, "cw"), (ccw, "ccw")]


def iou(box_a: List[int], box_b: List[int]) -> float:
    """두 [x1,y1,x2,y2] 박스의 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def center_distance(box_a: List[int], box_b: List[int]) -> float:
    """두 박스 중심간 거리"""
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def boxes_horizontally_aligned(b1: List[List[float]], b2: List[List[float]]) -> bool:
    """두 EasyOCR 폴리곤이 수평으로 인접한지 판단"""
    ys1 = [p[1] for p in b1]
    ys2 = [p[1] for p in b2]
    h1 = max(ys1) - min(ys1)
    h2 = max(ys2) - min(ys2)
    avg_h = (h1 + h2) / 2

    cy1 = np.mean(ys1)
    cy2 = np.mean(ys2)
    if abs(cy1 - cy2) > avg_h * 0.5:
        return False

    xs1 = [p[0] for p in b1]
    xs2 = [p[0] for p in b2]
    gap = min(xs2) - max(xs1)
    return gap < avg_h * 2


def is_primarily_numeric(text: str) -> bool:
    """텍스트가 주로 숫자/AL 문자인지 판단 (한글 라벨 제외용)"""
    cleaned = text.replace(" ", "")
    if not cleaned:
        return False
    al_num = sum(1 for c in cleaned if c.upper() in 'AL0123456789')
    return al_num / len(cleaned) >= 0.6


def merge_horizontal(results: list) -> list:
    """수평 인접 박스 병합 (숫자끼리만, 한글 라벨은 제외)"""
    if not results:
        return []
    sorted_r = sorted(results, key=lambda x: min(p[0] for p in x[0]))
    merged: list = []
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
            # 한글 라벨과 숫자를 합치지 않음
            if not is_primarily_numeric(m_text) or not is_primarily_numeric(nt):
                break
            if boxes_horizontally_aligned(m_bbox, nb):
                m_text += nt.replace(" ", "")
                m_prob = (m_prob * cnt + np_) / (cnt + 1)
                cnt += 1
                xs = [p[0] for p in m_bbox] + [p[0] for p in nb]
                ys = [p[1] for p in m_bbox] + [p[1] for p in nb]
                m_bbox = [
                    [min(xs), min(ys)],
                    [max(xs), min(ys)],
                    [max(xs), max(ys)],
                    [min(xs), max(ys)],
                ]
                j += 1
            else:
                break
        merged.append((m_bbox, m_text, m_prob))
        i = j
    return merged


# ── "이력번호" 라벨 근처인지 판단하는 키워드 ────────────────
TRACE_LABEL_KEYWORDS = ["이력번호", "이력(묶음)번호", "이력 번호", "묶음번호", "생산이력코드", "생산이력", "이력코드"]
EXCLUDE_KEYWORDS = ["품목", "보고번호", "소비기한", "전화", "TEL", "FAX", "판매코드",
                    "바코드", "제조원", "도축장", "포장재질", "보관방법", "품번",
                    "포장일자", "중량"]


# ── 핵심 로직 ──────────────────────────────────────────────
def detect_all_text(reader: easyocr.Reader, image_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    EasyOCR 로 이미지에서 모든 텍스트 영역을 탐지합니다.

    Returns:
        (number_candidates, all_text_boxes)
        - number_candidates: 숫자 패턴 후보 [{"bbox", "ocr_text", "conf"}, ...]
        - all_text_boxes: 전체 텍스트 [{"bbox", "raw_text", "conf"}, ...]
    """
    img = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return [], []

    h, w = img.shape[:2]

    # bilateral 전처리
    enlarged = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    results1 = reader.readtext(filtered, detail=1)

    # adaptive 전처리
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    results2 = reader.readtext(adaptive, detail=1)

    all_results = results1 + results2

    # ── 전체 텍스트 박스 (라벨 앵커 탐지용) ──
    all_text_boxes = []
    seen_texts = set()
    for bbox, text, prob in all_results:
        raw_box = poly_to_xywh(bbox)
        box = [raw_box[0] // 2, raw_box[1] // 2, raw_box[2] // 2, raw_box[3] // 2]
        box = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
        key = (text.strip(), box[0] // 20, box[1] // 20)  # 중복 제거
        if key not in seen_texts:
            seen_texts.add(key)
            all_text_boxes.append({"bbox": box, "raw_text": text.strip(), "conf": prob})

    # ── 숫자 후보 (병합 후 필터링) ──
    merged = merge_horizontal(all_results)
    number_candidates = []
    for bbox, text, prob in merged:
        ct = clean_text(text)
        if len(ct) >= 12 and re.match(r"^[AL0-9]{12,30}$", ct):
            raw_box = poly_to_xywh(bbox)
            box = [raw_box[0] // 2, raw_box[1] // 2, raw_box[2] // 2, raw_box[3] // 2]
            box = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
            number_candidates.append({"bbox": box, "ocr_text": ct, "conf": prob})

    return number_candidates, all_text_boxes


def find_trace_label_anchor(all_text_boxes: List[Dict]) -> Optional[List[int]]:
    """
    "이력번호" 또는 "이력(묶음)번호" 라벨 텍스트의 bbox를 찾습니다.
    이 위치를 앵커로 사용하여 바로 오른쪽/같은 행의 숫자를 이력번호로 판정합니다.
    """
    for tb in all_text_boxes:
        raw = tb["raw_text"].replace(" ", "")
        for kw in TRACE_LABEL_KEYWORDS:
            if kw.replace(" ", "") in raw.replace(" ", ""):
                return tb["bbox"]
    return None


def is_near_exclude_label(box: List[int], all_text_boxes: List[Dict]) -> bool:
    """해당 박스가 '품목보고번호', '전화', '소비기한' 등 비이력 라벨 바로 옆인지 판단"""
    bh = box[3] - box[1]
    cy = (box[1] + box[3]) / 2

    for tb in all_text_boxes:
        raw = tb["raw_text"].replace(" ", "")
        is_exclude = any(kw in raw for kw in EXCLUDE_KEYWORDS)
        if not is_exclude:
            continue
        # 같은 행인지 (y 중심 차이가 박스 높이 이내)
        tb_cy = (tb["bbox"][1] + tb["bbox"][3]) / 2
        if abs(cy - tb_cy) <= bh * 0.8:
            # 그리고 이 박스가 제외 라벨의 오른쪽에 있으면 → 제외
            if box[0] >= tb["bbox"][0] - bh:
                return True
    return False


def is_same_row_right_of(anchor: List[int], box: List[int]) -> bool:
    """box 가 anchor 의 같은 행 오른쪽에 있는지 판단"""
    anchor_h = anchor[3] - anchor[1]
    anchor_cy = (anchor[1] + anchor[3]) / 2
    box_cy = (box[1] + box[3]) / 2

    # y 중심이 앵커 높이의 1.5배 이내 (같은 행 또는 바로 아래 행)
    if abs(anchor_cy - box_cy) > anchor_h * 1.5:
        return False
    # x는 앵커 왼쪽 끝보다 오른쪽이어야
    if box[2] < anchor[0]:
        return False
    return True


def match_best_box(
    gt_text: str,
    old_bbox: List[int],
    det_boxes: List[Dict],
    all_text_boxes: List[Dict],
    return_score: bool = False,
) -> Any:
    """
    정답 텍스트와 기존 bbox 를 기준으로 가장 적합한 det 박스를 선택합니다.

    개선된 전략:
      1. "이력번호" 라벨 앵커를 찾아서 같은 행의 숫자만 1차 필터
      2. 비이력 라벨(품목보고번호, 전화 등) 옆의 숫자는 제외
      3. 나머지 중 OLD bbox 와 IoU 가 가장 높은 것 선택 (위치 우선)
      4. IoU 0 이면 OLD bbox 에 가장 가까운 것 (단, 거리 제한)

    Args:
        return_score: True 면 (bbox, score) 튜플 반환. False 면 bbox 만 반환.
    """
    if not det_boxes:
        return ([], -1.0) if return_score else None

    gt_clean = clean_text(gt_text)

    # ── Phase 1: 앵커 기반 필터링 ──
    anchor = find_trace_label_anchor(all_text_boxes)

    # 비이력 라벨 근처 제외
    filtered = []
    for d in det_boxes:
        if is_near_exclude_label(d["bbox"], all_text_boxes):
            continue
        filtered.append(d)

    # 앵커가 있으면 같은 행 오른쪽만
    if anchor and filtered:
        anchored = [d for d in filtered if is_same_row_right_of(anchor, d["bbox"])]
        if anchored:
            filtered = anchored

    # 필터 후 후보가 없으면 원본으로 폴백
    if not filtered:
        filtered = det_boxes

    # ── Phase 2: OLD bbox 와 IoU 기반 매칭 (위치 최우선) ──
    scored = []
    for d in filtered:
        box_iou = iou(old_bbox, d["bbox"])
        dist = center_distance(old_bbox, d["bbox"])

        # 텍스트 유사도 (보조용)
        match_chars = sum(1 for a, b in zip(gt_clean, d["ocr_text"]) if a == b)
        text_ratio = match_chars / max(len(gt_clean), 1)

        # 이력번호 패턴 정확 매칭 보너스
        pattern_bonus = 1.0 if is_trace_number(d["ocr_text"]) else 0.0

        # 종합 점수: IoU 가중치 50% + 거리 역수 30% + 텍스트 유사도 20%
        img_diag = ((old_bbox[2] - old_bbox[0]) ** 2 + (old_bbox[3] - old_bbox[1]) ** 2) ** 0.5
        dist_norm = max(0, 1.0 - dist / max(img_diag * 3, 1))

        score = (box_iou * 5.0) + (dist_norm * 3.0) + (text_ratio * 1.5) + (pattern_bonus * 0.5)
        scored.append((d, score, box_iou, dist, text_ratio))

    if not scored:
        return (None, -1.0) if return_score else None

    scored.sort(key=lambda x: -x[1])
    best = scored[0]

    # 거리가 너무 멀면 (OLD bbox 대각선의 5배 이상) 교정 포기
    old_diag = ((old_bbox[2] - old_bbox[0]) ** 2 + (old_bbox[3] - old_bbox[1]) ** 2) ** 0.5
    if best[3] > old_diag * 5 and best[2] == 0:
        return (None, -1.0) if return_score else None

    if return_score:
        return best[0]["bbox"], best[1]  # (bbox, score)
    return best[0]["bbox"]


def process_split(reader: easyocr.Reader, split: str) -> Dict[str, int]:
    """한 split(train/val/test) 전체를 처리"""
    ann_dir = DATA_DIR / split / "annotations"
    img_dir = DATA_DIR / split / "images"
    out_ann_dir = OUTPUT_DIR / split / "annotations"
    out_img_dir = OUTPUT_DIR / split / "images"
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "updated": 0, "kept": 0, "skipped_no_image": 0, "skipped_no_det": 0}

    ann_files = sorted(ann_dir.glob("*.json"))
    for ann_path in ann_files:
        stats["total"] += 1
        stem = ann_path.stem  # trace_XXXX

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            print(f"  [SKIP] {stem} — 이미지 없음 (삭제됨)")
            stats["skipped_no_image"] += 1
            continue

        # ── 실제 이미지 크기 & 좌표 변환 준비 ──
        img_for_size = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
        if img_for_size is None:
            print(f"  [SKIP] {stem} — 이미지 읽기 실패")
            stats["skipped_no_image"] += 1
            continue
        actual_h, actual_w = img_for_size.shape[:2]
        ann_w, ann_h = ann.get("image_size", [actual_w, actual_h])
        needs_transform = (ann_w != actual_w or ann_h != actual_h)
        if needs_transform:
            transform_tag = "SWAP" if ((ann_w >= ann_h) != (actual_w >= actual_h)) else "SCALE"
        del img_for_size  # 메모리 절약

        # EasyOCR 탐지 (숫자 후보 + 전체 텍스트)
        det_boxes, all_text_boxes = detect_all_text(reader, str(img_path))

        # 각 object 의 bbox 교정
        updated = False
        for obj in ann["objects"]:
            old_bbox = obj["bbox"]
            gt_text = obj["text"]

            if not det_boxes:
                stats["skipped_no_det"] += 1
                break

            # ── 해상도/회전 변경 시 old_bbox 좌표 변환 ──
            if needs_transform:
                bbox_options = transform_old_bbox(old_bbox, ann_w, ann_h, actual_w, actual_h)
                if len(bbox_options) == 1:
                    # 단순 스케일: 변환된 좌표로 매칭
                    transformed_bbox = bbox_options[0][0]
                    new_bbox = match_best_box(gt_text, transformed_bbox, det_boxes, all_text_boxes)
                else:
                    # 회전: CW/CCW 두 방향 시도, 더 높은 점수 채택
                    best_new_bbox = None
                    best_score = -1.0
                    for trans_bbox, trans_name in bbox_options:
                        candidate, score = match_best_box(
                            gt_text, trans_bbox, det_boxes, all_text_boxes, return_score=True
                        )
                        if candidate is not None and score > best_score:
                            best_score = score
                            best_new_bbox = candidate
                    new_bbox = best_new_bbox
            else:
                new_bbox = match_best_box(gt_text, old_bbox, det_boxes, all_text_boxes)

            if new_bbox and new_bbox != old_bbox:
                obj["bbox"] = new_bbox
                updated = True

        if updated:
            stats["updated"] += 1
        else:
            stats["kept"] += 1

        # 실제 이미지 크기로 image_size 교정 (이미 위에서 읽은 값 재활용)
        ann["image_size"] = [actual_w, actual_h]

        # 수정된 라벨 json 파일(UTF-8) 저장
        with open(out_ann_dir / ann_path.name, "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=4)

        # 이미지 심볼릭 링크 또는 복사
        dst_img = out_img_dir / f"{stem}.jpg"
        if not dst_img.exists():
            shutil.copy2(str(img_path), str(dst_img))

    return stats


# ── 메인 ───────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Step 1: EasyOCR 기반 라벨 좌표 자동 재생성")
    print(f"  데이터 경로: {DATA_DIR}")
    print("=" * 70)

    print("\n[1/2] EasyOCR 모델 로딩 중...")
    reader = easyocr.Reader(["ko", "en"], gpu=False)
    print("  → 로드 완료")

    total_stats = {"total": 0, "updated": 0, "kept": 0, "skipped_no_image": 0, "skipped_no_det": 0}

    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"\n[SKIP] {split}/ 디렉터리 없음")
            continue
        print(f"\n[2/2] {split} 처리 중... ({len(list((split_dir / 'annotations').glob('*.json')))} 파일)")
        stats = process_split(reader, split)
        for k in total_stats:
            total_stats[k] += stats[k]
        print(f"  → 완료: 수정={stats['updated']}, 유지={stats['kept']}, "
              f"이미지없음={stats['skipped_no_image']}, 탐지실패={stats['skipped_no_det']}")

    print("\n" + "=" * 70)
    print("전체 결과 요약")
    print(f"  총 어노테이션: {total_stats['total']}")
    print(f"  좌표 수정됨:   {total_stats['updated']}")
    print(f"  좌표 유지됨:   {total_stats['kept']}")
    print(f"  이미지 없음:   {total_stats['skipped_no_image']}")
    print(f"  탐지 실패:     {total_stats['skipped_no_det']}")
    print(f"\n결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
