"""
Step 0: 원본 어노테이션 해상도 교정
===================================
이미지 회전/리사이즈로 인해 image_size 와 bbox 가
현재 이미지와 안 맞는 어노테이션을 자동 교정합니다.

- SCALE 변경: bbox 좌표를 비례 변환
- SWAP 변경(회전): CW/CCW 두 방향 중 이미지 중앙에
  가까운 변환을 채택 (다수결 기반 방향 판별)

사용법:
  python step0_fix_annotations.py          # 검사만 (dry-run)
  python step0_fix_annotations.py --apply  # 실제 교정 적용
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

# ── 경로 설정 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # Meat_A_Eye-aimodels
DATA_DIR = PROJECT_ROOT / "data" / "Paddle_train_images"
SPLITS = ["train", "val", "test"]


# ── 좌표 변환 함수 (step1 과 동일) ──────────────────────────
def scale_bbox(bbox: List[int], old_w: int, old_h: int,
               new_w: int, new_h: int) -> List[int]:
    sx = new_w / old_w
    sy = new_h / old_h
    return [int(bbox[0] * sx), int(bbox[1] * sy),
            int(bbox[2] * sx), int(bbox[3] * sy)]


def rotate_bbox_cw(bbox: List[int], old_w: int, old_h: int) -> List[int]:
    """90° CW: (x,y)→(old_h-1-y, x)"""
    x1, y1, x2, y2 = bbox
    new_x1 = old_h - 1 - y2
    new_y1 = x1
    new_x2 = old_h - 1 - y1
    new_y2 = x2
    return [min(new_x1, new_x2), min(new_y1, new_y2),
            max(new_x1, new_x2), max(new_y1, new_y2)]


def rotate_bbox_ccw(bbox: List[int], old_w: int, old_h: int) -> List[int]:
    """90° CCW: (x,y)→(y, old_w-1-x)"""
    x1, y1, x2, y2 = bbox
    new_x1 = y1
    new_y1 = old_w - 1 - x2
    new_x2 = y2
    new_y2 = old_w - 1 - x1
    return [min(new_x1, new_x2), min(new_y1, new_y2),
            max(new_x1, new_x2), max(new_y1, new_y2)]


def get_image_size(img_path: Path) -> Tuple[int, int]:
    """이미지 파일에서 (width, height) 반환 (PIL 사용)"""
    from PIL import Image
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def bbox_center_relative(bbox: List[int], w: int, h: int) -> Tuple[float, float]:
    """bbox 중심의 상대 좌표 (0~1)"""
    cx = (bbox[0] + bbox[2]) / 2 / max(w, 1)
    cy = (bbox[1] + bbox[3]) / 2 / max(h, 1)
    return cx, cy


def pick_rotation_direction(bboxes: List[List[int]], ann_w: int, ann_h: int,
                            actual_w: int, actual_h: int) -> str:
    """
    여러 어노테이션의 bbox를 CW/CCW 로 변환해보고,
    이전 상대 위치와 더 가까운 방향을 선택합니다.
    
    육류 라벨 이력번호는 보통 이미지 중앙~하단에 위치하므로,
    회전 후에도 비슷한 상대 위치에 있어야 합니다.
    """
    if not bboxes:
        return "cw"  # 기본값
    
    cw_score = 0
    ccw_score = 0
    
    for bbox in bboxes:
        # 원래 상대 위치
        orig_cx, orig_cy = bbox_center_relative(bbox, ann_w, ann_h)
        
        # CW 변환 후 상대 위치
        cw_box = rotate_bbox_cw(bbox, ann_w, ann_h)
        cw_cx, cw_cy = bbox_center_relative(cw_box, actual_w, actual_h)
        
        # CCW 변환 후 상대 위치
        ccw_box = rotate_bbox_ccw(bbox, ann_w, ann_h)
        ccw_cx, ccw_cy = bbox_center_relative(ccw_box, actual_w, actual_h)
        
        # 원래 상대 위치와의 거리 (가까울수록 좋음)
        cw_dist = ((orig_cx - cw_cx) ** 2 + (orig_cy - cw_cy) ** 2) ** 0.5
        ccw_dist = ((orig_cx - ccw_cx) ** 2 + (orig_cy - ccw_cy) ** 2) ** 0.5
        
        if cw_dist < ccw_dist:
            cw_score += 1
        else:
            ccw_score += 1
    
    return "cw" if cw_score >= ccw_score else "ccw"


def main():
    apply = "--apply" in sys.argv
    
    print("=" * 70)
    print("Step 0: 원본 어노테이션 해상도 교정")
    if not apply:
        print("  [!] DRY-RUN 모드 (실제 변경 없음). --apply 로 실행하세요.")
    print(f"  데이터 경로: {DATA_DIR}")
    print("=" * 70)
    
    total = {"checked": 0, "ok": 0, "scale_fixed": 0, "swap_fixed": 0, "no_image": 0}
    
    for split in SPLITS:
        ann_dir = DATA_DIR / split / "annotations"
        img_dir = DATA_DIR / split / "images"
        if not ann_dir.exists():
            continue
        
        print(f"\n[{split}] 처리 중...")
        
        for jf in sorted(ann_dir.glob("*.json")):
            total["checked"] += 1
            
            with open(jf, encoding="utf-8") as f:
                ann = json.load(f)
            
            img_path = img_dir / ann["file_name"]
            if not img_path.exists():
                total["no_image"] += 1
                continue
            
            actual_w, actual_h = get_image_size(img_path)
            ann_w, ann_h = ann.get("image_size", [actual_w, actual_h])
            
            if ann_w == actual_w and ann_h == actual_h:
                total["ok"] += 1
                continue
            
            # 방향이 같으면 → SCALE
            same_orientation = (ann_w >= ann_h) == (actual_w >= actual_h)
            
            if same_orientation:
                tag = "SCALE"
                for obj in ann["objects"]:
                    obj["bbox"] = scale_bbox(obj["bbox"], ann_w, ann_h, actual_w, actual_h)
                ann["image_size"] = [actual_w, actual_h]
                total["scale_fixed"] += 1
            else:
                # 방향이 다르면 → SWAP (회전)
                tag = "SWAP"
                bboxes = [obj["bbox"] for obj in ann["objects"]]
                direction = pick_rotation_direction(bboxes, ann_w, ann_h, actual_w, actual_h)
                
                rot_w, rot_h = ann_h, ann_w  # 회전 후 예상 크기
                
                for obj in ann["objects"]:
                    if direction == "cw":
                        new_box = rotate_bbox_cw(obj["bbox"], ann_w, ann_h)
                    else:
                        new_box = rotate_bbox_ccw(obj["bbox"], ann_w, ann_h)
                    
                    # 회전 후 추가 스케일 필요시
                    if rot_w != actual_w or rot_h != actual_h:
                        new_box = scale_bbox(new_box, rot_w, rot_h, actual_w, actual_h)
                    
                    obj["bbox"] = new_box
                
                ann["image_size"] = [actual_w, actual_h]
                total["swap_fixed"] += 1
            
            print(f"  [{tag}] {jf.name}: [{ann_w},{ann_h}] → [{actual_w},{actual_h}]"
                  + (f" ({direction})" if not same_orientation else ""))
            
            if apply:
                with open(jf, "w", encoding="utf-8") as f:
                    json.dump(ann, f, ensure_ascii=False, indent=4)
    
    print("\n" + "=" * 70)
    print("결과 요약")
    print(f"  검사: {total['checked']}")
    print(f"  정상: {total['ok']}")
    print(f"  SCALE 교정: {total['scale_fixed']}")
    print(f"  SWAP  교정: {total['swap_fixed']}")
    print(f"  이미지 없음: {total['no_image']}")
    if apply:
        print("\n  [OK] 교정 적용 완료!")
    else:
        print(f"\n  [!] 총 {total['scale_fixed'] + total['swap_fixed']}개 교정 필요.")
        print("  실제 적용하려면:  python step0_fix_annotations.py --apply")
    print("=" * 70)


if __name__ == "__main__":
    main()
