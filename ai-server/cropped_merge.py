import os
from pathlib import Path
# Hugging Face / transformers 내부 진행 바로 인해 '프로세싱'이 두 번 보이는 것 방지
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings
import logging
import torch
import shutil
import random
import numpy as np
from PIL import Image
from transformers import pipeline
import easyocr
from tqdm import tqdm

# MPS(Mac)에서는 pin_memory 미지원 → PyTorch DataLoader 경고 무시 (동작에는 영향 없음)
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")
# Hugging Face pipeline 내부 진행 로그 비활성화 → tqdm 하나만 보이도록
logging.getLogger("transformers.pipelines.base").setLevel(logging.WARNING)

# ==========================================
# 1. 경로 설정 (팀장님 환경)
# ==========================================
# [중요] 부위 바꿀 때 아래 두 개를 반드시 같은 부위로 맞출 것!
#       입력 폴더 = raw_images/{부위}, PREFIX = master_dataset 저장 폴더명
BASE = Path(__file__).resolve().parent.parent / "data"
RAW_INPUT_FOLDER = BASE / "raw_images" / "Pork_Loin"   # 정제할 원본 이미지 폴더 (부위별로 변경)
MASTER_DATA_ROOT = BASE / "master_dataset"
FINAL_SPLIT_ROOT = BASE / "dataset_final"

PREFIX = "Pork_Loin"   # master_dataset 안에 만들어질 폴더명 (RAW_INPUT_FOLDER 부위와 동일하게)
# 목표 비율 (8:1:1)
RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
# True: 정제 후 dataset_final로 이동, False: master_dataset에만 저장 (압축 후 구글 드라이브 업로드용)
SEND_TO_DATASET_FINAL = False

# 필터링 설정
THRESHOLD = 0.35
MIN_SIZE = 640
OCR_CONFIDENCE = 0.4

os.makedirs(os.path.join(MASTER_DATA_ROOT, PREFIX), exist_ok=True)

# ==========================================
# 2. 모델 로드 (Mac M2: MPS 시도 → 실패 시 CPU, NVIDIA: CUDA)
# ==========================================
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    _device = "mps"
elif torch.cuda.is_available():
    _device = 0
else:
    _device = -1
try:
    detector = pipeline(model="IDEA-Research/grounding-dino-base", task="zero-shot-object-detection", device=_device)
    print(f"[cropped_merge] 추론 디바이스: {_device}")
except Exception as e:
    print(f"[cropped_merge] MPS/CUDA 로드 실패, CPU 사용: {e}")
    detector = pipeline(model="IDEA-Research/grounding-dino-base", task="zero-shot-object-detection", device=-1)
# EasyOCR는 CUDA만 지원하므로 Mac에서는 CPU 사용
reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

# ==========================================
# 3. 스마트 동기화 함수들
# ==========================================

def get_current_split_files():
    """현재 Train/Val/Test 폴더에 실제로 존재하는 파일 목록을 가져옵니다."""
    existing_files = {'train': set(), 'val': set(), 'test': set(), 'all': set()}
    for split in ['train', 'val', 'test']:
        path = os.path.join(FINAL_SPLIT_ROOT, split, PREFIX)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.jpg')]
            existing_files[split] = set(files)
            existing_files['all'].update(files)
    return existing_files

def get_next_vacant_number(master_dir, prefix):
    """마스터와 스플릿 폴더 모두를 뒤져서 비어있는 가장 빠른 번호를 찾습니다."""
    split_info = get_current_split_files()
    i = 1
    while True:
        filename = f"{prefix}_{i:04d}.jpg"
        master_path = os.path.join(master_dir, filename)
        # 마스터에도 없고, 어떤 스플릿 폴더에도 없는 번호가 '진짜 빈자리'
        if not os.path.exists(master_path) and filename not in split_info['all']:
            return i, master_path
        i += 1


def average_hash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """
    간단한 aHash(average hash) 구현.
    - 이미지를 작은 그레이스케일로 리사이즈 후
    - 픽셀 평균보다 크면 1, 아니면 0
    """
    img = image.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = np.asarray(img, dtype=np.float32)
    mean = pixels.mean()
    return (pixels > mean).astype(np.uint8).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """두 해시(0/1 배열) 사이의 해밍 거리."""
    # 길이가 다른 경우를 방지
    if a.shape != b.shape:
        return 9999
    return int(np.count_nonzero(a != b))


def remove_near_duplicate_images(folder: str, hamming_thresh: int = 5) -> int:
    """
    master_dataset/{PREFIX} 안에서 '연타로 찍힌 거의 같은 사진'을 정리.
    - 파일명을 기준으로 정렬 후, 이웃한 이미지끼리만 비교 (연사 기준)
    - average hash의 해밍 거리가 hamming_thresh 이하이면 중복으로 간주하고 뒤에 것을 삭제

    Returns:
        삭제된 이미지 개수
    """
    if not os.path.exists(folder):
        return 0

    files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    files.sort()

    if len(files) < 2:
        return 0

    prev_hash = None
    prev_name = None
    deleted = 0

    print(f"\n🧹 중복/연사 이미지 정리 시작 ({len(files)}장 대상, 기준={hamming_thresh})")

    for name in files:
        path = os.path.join(folder, name)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"   ⚠️ {name} 열기 실패, 건너뜀: {e}")
            continue

        curr_hash = average_hash(img)

        if prev_hash is not None:
            dist = hamming_distance(prev_hash, curr_hash)
            # 해밍 거리가 작을수록 더 비슷한 이미지
            if dist <= hamming_thresh:
                try:
                    os.remove(path)
                    deleted += 1
                    print(f"   ❌ 중복 삭제: {name} (기준: {prev_name}, 거리={dist})")
                    continue  # prev_hash 유지 (가장 앞의 것만 남김)
                except Exception as e:
                    print(f"   ⚠️ {name} 삭제 실패: {e}")
                    # 삭제 실패 시에는 해시를 갱신해 중복 연쇄를 막음

        prev_hash = curr_hash
        prev_name = name

    print(f"🧹 중복/연사 정리 완료: {deleted}장 삭제")
    return deleted

# ==========================================
# 4. 통합 실행 로직
# ==========================================
def run_smart_sync_pipeline():
    # --- STEP 1: 마스터 동기화 및 크롭 ---
    print(f"🚀 [Step 1] {PREFIX} 정제 및 빈자리 채우기 시작...")
    # Path 객체를 str로 변환하여 os.listdir 사용
    raw_folder_str = str(RAW_INPUT_FOLDER)
    image_files = [f for f in os.listdir(raw_folder_str) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📁 원본 이미지 파일: {len(image_files)}개 발견")
    target_master_dir = os.path.join(MASTER_DATA_ROOT, PREFIX)
    
    # 통계 추적
    stats = {
        'total_images': len(image_files),
        'no_detection': 0,  # Grounding DINO에서 아무것도 못 찾음
        'too_small': 0,     # 크기가 MIN_SIZE 미만
        'has_text': 0,      # OCR로 텍스트 감지됨
        'saved': 0          # 최종 저장됨
    }
    
    new_crops = []
    for filename in tqdm(image_files, desc="이미지 정제"):
        img_path = os.path.join(raw_folder_str, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            results = detector(image, candidate_labels=["raw pork meat"], threshold=THRESHOLD)
            
            if not results:
                stats['no_detection'] += 1
                continue
            
            found_valid_crop = False
            for res in results:
                box = res['box']
                l, t, r, b = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
                width = r - l
                height = b - t
                
                if width < MIN_SIZE or height < MIN_SIZE:
                    stats['too_small'] += 1
                    continue
                
                cropped_img = image.crop((l, t, r, b))
                ocr_results = reader.readtext(np.array(cropped_img))
                if any(prob > OCR_CONFIDENCE for (_, _, prob) in ocr_results):
                    stats['has_text'] += 1
                    continue

                # 빈 번호 찾아서 마스터에 임시 저장
                _, save_path = get_next_vacant_number(target_master_dir, PREFIX)
                cropped_img.save(save_path, quality=100)
                new_crops.append(os.path.basename(save_path))
                stats['saved'] += 1
                found_valid_crop = True
            
        except Exception as e: 
            print(f"❌ Error {filename}: {e}")
    
    # 통계 출력
    print(f"\n{'='*60}")
    print(f"📊 정제 통계:")
    print(f"   전체 이미지: {stats['total_images']}개")
    print(f"   ❌ 탐지 실패 (Grounding DINO): {stats['no_detection']}개")
    print(f"   ❌ 크기 부족 (<{MIN_SIZE}px): {stats['too_small']}개")
    print(f"   ❌ 텍스트 감지 (OCR >{OCR_CONFIDENCE}): {stats['has_text']}개")
    print(f"   ✅ 최종 저장: {stats['saved']}개")
    print(f"{'='*60}")

    # --- STEP 1.5: 정제 결과에서 연사/중복 이미지 제거 ---
    # master_dataset/{PREFIX} 전체를 대상으로 인접한 이미지끼리 aHash 비교
    removed = remove_near_duplicate_images(str(target_master_dir), hamming_thresh=5)
    if removed > 0:
        print(f"   ➕ 중복/연사 제거 후 남은 이미지 수: {stats['saved']}개 - {removed}개 (삭제) = {stats['saved'] - removed}개 예상")

    # --- STEP 2: 신규 파일을 dataset_final로 배분 (SEND_TO_DATASET_FINAL=True일 때만) ---
    if SEND_TO_DATASET_FINAL:
        print(f"\n📂 [Step 2] 신규 데이터({len(new_crops)}개) dataset_final 배분 시작...")
        split_info = get_current_split_files()
        for filename in new_crops:
            master_path = os.path.join(target_master_dir, filename)
            current_counts = {k: len(split_info[k]) for k in ['train', 'val', 'test']}
            total = sum(current_counts.values()) + 1
            best_split = 'train'
            max_diff = -1
            for s in ['train', 'val', 'test']:
                diff = RATIOS[s] - (current_counts[s] / total)
                if diff > max_diff:
                    max_diff = diff
                    best_split = s
            target_path = os.path.join(FINAL_SPLIT_ROOT, best_split, PREFIX)
            os.makedirs(target_path, exist_ok=True)
            shutil.move(master_path, os.path.join(target_path, filename))
            split_info[best_split].add(filename)
        print(f"\n✨ 작업 완료! 신규 데이터가 dataset_final(train/val/test)로 배분되었습니다.")
    else:
        print(f"\n✨ [Step 1만 완료] 정제된 데이터 {len(new_crops)}개가 master_dataset/{PREFIX}/ 에만 저장되었습니다.")
        print(f"   → 압축 후 구글 드라이브에 올리려면 master_dataset 폴더를 zip 하세요.")
        print(f"   → 나중에 dataset_final로 배분하려면 SEND_TO_DATASET_FINAL=True 로 바꾼 뒤 split.py 실행.")

if __name__ == "__main__":
    run_smart_sync_pipeline()