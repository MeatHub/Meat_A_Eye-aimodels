import os
import torch
import shutil
import random
import numpy as np
from PIL import Image, ImageOps
from transformers import pipeline
import easyocr
from tqdm import tqdm

# ==========================================
# 1. 경로 및 설정 (기존 유지 및 추가)
# ==========================================
RAW_INPUT_FOLDER = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\raw_images\Beef_BottomRound"
MASTER_DATA_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\master_dataset"
FINAL_SPLIT_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final2"

PREFIX = "Beef_BottomRound"
RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

THRESHOLD = 0.35
TARGET_SIZE = 640  # 최종 이미지 사이즈
OCR_CONFIDENCE = 0.4

os.makedirs(os.path.join(MASTER_DATA_ROOT, PREFIX), exist_ok=True)

# ==========================================
# 2. 모델 로드
# ==========================================
device = 0 if torch.cuda.is_available() else -1
detector = pipeline(model="IDEA-Research/grounding-dino-base", task="zero-shot-object-detection", device=device)
reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

# ==========================================
# 3. 유틸리티 함수 (기존 유지)
# ==========================================
def get_current_split_files():
    existing_files = {'train': set(), 'val': set(), 'test': set(), 'all': set()}
    for split in ['train', 'val', 'test']:
        path = os.path.join(FINAL_SPLIT_ROOT, split, PREFIX)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.jpg')]
            existing_files[split] = set(files)
            existing_files['all'].update(files)
    return existing_files

def get_next_vacant_number(master_dir, prefix):
    split_info = get_current_split_files()
    i = 1
    while True:
        filename = f"{prefix}_{i:04d}.jpg"
        master_path = os.path.join(master_dir, filename)
        if not os.path.exists(master_path) and filename not in split_info['all']:
            return i, master_path
        i += 1

# ==========================================
# 4. 핵심 로직: 현실적인 크기로 재구성
# ==========================================
def process_to_realistic_scale(image, box):
    """
    고기 영역을 기반으로 주변 배경을 포함하거나
    이미지 내에서 고기의 비중을 낮춰 현실적인 스케일을 생성합니다.
    """
    img_w, img_h = image.size
    l, t, r, b = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])

    # 1. Loose Crop: 고기 영역의 주변을 20~40% 정도 더 넓게 잡음
    padding_w = (r - l) * random.uniform(0.2, 0.4)
    padding_h = (b - t) * random.uniform(0.2, 0.4)

    new_l = max(0, l - padding_w)
    new_t = max(0, t - padding_h)
    new_r = min(img_w, r + padding_w)
    new_b = min(img_h, b + padding_h)

    meat_crop = image.crop((new_l, new_t, new_r, new_b))

    # 2. Resizing & Padding (고기가 화면의 50~80%만 차지하게 함)
    # 캔버스(TARGET_SIZE)를 만들고 그 안에 고기를 작게 배치
    meat_crop.thumbnail((int(TARGET_SIZE * random.uniform(0.6, 0.8)),
                         int(TARGET_SIZE * random.uniform(0.6, 0.8))), Image.Resampling.LANCZOS)

    # 배경색은 고기 주변의 평균 색상 혹은 회색으로 설정하여 이질감을 줄임
    new_img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (128, 128, 128))

    # 중앙 또는 무작위 위치에 배치
    offset_x = (TARGET_SIZE - meat_crop.size[0]) // 2
    offset_y = (TARGET_SIZE - meat_crop.size[1]) // 2
    new_img.paste(meat_crop, (offset_x, offset_y))

    return new_img

# ==========================================
# 5. 실행 로직
# ==========================================
def run_smart_sync_pipeline():
    print(f":rocket: [Step 1] {PREFIX} 현실적 스케일로 정제 시작...")
    image_files = [f for f in os.listdir(RAW_INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    target_master_dir = os.path.join(MASTER_DATA_ROOT, PREFIX)

    new_crops = []
    for filename in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(RAW_INPUT_FOLDER, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            # "raw beef meat" 탐지
            results = detector(image, candidate_labels=["raw beef meat"], threshold=THRESHOLD)

            for res in results:
                # OCR 필터링 (기존 로직)
                box = res['box']
                l, t, r, b = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
                temp_crop = image.crop((l, t, r, b))
                if any(prob > OCR_CONFIDENCE for (_, _, prob) in reader.readtext(np.array(temp_crop))):
                    continue

                # 현실적인 크기로 변환 적용
                final_img = process_to_realistic_scale(image, res['box'])

                # 빈 번호 찾아서 저장
                _, save_path = get_next_vacant_number(target_master_dir, PREFIX)
                final_img.save(save_path, quality=95)
                new_crops.append(os.path.basename(save_path))

        except Exception as e:
            print(f"Error {filename}: {e}")

    # --- STEP 2: 배분 로직 (기존 유지) ---
    print(f"\n:open_file_folder: [Step 2] 신규 데이터({len(new_crops)}개) 배분 시작...")
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

    print(f"\n:sparkles: 작업 완료! 고기가 너무 크게 찍혔던 데이터들을 현실적인 크기로 재구성하여 채워 넣었습니다.")

if __name__ == "__main__":
    run_smart_sync_pipeline()