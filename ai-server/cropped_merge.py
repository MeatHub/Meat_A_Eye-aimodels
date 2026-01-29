import os
import torch
import shutil
import random
import numpy as np
from PIL import Image
from transformers import pipeline
import easyocr
from tqdm import tqdm

# ==========================================
# 1. 경로 설정 (팀장님 환경)
# ==========================================
RAW_INPUT_FOLDER = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\raw_images\Beef_BottomRound"
MASTER_DATA_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\master_dataset"
FINAL_SPLIT_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final"

PREFIX = "Beef_BottomRound"
# 목표 비율 (8:1:1)
RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# 필터링 설정
THRESHOLD = 0.35
MIN_SIZE = 640
OCR_CONFIDENCE = 0.4

os.makedirs(os.path.join(MASTER_DATA_ROOT, PREFIX), exist_ok=True)

# ==========================================
# 2. 모델 로드
# ==========================================
device = 0 if torch.cuda.is_available() else -1
detector = pipeline(model="IDEA-Research/grounding-dino-base", task="zero-shot-object-detection", device=device)
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

# ==========================================
# 4. 통합 실행 로직
# ==========================================
def run_smart_sync_pipeline():
    # --- STEP 1: 마스터 동기화 및 크롭 ---
    print(f"🚀 [Step 1] {PREFIX} 정제 및 빈자리 채우기 시작...")
    image_files = [f for f in os.listdir(RAW_INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    target_master_dir = os.path.join(MASTER_DATA_ROOT, PREFIX)
    
    new_crops = []
    for filename in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(RAW_INPUT_FOLDER, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            results = detector(image, candidate_labels=["raw beef meat"], threshold=THRESHOLD)
            
            for res in results:
                box = res['box']
                l, t, r, b = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
                if (r - l) < MIN_SIZE or (b - t) < MIN_SIZE: continue
                
                cropped_img = image.crop((l, t, r, b))
                if any(prob > OCR_CONFIDENCE for (_, _, prob) in reader.readtext(np.array(cropped_img))): continue

                # 빈 번호 찾아서 마스터에 임시 저장
                _, save_path = get_next_vacant_number(target_master_dir, PREFIX)
                cropped_img.save(save_path, quality=100)
                new_crops.append(os.path.basename(save_path))
        except Exception as e: print(f"Error {filename}: {e}")

    # --- STEP 2: 신규 파일을 기존 폴더에 '배분' ---
    print(f"\n📂 [Step 2] 신규 데이터({len(new_crops)}개) 배분 시작...")
    
    # 현재 배분 상태 확인
    split_info = get_current_split_files()
    
    for filename in new_crops:
        master_path = os.path.join(target_master_dir, filename)
        
        # 어느 폴더에 넣을지 결정 (비율 유지 로직)
        current_counts = {k: len(split_info[k]) for k in ['train', 'val', 'test']}
        total = sum(current_counts.values()) + 1
        
        # 목표 대비 가장 부족한 폴더 찾기
        best_split = 'train'
        max_diff = -1
        for s in ['train', 'val', 'test']:
            diff = RATIOS[s] - (current_counts[s] / total)
            if diff > max_diff:
                max_diff = diff
                best_split = s
        
        # 파일 이동 및 기록 업데이트
        target_path = os.path.join(FINAL_SPLIT_ROOT, best_split, PREFIX)
        os.makedirs(target_path, exist_ok=True)
        shutil.move(master_path, os.path.join(target_path, filename))
        split_info[best_split].add(filename)

    print(f"\n✨ 작업 완료! 팀장님이 삭제한 번호는 건드리지 않고, 신규 데이터만 빈 칸에 채워 넣었습니다.")

if __name__ == "__main__":
    run_smart_sync_pipeline()