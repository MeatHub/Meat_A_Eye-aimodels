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
# 1. 경로 및 설정 (자동화 중심)
# ==========================================
# 원본 데이터 루트 (하위에 train, val, test 폴더가 있고 그 안에 부위 폴더들이 있는 구조)
SOURCE_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final2"
# 처리된 결과를 임시 저장할 마스터 경로
MASTER_DATA_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\master_dataset_refined"
# 최종 결과 경로
FINAL_OUTPUT_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final_v3"

RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
TARGET_SIZE = 640
THRESHOLD = 0.35
OCR_CONFIDENCE = 0.4

# ==========================================
# 2. 모델 로드
# ==========================================
device = 0 if torch.cuda.is_available() else -1
detector = pipeline(model="IDEA-Research/grounding-dino-base", task="zero-shot-object-detection", device=device)
reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

# ==========================================
# 3. 핵심 로직 함수
# ==========================================

def get_all_categories(root_path):
    """train, val, test 폴더를 뒤져서 존재하는 모든 부위명(폴더명) 세트를 가져옵니다."""
    categories = set()
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_path, split)
        if os.path.exists(split_path):
            folders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
            categories.update(folders)
    return sorted(list(categories))

def process_tight_crop(image, box):
    """고기 영역을 빡세게(타이트하게) 크롭하여 640x640 캔버스에 배치"""
    img_w, img_h = image.size
    l, t, r, b = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])

    # 패딩 최소화 (5~12%)
    p_w = (r - l) * random.uniform(0.05, 0.12)
    p_h = (b - t) * random.uniform(0.05, 0.12)

    meat_crop = image.crop((max(0, l-p_w), max(0, t-p_h), min(img_w, r+p_w), min(img_h, b+p_h)))

    # 캔버스 대비 고기 크기를 크게 (75~90%)
    meat_crop.thumbnail((int(TARGET_SIZE * random.uniform(0.75, 0.9)),
                         int(TARGET_SIZE * random.uniform(0.75, 0.9))), Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (128, 128, 128))
    offset = ((TARGET_SIZE - meat_crop.size[0]) // 2, (TARGET_SIZE - meat_crop.size[1]) // 2)
    new_img.paste(meat_crop, offset)
    return new_img

# ==========================================
# 4. 전체 파이프라인 실행
# ==========================================
def run_full_auto_pipeline():
    # 1. 모든 부위(카테고리) 파악
    categories = get_all_categories(SOURCE_ROOT)
    print(f":open_file_folder: 감지된 부위 리스트 ({len(categories)}개): {categories}")

    for cat in categories:
        print(f"\n#--- [작업 시작] 부위명: {cat} ---#")

        # 각 부위별 마스터 폴더 생성
        cat_master_dir = os.path.join(MASTER_DATA_ROOT, cat)
        os.makedirs(cat_master_dir, exist_ok=True)

        # 2. 해당 부위의 모든 이미지 경로 수집 (train/val/test 통합)
        all_img_paths = []
        for split in ['train', 'val', 'test']:
            split_cat_path = os.path.join(SOURCE_ROOT, split, cat)
            if os.path.exists(split_cat_path):
                imgs = [os.path.join(split_cat_path, f) for f in os.listdir(split_cat_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_img_paths.extend(imgs)

        print(f"  > 총 {len(all_img_paths)}개의 원본 파일 발견. 정제 중...")

        refined_files = []
        # 3. 이미지별 탐지 및 빡센 크롭 실행
        for idx, img_path in enumerate(tqdm(all_img_paths, desc=f"Processing {cat}")):
            try:
                image = Image.open(img_path).convert("RGB")
                results = detector(image, candidate_labels=["raw beef meat"], threshold=THRESHOLD)

                if not results: continue

                # 가장 큰 객체 선택
                res = max(results, key=lambda x: (x['box']['xmax']-x['box']['xmin'])*(x['box']['ymax']-x['box']['ymin']))

                # OCR 필터링 (텍스트 있는 데이터 제외)
                box = res['box']
                check_crop = image.crop((int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])))
                if any(p > OCR_CONFIDENCE for (_, _, p) in reader.readtext(np.array(check_crop))):
                    continue

                # 정제된 이미지 생성 및 저장
                final_img = process_tight_crop(image, res['box'])
                new_name = f"{cat}_refined_{idx:04d}.jpg"
                save_path = os.path.join(cat_master_dir, new_name)
                final_img.save(save_path, quality=95)
                refined_files.append(new_name)

            except Exception as e:
                print(f"  ! Error {img_path}: {e}")

        # 4. 부위별 재분배 (Split)
        print(f"  > 정제 완료 ({len(refined_files)}개). 다시 Train/Val/Test로 나누는 중...")
        random.shuffle(refined_files)

        for i, filename in enumerate(refined_files):
            # 비율 계산
            rand_val = random.random()
            if rand_val < RATIOS['train']: split_type = 'train'
            elif rand_val < (RATIOS['train'] + RATIOS['val']): split_type = 'val'
            else: split_type = 'test'

            target_path = os.path.join(FINAL_OUTPUT_ROOT, split_type, cat)
            os.makedirs(target_path, exist_ok=True)

            # 마스터에서 최종 경로로 복사
            shutil.copy2(os.path.join(cat_master_dir, filename), os.path.join(target_path, filename))

    print(f"\n:sparkles: [모든 공정 완료] 최종 결과가 {FINAL_OUTPUT_ROOT} 에 저장되었습니다.")

if __name__ == "__main__":
    run_full_auto_pipeline()