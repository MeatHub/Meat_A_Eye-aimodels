import os
import shutil
import random
from tqdm import tqdm

# ==========================================
# 1. 경로 및 설정
# ==========================================
# 원본 데이터 루트 (하위에 train, val, test 폴더가 있고 그 안에 부위 폴더들이 있는 구조)
SOURCE_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final2"
# 최종 결과 경로 (train / val / test 로 나눠서 저장)
FINAL_OUTPUT_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final_v3"

RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# ==========================================
# 2. 유틸 함수
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

# ==========================================
# 3. 분할만 실행 (정제 없이 원본 그대로 복사)
# ==========================================
def run_split_only():
    categories = get_all_categories(SOURCE_ROOT)
    print(f"📁 감지된 부위 리스트 ({len(categories)}개): {categories}")

    for cat in categories:
        print(f"\n--- [작업] 부위: {cat} ---")

        # 해당 부위의 모든 이미지 경로 수집 (train/val/test 통합)
        all_img_paths = []
        for split in ['train', 'val', 'test']:
            split_cat_path = os.path.join(SOURCE_ROOT, split, cat)
            if os.path.exists(split_cat_path):
                imgs = [os.path.join(split_cat_path, f) for f in os.listdir(split_cat_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_img_paths.extend(imgs)

        if not all_img_paths:
            print(f"  > 이미지 없음, 스킵.")
            continue

        print(f"  > 총 {len(all_img_paths)}개 파일. 비율대로 나눠 복사 중...")
        random.shuffle(all_img_paths)

        for img_path in tqdm(all_img_paths, desc=cat):
            rand_val = random.random()
            if rand_val < RATIOS['train']:
                split_type = 'train'
            elif rand_val < (RATIOS['train'] + RATIOS['val']):
                split_type = 'val'
            else:
                split_type = 'test'

            target_dir = os.path.join(FINAL_OUTPUT_ROOT, split_type, cat)
            os.makedirs(target_dir, exist_ok=True)
            filename = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(target_dir, filename))

    print(f"\n✅ 완료. 결과: {FINAL_OUTPUT_ROOT}")

if __name__ == "__main__":
    run_split_only()
