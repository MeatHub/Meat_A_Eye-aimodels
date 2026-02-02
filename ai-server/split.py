import os
import shutil
import random
from tqdm import tqdm

# ==========================================
# 1. 경로 설정
# ==========================================
# 모든 부위가 폴더별로 모여있는 곳
MASTER_DATA_ROOT = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\data\master_dataset"
# 최종적으로 학습에 사용될 분할 폴더
FINAL_SPLIT_ROOT = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final"

# 목표 비율 (8:1:1)
RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# ==========================================
# 2. 유틸리티 함수
# ==========================================

def get_current_split_stats(class_name):
    """특정 부위의 최종 분할 폴더 현황을 파악합니다."""
    existing_all = set()
    counts = {'train': 0, 'val': 0, 'test': 0}
    
    for s in ['train', 'val', 'test']:
        path = os.path.join(FINAL_SPLIT_ROOT, s, class_name)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.jpg')]
            existing_all.update(files)
            counts[s] = len(files)
    return existing_all, counts

# ==========================================
# 3. 분할 실행 로직
# ==========================================
def run_only_split_logic():
    # 마스터 폴더 내의 모든 부위(폴더) 목록 가져오기
    class_list = [d for d in os.listdir(MASTER_DATA_ROOT) if os.path.isdir(os.path.join(MASTER_DATA_ROOT, d))]
    
    print(f"📂 총 {len(class_list)}개의 부위 폴더를 발견했습니다.")

    for class_name in class_list:
        master_class_path = os.path.join(MASTER_DATA_ROOT, class_name)
        master_files = [f for f in os.listdir(master_class_path) if f.endswith('.jpg')]
        
        # 현재 배분된 상태 확인
        existing_files, counts = get_current_split_stats(class_name)
        
        # 마스터에는 있지만 아직 분할 폴더에는 없는 '새로운 파일'들 찾기
        new_files = [f for f in master_files if f not in existing_files]
        
        if not new_files:
            print(f"✅ {class_name}: 추가할 새 데이터가 없습니다.")
            continue

        print(f"📦 {class_name}: 신규 데이터 {len(new_files)}개 배분 시작...")

        # 신규 데이터 배분
        for filename in tqdm(new_files, desc=f"{class_name} 배분 중"):
            # 실시간으로 비율이 가장 부족한 폴더 찾기
            total = sum(counts.values()) + 1
            best_split = 'train'
            max_diff = -1
            
            for s in ['train', 'val', 'test']:
                diff = RATIOS[s] - (counts[s] / total)
                if diff > max_diff:
                    max_diff = diff
                    best_split = s
            
            # 파일 복사 (Master는 보존하고 Final로 보냄)
            src = os.path.join(master_class_path, filename)
            dst_dir = os.path.join(FINAL_SPLIT_ROOT, best_split, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            
            shutil.copy2(src, os.path.join(dst_dir, filename))
            
            # 카운트 업데이트
            counts[best_split] += 1

    print("\n" + "="*50)
    print(f"✨ 모든 부위의 데이터 분할 및 비율 동기화가 완료되었습니다!")
    print(f"📍 위치: {FINAL_SPLIT_ROOT}")
    print("="*50)

if __name__ == "__main__":
    run_only_split_logic()