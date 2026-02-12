import os
import shutil
from tqdm import tqdm

# ==========================================
# 1. 경로 설정 (사용자 환경에 맞게 수정하세요)
# ==========================================
# 기존에 나눠져 있던 돼지 데이터셋 경로
OLD_PORK_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\pork_dataset" 
# 새로 추가할 돼지 테스트 데이터 경로 (부위별 폴더 구조)
NEW_PORK_TEST_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\pork_test"
# 최종적으로 합쳐서 저장될 경로
FINAL_PORK_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\pork_final"

RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def run_pork_merge_and_split():
    # 1. 모든 소스 경로 정의
    # 기존 데이터셋의 각 세부 폴더들 + 새로운 테스트 폴더
    source_dirs = [
        os.path.join(OLD_PORK_ROOT, 'train'),
        os.path.join(OLD_PORK_ROOT, 'val'),
        os.path.join(OLD_PORK_ROOT, 'test'),
        NEW_PORK_TEST_ROOT
    ]

    # 2. 모든 부위(클래스) 목록 파악
    all_classes = set()
    for d in source_dirs:
        if os.path.exists(d):
            all_classes.update([c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))])
    
    print(f"📂 발견된 부위 목록: {sorted(list(all_classes))}")

    for class_name in all_classes:
        # 이 부위에 해당하는 모든 파일을 담을 리스트
        all_files_path = []
        
        for d in source_dirs:
            class_path = os.path.join(d, class_name)
            if os.path.exists(class_path):
                files = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                         if f.lower().endswith(IMG_EXTENSIONS)]
                all_files_path.extend(files)
        
        if not all_files_path:
            continue

        print(f"📦 {class_name}: 총 {len(all_files_path)}개 합치기 및 분할 시작...")

        # 비율별 카운트 초기화
        counts = {'train': 0, 'val': 0, 'test': 0}
        
        for i, src_path in enumerate(tqdm(all_files_path, desc=f"{class_name} 처리 중")):
            # 실시간 비율 계산하여 배분
            total_now = sum(counts.values()) + 1
            best_split = 'train'
            max_diff = -1e9
            for s in ['train', 'val', 'test']:
                diff = RATIOS[s] - (counts[s] / total_now)
                if diff > max_diff:
                    max_diff = diff
                    best_split = s
            
            # 새 파일명 및 경로 설정
            new_filename = f"{class_name}_{counts[best_split] + 1:04d}.jpg"
            dst_dir = os.path.join(FINAL_PORK_ROOT, best_split, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            
            # 복사 (원본 보존)
            shutil.copy2(src_path, os.path.join(dst_dir, new_filename))
            counts[best_split] += 1

    print("\n✅ 돼지 데이터 통합 및 재분할 완료!")
    print(f"📍 결과 위치: {FINAL_PORK_ROOT}")

if __name__ == "__main__":
    run_pork_merge_and_split()