import os
import shutil
from tqdm import tqdm

# ==========================================
# 1. 경로 및 설정
# ==========================================
# 원본 데이터가 있는 곳 (한글 파일명이 포함된 곳)
MASTER_DATA_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\master_dataset"
# 최종적으로 정리될 곳
FINAL_SPLIT_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\Beef_dataset2"

# 목표 비율 (8:1:1)
RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
# 지원할 확장자 목록 (대소문자 구분 없이 처리)
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ==========================================
# 2. 유틸리티 함수
# ==========================================

def get_current_split_info(class_name):
    """최종 폴더의 현재 상태(파일 목록 및 개수)를 파악합니다."""
    info = {
        'all_filenames': set(),
        'counts': {'train': 0, 'val': 0, 'test': 0}
    }
    
    for s in ['train', 'val', 'test']:
        path = os.path.join(FINAL_SPLIT_ROOT, s, class_name)
        if os.path.exists(path):
            # 이미 변환되어 들어간 파일들 목록 수집
            files = [f for f in os.listdir(path) if f.lower().endswith(IMG_EXTENSIONS)]
            info['all_filenames'].update(files)
            info['counts'][s] = len(files)
    return info

def get_next_available_filename(class_name, existing_filenames):
    """부위명_0001.jpg 형식의 다음 빈 번호를 생성합니다."""
    i = 1
    while True:
        new_name = f"{class_name}_{i:04d}.jpg" # 모든 결과를 .jpg로 통일 (원하면 유지 가능)
        if new_name not in existing_filenames:
            return new_name
        i += 1

# ==========================================
# 3. 실행 로직
# ==========================================

def run_smart_split_logic():
    if not os.path.exists(MASTER_DATA_ROOT):
        print(f"❌ 경로를 찾을 수 없습니다: {MASTER_DATA_ROOT}")
        return

    # 1. 마스터 폴더 내의 부위(폴더) 목록
    class_list = [d for d in os.listdir(MASTER_DATA_ROOT) 
                  if os.path.isdir(os.path.join(MASTER_DATA_ROOT, d))]
    
    print(f"📂 총 {len(class_list)}개의 부위 폴더를 발견했습니다.")

    for class_name in class_list:
        master_class_path = os.path.join(MASTER_DATA_ROOT, class_name)
        
        # 2. 마스터 폴더의 파일들 추출 (공백 제거 및 확장자 체크 강화)
        master_files = [f for f in os.listdir(master_class_path) 
                        if f.lower().strip().endswith(IMG_EXTENSIONS)]
        
        if not master_files:
            print(f"⚠️ {class_name}: 처리할 이미지 파일이 없습니다. (파일명 확인 필요)")
            continue

        # 3. 현재 최종 폴더(Final) 상태 확인
        split_info = get_current_split_info(class_name)
        
        print(f"📦 {class_name}: 총 {len(master_files)}개 데이터 처리 시작...")

        for filename in tqdm(master_files, desc=f"{class_name} 배분 중"):
            # 4. 실시간으로 비율이 가장 부족한 폴더 찾기
            counts = split_info['counts']
            total = sum(counts.values()) + 1
            
            best_split = 'train'
            max_diff = -1e9 
            
            for s in ['train', 'val', 'test']:
                diff = RATIOS[s] - (counts[s] / total)
                if diff > max_diff:
                    max_diff = diff
                    best_split = s
            
            # 5. 새로운 파일명 결정
            new_filename = get_next_available_filename(class_name, split_info['all_filenames'])
            
            # 6. 파일 복사 실행
            src = os.path.join(master_class_path, filename)
            dst_dir = os.path.join(FINAL_SPLIT_ROOT, best_split, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            
            dst_path = os.path.join(dst_dir, new_filename)
            shutil.copy2(src, dst_path)
            
            # 7. 정보 업데이트
            split_info['all_filenames'].add(new_filename)
            split_info['counts'][best_split] += 1

    print("\n" + "="*50)
    print(f"✨ 스마트 이름 변경 및 데이터 분할이 완료되었습니다!")
    print(f"📍 결과 위치: {FINAL_SPLIT_ROOT}")
    print("="*50)

if __name__ == "__main__":
    run_smart_split_logic()