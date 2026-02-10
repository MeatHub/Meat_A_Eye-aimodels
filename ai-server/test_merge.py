import os
import shutil
from tqdm import tqdm

# ==========================================
# 1. 경로 설정 (팀장님 환경에 맞춰 수정)
# ==========================================
# 분할된 데이터셋의 test 폴더 위치
TEST_SOURCE_ROOT = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\dataset_final\test"
# Grad-CAM 테스트를 위해 하나로 모을 폴더
MERGED_TARGET_DIR = r"D:\ahy\Projects\meathub\Meat_A_Eye-aimodels\data\test_images"

# ==========================================
# 2. 실행 로직
# ==========================================
def merge_test_images():
    # 대상 폴더 생성
    if not os.path.exists(MERGED_TARGET_DIR):
        os.makedirs(MERGED_TARGET_DIR)
        print(f":file_folder: 폴더 생성 완료: {MERGED_TARGET_DIR}")

    # test 폴더 내의 부위별 폴더 리스트 가져오기
    class_list = [d for d in os.listdir(TEST_SOURCE_ROOT) if os.path.isdir(os.path.join(TEST_SOURCE_ROOT, d))]

    print(f":rocket: 총 {len(class_list)}개 부위의 테스트 데이터를 병합합니다.")

    total_merged = 0
    for class_name in class_list:
        class_path = os.path.join(TEST_SOURCE_ROOT, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in tqdm(images, desc=f"Merging {class_name}"):
            src_path = os.path.join(class_path, img_name)

            # 파일명이 겹칠 수 있으므로 부위명을 붙여서 저장 (예: Beef_Tenderloin_0001.jpg)
            # 이미 부위명이 붙어있다면 그대로 유지, 없다면 붙여줍니다.
            if class_name in img_name:
                target_name = img_name
            else:
                target_name = f"{class_name}_{img_name}"

            dst_path = os.path.join(MERGED_TARGET_DIR, target_name)

            # 파일 복사 (원본 보존)
            shutil.copy2(src_path, dst_path)
            total_merged += 1

    print("\n" + "="*50)
    print(f":sparkles: 병합 완료! 총 {total_merged}개의 이미지가 준비되었습니다.")
    print(f":round_pushpin: 위치: {MERGED_TARGET_DIR}")
    print("="*50)

if __name__ == "__main__":
    merge_test_images()