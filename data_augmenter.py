"""
데이터 증강 도구
이미지를 변형하여 더 많은 학습 데이터를 생성합니다.
"""

import cv2
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple

class DataAugmenter:
    """데이터 증강 클래스"""
    
    def __init__(self, input_dir: str = "data/labeled/images", 
                 output_dir: str = "data/augmented",
                 label_file: str = "data/labeled/labels.txt"):
        """
        초기화
        
        Args:
            input_dir: 원본 이미지 폴더
            output_dir: 증강된 이미지 저장 폴더
            label_file: 라벨 파일 경로
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_file = Path(label_file)
        
        # 라벨 로드
        self.labels = {}
        self.load_labels()
    
    def load_labels(self):
        """라벨 파일 로드"""
        if self.label_file.exists():
            with open(self.label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '|' in line:
                        filename, label = line.split('|', 1)
                        self.labels[filename] = label
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """이미지 회전"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height), 
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """밝기 조정 (factor > 1: 밝게, < 1: 어둡게)"""
        adjusted = cv2.convertScaleAbs(image, alpha=1, beta=int(255 * (factor - 1)))
        return adjusted
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """대비 조정 (factor > 1: 높게, < 1: 낮게)"""
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
    
    def add_noise(self, image: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """가우시안 노이즈 추가"""
        noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return noisy
    
    def apply_blur(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """블러 적용"""
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def augment_image(self, image: np.ndarray, augmentation_type: str) -> np.ndarray:
        """
        이미지 증강 적용
        
        Args:
            image: 원본 이미지
            augmentation_type: 증강 타입
                - 'rotate_small': 작은 각도 회전
                - 'rotate_medium': 중간 각도 회전
                - 'bright_dark': 어둡게
                - 'bright_light': 밝게
                - 'contrast_low': 낮은 대비
                - 'contrast_high': 높은 대비
                - 'noise': 노이즈 추가
                - 'blur': 블러 적용
        """
        if augmentation_type == 'rotate_small':
            angle = random.uniform(-5, 5)
            return self.rotate_image(image, angle)
        
        elif augmentation_type == 'rotate_medium':
            angle = random.uniform(-10, 10)
            return self.rotate_image(image, angle)
        
        elif augmentation_type == 'bright_dark':
            factor = random.uniform(0.7, 0.9)
            return self.adjust_brightness(image, factor)
        
        elif augmentation_type == 'bright_light':
            factor = random.uniform(1.1, 1.3)
            return self.adjust_brightness(image, factor)
        
        elif augmentation_type == 'contrast_low':
            factor = random.uniform(0.7, 0.9)
            return self.adjust_contrast(image, factor)
        
        elif augmentation_type == 'contrast_high':
            factor = random.uniform(1.1, 1.3)
            return self.adjust_contrast(image, factor)
        
        elif augmentation_type == 'noise':
            factor = random.uniform(0.05, 0.15)
            return self.add_noise(image, factor)
        
        elif augmentation_type == 'blur':
            kernel = random.choice([3, 5])
            return self.apply_blur(image, kernel)
        
        else:
            return image
    
    def augment_dataset(self, augmentations_per_image: int = 5):
        """
        데이터셋 증강
        
        Args:
            augmentations_per_image: 이미지당 생성할 증강 이미지 개수
        """
        # 증강 타입 리스트
        augmentation_types = [
            'rotate_small', 'rotate_medium',
            'bright_dark', 'bright_light',
            'contrast_low', 'contrast_high',
            'noise', 'blur'
        ]
        
        # 이미지 파일 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
        
        if not image_files:
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {self.input_dir}")
            return
        
        print(f"📊 총 {len(image_files)}개 이미지 증강 시작")
        print(f"이미지당 {augmentations_per_image}개 생성")
        print("=" * 60)
        
        # 증강된 라벨 파일
        aug_label_file = self.output_dir / "labels.txt"
        aug_labels = []
        
        total_created = 0
        
        for img_file in image_files:
            # 원본 이미지 로드
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"❌ 이미지를 읽을 수 없습니다: {img_file}")
                continue
            
            # 원본 라벨 찾기
            original_label = self.labels.get(img_file.name, "")
            
            # 원본 이미지 복사 (증강된 데이터에도 포함)
            base_name = img_file.stem
            ext = img_file.suffix
            
            # 증강 이미지 생성
            for i in range(augmentations_per_image):
                # 랜덤 증강 타입 선택
                aug_type = random.choice(augmentation_types)
                
                # 증강 적용
                augmented = self.augment_image(image, aug_type)
                
                # 파일명 생성
                aug_filename = f"{base_name}_aug{i+1}_{aug_type}{ext}"
                aug_path = self.output_dir / aug_filename
                
                # 저장
                cv2.imwrite(str(aug_path), augmented)
                aug_labels.append(f"{aug_filename}|{original_label}")
                
                total_created += 1
                
                if total_created % 10 == 0:
                    print(f"진행 중... {total_created}개 생성됨")
        
        # 증강된 라벨 파일 저장
        with open(aug_label_file, 'w', encoding='utf-8') as f:
            for label_line in aug_labels:
                f.write(label_line + '\n')
        
        print(f"\n✅ 증강 완료!")
        print(f"   원본: {len(image_files)}개")
        print(f"   증강: {total_created}개")
        print(f"   총합: {len(image_files) + total_created}개")
        print(f"   저장 위치: {self.output_dir}")


if __name__ == "__main__":
    # 사용 예시
    augmenter = DataAugmenter(
        input_dir="data/labeled/images",
        output_dir="data/augmented",
        label_file="data/labeled/labels.txt"
    )
    
    print("\n데이터 증강 도구")
    print("=" * 60)
    
    num = input("이미지당 생성할 증강 이미지 개수 (기본값: 5): ").strip()
    num_aug = int(num) if num.isdigit() else 5
    
    augmenter.augment_dataset(augmentations_per_image=num_aug)
