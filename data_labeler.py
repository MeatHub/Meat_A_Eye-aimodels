"""
데이터 라벨링 도구
이미지에 대한 정답(이력번호)을 입력하고 저장합니다.
"""

import cv2
import os
from pathlib import Path
from typing import Dict, Optional

class DataLabeler:
    """데이터 라벨링 클래스"""
    
    def __init__(self, image_dir: str = "data/raw", label_file: str = "data/labeled/labels.txt"):
        """
        초기화
        
        Args:
            image_dir: 이미지가 있는 폴더
            label_file: 라벨을 저장할 파일 경로
        """
        self.image_dir = Path(image_dir)
        self.label_file = Path(label_file)
        
        # 라벨 파일이 있는 폴더 생성
        self.label_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 기존 라벨 로드
        self.labels: Dict[str, str] = {}
        self.load_labels()
    
    def load_labels(self):
        """기존 라벨 파일 로드"""
        if self.label_file.exists():
            with open(self.label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '|' in line:
                        filename, label = line.split('|', 1)
                        self.labels[filename] = label
            print(f"✅ 기존 라벨 {len(self.labels)}개 로드됨")
    
    def save_labels(self):
        """라벨 파일 저장"""
        with open(self.label_file, 'w', encoding='utf-8') as f:
            for filename, label in sorted(self.labels.items()):
                f.write(f"{filename}|{label}\n")
        print(f"✅ 라벨 저장됨: {self.label_file}")
    
    def label_images(self, use_ocr_suggestion: bool = True):
        """
        이미지 라벨링 시작
        
        Args:
            use_ocr_suggestion: EasyOCR로 먼저 인식하여 제안할지 여부
        """
        # 이미지 파일 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.image_dir.glob(f"*{ext}"))
            image_files.extend(self.image_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {self.image_dir}")
            return
        
        # 아직 라벨링되지 않은 이미지 필터링
        unlabeled = [img for img in image_files if img.name not in self.labels]
        
        if not unlabeled:
            print("✅ 모든 이미지가 라벨링되었습니다!")
            return
        
        print(f"\n📊 총 {len(image_files)}개 이미지 중 {len(unlabeled)}개 미라벨링")
        print("=" * 60)
        
        # OCR 제안을 위한 EasyOCR 초기화 (선택적)
        ocr_reader = None
        if use_ocr_suggestion:
            try:
                import easyocr
                print("EasyOCR 초기화 중...")
                ocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)
                print("✅ EasyOCR 준비 완료")
            except Exception as e:
                print(f"⚠️ EasyOCR 초기화 실패: {e}")
                print("OCR 제안 없이 진행합니다.")
        
        # 각 이미지 라벨링
        for i, img_path in enumerate(unlabeled, 1):
            print(f"\n[{i}/{len(unlabeled)}] {img_path.name}")
            print("-" * 60)
            
            # 이미지 표시
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"❌ 이미지를 읽을 수 없습니다: {img_path}")
                continue
            
            # 이미지 크기 조정 (너무 크면)
            height, width = img.shape[:2]
            max_size = 800
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            cv2.imshow('Image - Press any key to continue', img)
            
            # OCR 제안 (선택적)
            suggestion = None
            if ocr_reader:
                try:
                    results = ocr_reader.readtext(img)
                    # 숫자만 추출
                    import re
                    all_text = ' '.join([r[1] for r in results])
                    numbers = re.findall(r'\d+', all_text)
                    # 12~15자리 숫자 찾기
                    valid_numbers = [n for n in numbers if 12 <= len(n) <= 15]
                    if valid_numbers:
                        suggestion = max(valid_numbers, key=len)
                        print(f"💡 OCR 제안: {suggestion}")
                except Exception as e:
                    print(f"⚠️ OCR 제안 실패: {e}")
            
            # 사용자 입력
            if suggestion:
                user_input = input(f"이력번호 입력 (제안: {suggestion}, Enter로 수락, 's'로 건너뛰기): ").strip()
                if user_input.lower() == 's':
                    print("건너뜀")
                    continue
                elif user_input == "":
                    label = suggestion
                else:
                    label = user_input
            else:
                user_input = input("이력번호 입력 (12~15자리 숫자, 's'로 건너뛰기): ").strip()
                if user_input.lower() == 's':
                    print("건너뜀")
                    continue
                label = user_input
            
            # 라벨 저장
            if label:
                self.labels[img_path.name] = label
                print(f"✅ 라벨 저장: {label}")
            
            cv2.destroyAllWindows()
        
        # 최종 저장
        self.save_labels()
        print(f"\n✅ 라벨링 완료! 총 {len(self.labels)}개 라벨 저장됨")
    
    def view_labels(self):
        """저장된 라벨 확인"""
        if not self.labels:
            print("저장된 라벨이 없습니다.")
            return
        
        print(f"\n📋 저장된 라벨 ({len(self.labels)}개):")
        print("=" * 60)
        for filename, label in sorted(self.labels.items()):
            print(f"{filename:30s} | {label}")
        print("=" * 60)


if __name__ == "__main__":
    # 사용 예시
    labeler = DataLabeler("data/raw", "data/labeled/labels.txt")
    
    print("\n라벨링 도구")
    print("=" * 60)
    print("1. 이미지 라벨링 시작")
    print("2. 저장된 라벨 확인")
    
    choice = input("\n선택 (1 또는 2): ").strip()
    
    if choice == "1":
        use_ocr = input("OCR 제안 사용? (y/n, 기본값: y): ").strip().lower()
        use_ocr_suggestion = use_ocr != 'n'
        labeler.label_images(use_ocr_suggestion=use_ocr_suggestion)
    elif choice == "2":
        labeler.view_labels()
    else:
        print("❌ 잘못된 선택입니다.")
