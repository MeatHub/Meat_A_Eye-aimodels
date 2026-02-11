"""
데이터 수집 도구
웹캠이나 이미지 폴더에서 데이터를 수집합니다.
"""

import cv2
import os
from datetime import datetime
from pathlib import Path

class DataCollector:
    """데이터 수집 클래스"""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        초기화
        
        Args:
            output_dir: 저장할 폴더 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 1
    
    def collect_from_webcam(self, num_images: int = 10):
        """
        웹캠에서 이미지 수집
        
        Args:
            num_images: 수집할 이미지 개수
        """
        print("=" * 60)
        print("웹캠 데이터 수집 시작")
        print("=" * 60)
        print(f"총 {num_images}장의 이미지를 수집합니다.")
        print("\n사용법:")
        print("  - 스페이스바: 현재 프레임 저장")
        print("  - ESC 또는 'q': 종료")
        print("=" * 60)
        
        # 웹캠 열기
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다!")
            return
        
        collected = 0
        
        while collected < num_images:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다!")
                break
            
            # 프레임 표시
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Collected: {collected}/{num_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to save, ESC to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Data Collector - Press SPACE to save', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # 스페이스바
                # 파일명 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{self.counter:03d}_{timestamp}.jpg"
                filepath = self.output_dir / filename
                
                # 저장
                cv2.imwrite(str(filepath), frame)
                print(f"✅ 저장됨: {filename} ({collected + 1}/{num_images})")
                
                collected += 1
                self.counter += 1
                
            elif key == 27 or key == ord('q'):  # ESC 또는 'q'
                print("\n수집 중단")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ 총 {collected}장의 이미지가 저장되었습니다: {self.output_dir}")
    
    def collect_from_folder(self, source_folder: str):
        """
        폴더에서 이미지 복사
        
        Args:
            source_folder: 원본 이미지가 있는 폴더
        """
        source_path = Path(source_folder)
        if not source_path.exists():
            print(f"❌ 폴더를 찾을 수 없습니다: {source_folder}")
            return
        
        # 이미지 파일 확장자
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # 이미지 파일 찾기
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {source_folder}")
            return
        
        print(f"📁 {len(image_files)}개의 이미지 파일 발견")
        
        # 복사
        copied = 0
        for img_file in image_files:
            filename = f"image_{self.counter:03d}_{img_file.name}"
            dest_path = self.output_dir / filename
            
            # 파일 복사
            import shutil
            shutil.copy2(img_file, dest_path)
            print(f"✅ 복사됨: {filename}")
            
            copied += 1
            self.counter += 1
        
        print(f"\n✅ 총 {copied}장의 이미지가 복사되었습니다: {self.output_dir}")


if __name__ == "__main__":
    # 사용 예시
    collector = DataCollector("data/raw")
    
    print("\n데이터 수집 방법을 선택하세요:")
    print("1. 웹캠에서 수집")
    print("2. 폴더에서 복사")
    
    choice = input("\n선택 (1 또는 2): ").strip()
    
    if choice == "1":
        num = input("수집할 이미지 개수 (기본값: 10): ").strip()
        num_images = int(num) if num.isdigit() else 10
        collector.collect_from_webcam(num_images)
    elif choice == "2":
        folder = input("원본 이미지 폴더 경로: ").strip()
        collector.collect_from_folder(folder)
    else:
        print("❌ 잘못된 선택입니다.")
