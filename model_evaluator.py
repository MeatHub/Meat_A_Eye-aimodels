"""
모델 평가 도구
OCR 모델의 성능을 평가합니다.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from ocr_engine import OCREngine

class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, image_dir: str, label_file: str):
        """
        초기화
        
        Args:
            image_dir: 이미지 폴더 경로
            label_file: 라벨 파일 경로
        """
        self.image_dir = Path(image_dir)
        self.label_file = Path(label_file)
        
        # 라벨 로드
        self.labels = self.load_labels()
        
        # OCR 엔진 초기화
        print("OCR 엔진 초기화 중...")
        self.ocr = OCREngine(gpu=True)
        print("✅ OCR 엔진 준비 완료")
    
    def load_labels(self) -> Dict[str, str]:
        """라벨 파일 로드"""
        labels = {}
        if self.label_file.exists():
            with open(self.label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '|' in line:
                        filename, label = line.split('|', 1)
                        labels[filename] = label
        return labels
    
    def evaluate_single_image(self, image_path: Path, true_label: str) -> Tuple[bool, str, str]:
        """
        단일 이미지 평가
        
        Returns:
            (정확 여부, 예측값, 정답)
        """
        # OCR 수행
        predicted = self.ocr.extract_trace_number(str(image_path))
        
        # 정확도 계산 (완전 일치)
        is_correct = (predicted == true_label) if predicted else False
        
        return is_correct, predicted or "", true_label
    
    def evaluate_dataset(self, split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)) -> Dict:
        """
        데이터셋 평가
        
        Args:
            split_ratio: (학습, 검증, 테스트) 비율
        
        Returns:
            평가 결과 딕셔너리
        """
        # 이미지 파일 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.image_dir.glob(f"*{ext}"))
        
        # 라벨이 있는 이미지만 필터링
        labeled_images = []
        for img_file in image_files:
            if img_file.name in self.labels:
                labeled_images.append(img_file)
        
        if not labeled_images:
            print("❌ 라벨이 있는 이미지를 찾을 수 없습니다!")
            return {}
        
        print(f"📊 총 {len(labeled_images)}개 이미지 평가 시작")
        print("=" * 60)
        
        # 데이터 분할
        train_ratio, val_ratio, test_ratio = split_ratio
        n_total = len(labeled_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # 랜덤 셔플
        import random
        random.shuffle(labeled_images)
        
        train_images = labeled_images[:n_train]
        val_images = labeled_images[n_train:n_train+n_val]
        test_images = labeled_images[n_train+n_val:]
        
        print(f"학습용: {len(train_images)}개")
        print(f"검증용: {len(val_images)}개")
        print(f"테스트용: {len(test_images)}개")
        print("-" * 60)
        
        # 각 세트 평가
        results = {}
        for split_name, images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            if not images:
                continue
            
            print(f"\n[{split_name.upper()} 세트 평가 중...]")
            
            correct = 0
            total = len(images)
            errors = []
            
            for i, img_path in enumerate(images, 1):
                true_label = self.labels[img_path.name]
                is_correct, predicted, true_val = self.evaluate_single_image(img_path, true_label)
                
                if is_correct:
                    correct += 1
                else:
                    errors.append({
                        'image': img_path.name,
                        'predicted': predicted,
                        'true': true_val
                    })
                
                if i % 10 == 0:
                    print(f"  진행 중... {i}/{total}")
            
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            results[split_name] = {
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'errors': errors
            }
            
            print(f"  ✅ 정확도: {accuracy:.2f}% ({correct}/{total})")
        
        return results
    
    def print_evaluation_report(self, results: Dict):
        """평가 결과 리포트 출력"""
        print("\n" + "=" * 60)
        print("평가 결과 리포트")
        print("=" * 60)
        
        for split_name, result in results.items():
            print(f"\n[{split_name.upper()} 세트]")
            print(f"  총 이미지: {result['total']}개")
            print(f"  정확: {result['correct']}개")
            print(f"  오류: {result['total'] - result['correct']}개")
            print(f"  정확도: {result['accuracy']:.2f}%")
            
            # 오류 예시 출력 (최대 5개)
            if result['errors']:
                print(f"\n  오류 예시 (최대 5개):")
                for error in result['errors'][:5]:
                    print(f"    이미지: {error['image']}")
                    print(f"      예측: {error['predicted']}")
                    print(f"      정답: {error['true']}")
        
        # 전체 평균 정확도
        if results:
            avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
            print(f"\n[전체 평균 정확도]")
            print(f"  {avg_accuracy:.2f}%")
        
        print("=" * 60)


if __name__ == "__main__":
    # 사용 예시
    print("모델 평가 도구")
    print("=" * 60)
    
    image_dir = input("이미지 폴더 경로 (기본값: data/labeled/images): ").strip()
    if not image_dir:
        image_dir = "data/labeled/images"
    
    label_file = input("라벨 파일 경로 (기본값: data/labeled/labels.txt): ").strip()
    if not label_file:
        label_file = "data/labeled/labels.txt"
    
    evaluator = ModelEvaluator(image_dir, label_file)
    
    # 평가 실행
    results = evaluator.evaluate_dataset()
    
    # 리포트 출력
    evaluator.print_evaluation_report(results)
