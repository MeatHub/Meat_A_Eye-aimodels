import cv2
import numpy as np
import re
import os
import glob
import tempfile
import easyocr
from typing import Any, List, Tuple, Dict, Optional

# 싱글톤 OCR 인스턴스 (extract_text용)
_ocr_instance: Optional["MeatTraceabilityOCR"] = None


def extract_text(img_array: np.ndarray) -> Dict[str, Any]:
    """
    numpy 이미지 배열에서 이력번호 추출 (main.py /ai/analyze ocr 모드용).
    
    Args:
        img_array: RGB numpy 배열 (H, W, 3)
    
    Returns:
        {"success": bool, "text": str, "raw": str}
    """
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = MeatTraceabilityOCR()
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            path = f.name
        try:
            result = _ocr_instance.extract(path)
            return {
                "success": result != "Not Found",
                "text": result if result != "Not Found" else "",
                "raw": result,
            }
        finally:
            if os.path.exists(path):
                os.unlink(path)
    except Exception as e:
        return {"success": False, "text": "", "raw": str(e), "error": str(e)}


class MeatTraceabilityOCR:
    def __init__(self):
        # 한글(ko)과 영어(en) 모델 로드 (최초 실행 시 자동 다운로드)
        # GPU가 있다면 gpu=True, 없다면 gpu=False
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False)
        
    def _preprocess_image_bilateral(self, img: np.ndarray) -> np.ndarray:
        """
        전처리 파이프라인 1: Bilateral Filter 적용
        - 글자 경계는 유지하되 배경 노이즈만 제거
        """
        # 1. 이미지 2배 확대 (CUBIC interpolation)
        height, width = img.shape[:2]
        enlarged = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        # 2. 그레이스케일 변환
        if len(enlarged.shape) == 3:
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        else:
            gray = enlarged
        
        # 3. Bilateral Filter 적용 (경계 유지하면서 노이즈 제거)
        # d: 필터링에 사용되는 픽셀 이웃의 직경
        # sigmaColor: 색 공간의 표준 편차
        # sigmaSpace: 좌표 공간의 표준 편차
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        return filtered

    def _preprocess_image_adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """
        전처리 파이프라인 2: Adaptive Thresholding 적용
        - 밝기가 불균일한 이미지에 대응
        """
        # 1. 이미지 2배 확대
        height, width = img.shape[:2]
        enlarged = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        # 2. 그레이스케일 변환
        if len(enlarged.shape) == 3:
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        else:
            gray = enlarged
        
        # 3. Adaptive Thresholding 적용
        # 블록 크기: 11, C: 2 (평균에서 빼는 상수)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return adaptive

    def _validate_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """이미지 파일이 유효한지 검증하고 numpy 배열로 반환"""
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(image_path):
                return None, "File Not Found"
            
            # 파일 크기 확인
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                return None, "Empty File"
            
            # 이미지 디코딩 시도 (한글 경로 대응)
            img_array = np.fromfile(image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # 이미지가 제대로 로드되었는지 확인
            if img is None:
                return None, "Invalid Image Format"
            
            # 이미지 크기 확인
            if img.size == 0:
                return None, "Empty Image"
            
            if len(img.shape) < 2 or img.shape[0] == 0 or img.shape[1] == 0:
                return None, "Invalid Image Dimensions"
            
            return img, None
            
        except Exception as e:
            return None, f"Image Load Error: {str(e)}"

    def _get_box_height(self, bbox: List[List[float]]) -> float:
        """박스의 높이 반환"""
        y_coords = [point[1] for point in bbox]
        return max(y_coords) - min(y_coords)

    def _get_box_center_y(self, bbox: List[List[float]]) -> float:
        """박스의 수직 중심점 y좌표 반환"""
        y_coords = [point[1] for point in bbox]
        return np.mean(y_coords)

    def _get_box_x_range(self, bbox: List[List[float]]) -> Tuple[float, float]:
        """박스의 x좌표 범위 (min_x, max_x) 반환"""
        x_coords = [point[0] for point in bbox]
        return min(x_coords), max(x_coords)

    def _boxes_horizontally_aligned(self, box1: List[List[float]], box2: List[List[float]]) -> bool:
        """
        두 박스가 수평 방향으로 인접한지 판단 (강화된 로직)
        - 박스 높이의 50% 이내면 병합 대상
        """
        # 박스 높이 계산
        height1 = self._get_box_height(box1)
        height2 = self._get_box_height(box2)
        avg_height = (height1 + height2) / 2
        
        # y좌표 차이 계산
        y1 = self._get_box_center_y(box1)
        y2 = self._get_box_center_y(box2)
        y_diff = abs(y1 - y2)
        
        # 박스 높이의 50% 이내인지 확인
        if y_diff > avg_height * 0.5:
            return False
        
        # x좌표 간격 확인 (박스 높이의 2배 이내)
        _, max_x1 = self._get_box_x_range(box1)
        min_x2, _ = self._get_box_x_range(box2)
        x_gap = min_x2 - max_x1
        
        if x_gap > avg_height * 2:
            return False
        
        return True

    def _merge_horizontal_boxes(self, results: List[Tuple]) -> List[Tuple]:
        """
        수평 방향으로 인접한 박스들을 강제 병합
        긴 번호(특히 A나 L로 시작하는 것)가 중간에 잘리는 현상 방지
        """
        if not results:
            return []
        
        # 박스를 x좌표 기준으로 정렬
        sorted_results = sorted(results, key=lambda x: self._get_box_x_range(x[0])[0])
        
        merged = []
        i = 0
        
        while i < len(sorted_results):
            current_bbox, current_text, current_prob = sorted_results[i]
            merged_text = current_text
            merged_prob = current_prob
            merged_bbox = current_bbox.copy()
            merged_count = 1
            
            # 현재 박스와 병합 가능한 다음 박스들을 찾아서 병합
            j = i + 1
            while j < len(sorted_results):
                next_bbox, next_text, next_prob = sorted_results[j]
                
                if self._boxes_horizontally_aligned(merged_bbox, next_bbox):
                    # 병합: 텍스트 연결 (공백 제거)
                    merged_text += next_text.replace(' ', '')
                    merged_prob = (merged_prob * merged_count + next_prob) / (merged_count + 1)
                    merged_count += 1
                    
                    # 박스 확장 (최소/최대 좌표로)
                    min_x1, max_x1 = self._get_box_x_range(merged_bbox)
                    min_x2, max_x2 = self._get_box_x_range(next_bbox)
                    min_y1 = min([p[1] for p in merged_bbox])
                    max_y1 = max([p[1] for p in merged_bbox])
                    min_y2 = min([p[1] for p in next_bbox])
                    max_y2 = max([p[1] for p in next_bbox])
                    
                    merged_bbox = [
                        [min(min_x1, min_x2), min(min_y1, min_y2)],
                        [max(max_x1, max_x2), min(min_y1, min_y2)],
                        [max(max_x1, max_x2), max(max_y1, max_y2)],
                        [min(min_x1, min_x2), max(max_y1, max_y2)]
                    ]
                    j += 1
                else:
                    break
            
            merged.append((merged_bbox, merged_text, merged_prob))
            i = j
        
        return merged

    def _clean_text(self, text: str) -> str:
        """텍스트에서 공백 제거 후 A, L, 숫자만 추출"""
        # 공백 제거
        cleaned = text.replace(' ', '').replace('\t', '').replace('\n', '')
        # A, L, 숫자만 추출
        cleaned = re.sub(r'[^AL0-9]', '', cleaned.upper())
        return cleaned

    def _get_distance(self, box1: List[List[float]], box2: List[List[float]]) -> float:
        """두 텍스트 박스 중심점 사이의 유클리드 거리 계산"""
        p1 = np.mean(box1, axis=0)
        p2 = np.mean(box2, axis=0)
        return np.linalg.norm(p1 - p2)

    def extract(self, image_path: str) -> str:
        """
        축산물 이력번호 추출 메인 로직 (앙상블 방식)
        """
        # 1. 이미지 파일 유효성 검사 및 로드
        img, error_msg = self._validate_image(image_path)
        if img is None:
            return error_msg
        
        # 2. 앙상블: 두 가지 전처리 방식으로 OCR 수행
        all_results = []
        
        # 전처리 방식 1: Bilateral Filter
        try:
            preprocessed1 = self._preprocess_image_bilateral(img)
            results1 = self.reader.readtext(preprocessed1, detail=1)
            all_results.extend(results1)
        except Exception as e:
            print(f"Bilateral Filter OCR 실패: {e}")
        
        # 전처리 방식 2: Adaptive Thresholding
        try:
            preprocessed2 = self._preprocess_image_adaptive_threshold(img)
            results2 = self.reader.readtext(preprocessed2, detail=1)
            all_results.extend(results2)
        except Exception as e:
            print(f"Adaptive Threshold OCR 실패: {e}")
        
        if not all_results:
            return "Not Found"
        
        # 3. 수평 방향 박스 병합 (강화된 로직)
        merged_results = self._merge_horizontal_boxes(all_results)
        
        anchors = []
        candidates = []
        
        # 이력번호 관련 키워드 정의
        keyword_patterns = ["이력", "번호", "묶음", "축산물"]

        # 4. 텍스트 분류 및 분석
        for (bbox, text, prob) in merged_results:
            # 텍스트 정제 (공백 제거 후 A, L, 숫자만 추출)
            clean_text = self._clean_text(text)
            
            # 키워드(앵커) 위치 저장
            if any(k in text for k in keyword_patterns):
                anchors.append(bbox)
            
            # 이력번호 패턴 매칭: [AL0-9]{12,30}
            # A로 시작하는 긴 번호, L로 시작하는 15자리, 숫자만 12자리 모두 허용
            if re.match(r'^[AL0-9]{12,30}$', clean_text):
                # 유효한 패턴인지 추가 검증
                # 1. 숫자만 12자리
                # 2. L + 숫자 14자리 (총 15자)
                # 3. A + 숫자 19~29자리 (총 20~30자)
                is_valid = False
                
                if re.match(r'^\d{12}$', clean_text):
                    is_valid = True  # 국내산/수입산 기본 12자리
                elif re.match(r'^L\d{14}$', clean_text):
                    is_valid = True  # 국내산 묶음번호 L + 14자리
                elif re.match(r'^A\d{19,29}$', clean_text):
                    is_valid = True  # 수입산/Batch 번호 A + 19~29자리
                elif len(clean_text) >= 12 and len(clean_text) <= 30:
                    # 길이는 맞지만 패턴이 정확하지 않은 경우도 후보로 포함
                    # (오인식 가능성 고려)
                    is_valid = True
                
                if is_valid:
                    candidates.append({
                        'text': clean_text,
                        'box': bbox,
                        'conf': prob,
                        'length': len(clean_text)
                    })

        if not candidates:
            return "Not Found"

        # 5. 최적의 번호 선택
        # 우선순위: 정확한 패턴 > 길이 > 신뢰도
        
        # 패턴별 분류
        perfect_12 = [c for c in candidates if re.match(r'^\d{12}$', c['text'])]
        perfect_L15 = [c for c in candidates if re.match(r'^L\d{14}$', c['text'])]
        perfect_A20_30 = [c for c in candidates if re.match(r'^A\d{19,29}$', c['text'])]
        
        # 정확한 패턴이 있으면 그것만 사용
        if perfect_12:
            candidates = perfect_12
        elif perfect_L15:
            candidates = perfect_L15
        elif perfect_A20_30:
            candidates = perfect_A20_30
        
        final_candidate = None
        if anchors:
            # 앵커 기반 거리 가중치 부여
            scored_candidates = []
            for cand in candidates:
                min_dist = min([self._get_distance(cand['box'], a) for a in anchors])
                # 점수 = 신뢰도 * 패턴 가중치 / log(거리 + 2)
                pattern_weight = 1.5 if re.match(r'^\d{12}$|^L\d{14}$|^A\d{19,29}$', cand['text']) else 1.0
                score = (cand['conf'] * pattern_weight) / (np.log(min_dist + 2))
                scored_candidates.append((cand['text'], score, cand['conf']))
            
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            final_candidate = scored_candidates[0][0]
        else:
            # 앵커를 못 찾았을 경우 패턴 정확도와 신뢰도로 선택
            candidates.sort(
                key=lambda x: (
                    re.match(r'^\d{12}$|^L\d{14}$|^A\d{19,29}$', x['text']) is not None,
                    x['conf']
                ),
                reverse=True
            )
            final_candidate = candidates[0]['text']
        
        return final_candidate

    def _extract_ground_truth(self, file_path: str) -> Optional[str]:
        """
        파일명에서 실제 이력번호 추출 (확장자, 괄호, 공백 제거)
        A, L 패턴 모두 지원
        """
        file_name = os.path.basename(file_path)
        # 확장자 제거
        name_without_ext = os.path.splitext(file_name)[0]
        # 괄호와 공백 제거 (예: "L02601255978160 (2)" -> "L02601255978160")
        cleaned = re.sub(r'\s*\([^)]*\)\s*', '', name_without_ext)
        # 공백 제거
        cleaned = cleaned.replace(' ', '').replace('\t', '')
        # A, L, 숫자만 추출
        cleaned = self._clean_text(cleaned)
        
        # 유효한 패턴인지 확인
        if re.match(r'^\d{12}$', cleaned):
            return cleaned  # 국내산/수입산 기본 12자리
        elif re.match(r'^L\d{14}$', cleaned):
            return cleaned  # 국내산 묶음번호
        elif re.match(r'^A\d{19,29}$', cleaned):
            return cleaned  # 수입산/Batch 번호
        elif len(cleaned) >= 12 and len(cleaned) <= 30:
            # 길이는 맞지만 패턴이 정확하지 않은 경우도 반환
            return cleaned
        
        return None

# --- 테스트 실행부 ---
if __name__ == "__main__":
    TEST_DIR = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\data\OCR_test"
    
    # 지원하는 이미지 확장자
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(TEST_DIR, ext)))

    if not image_files:
        print(f"경로를 다시 확인해주세요: {TEST_DIR}")
    else:
        print(f"[{len(image_files)}개 파일 테스트 시작 - 고도화된 EasyOCR 엔진 (앙상블 + 강화 병합)]")
        print("=" * 100)
        
        extractor = MeatTraceabilityOCR()
        
        correct_count = 0
        total_count = 0
        error_log = []
        
        for img_path in image_files:
            file_name = os.path.basename(img_path)
            try:
                # 파일 유효성 사전 체크
                if not os.path.exists(img_path):
                    print(f"파일명: {file_name:<35} | 에러: 파일이 존재하지 않습니다")
                    continue
                
                if os.path.getsize(img_path) == 0:
                    print(f"파일명: {file_name:<35} | 에러: 파일 크기가 0입니다")
                    continue
                
                # 이력번호 추출
                result = extractor.extract(img_path)
                
                # 검증 모드: 파일명과 결과 비교 (A, L 패턴 모두 지원)
                ground_truth = extractor._extract_ground_truth(img_path)
                
                if ground_truth:
                    total_count += 1
                    is_correct = (result == ground_truth)
                    if is_correct:
                        correct_count += 1
                        status = "✓ 정확"
                    else:
                        status = "✗ 오류"
                        error_log.append({
                            'file': file_name,
                            'ground_truth': ground_truth,
                            'predicted': result
                        })
                    
                    print(f"파일명: {file_name:<35} | 실제: {ground_truth:<25} | 예측: {result:<25} | {status}")
                else:
                    # 파일명에서 이력번호를 추출할 수 없는 경우
                    print(f"파일명: {file_name:<35} | 이력번호: {result:<25} | (검증 불가)")
                
            except KeyboardInterrupt:
                print("\n사용자에 의해 중단되었습니다.")
                break
            except Exception as e:
                print(f"파일명: {file_name:<35} | 에러: {e}")
        
        print("=" * 100)
        
        # 정확도 계산 및 리포트
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            print(f"\n[검증 결과]")
            print(f"총 검증 가능 파일: {total_count}개")
            print(f"정확히 인식: {correct_count}개")
            print(f"오류: {len(error_log)}개")
            print(f"정확도: {accuracy:.2f}%")
            
            if error_log:
                print(f"\n[오류 상세 로그]")
                for err in error_log:
                    print(f"  파일: {err['file']}")
                    print(f"    실제값: {err['ground_truth']}")
                    print(f"    예측값: {err['predicted']}")
                    print()
        else:
            print("\n검증 가능한 파일이 없습니다 (파일명이 이력번호 형식이 아님)")
        
        print("테스트 완료.")
