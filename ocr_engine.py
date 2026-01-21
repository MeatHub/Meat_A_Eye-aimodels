"""
Meat A Eye - OCR 엔진 모듈
축산물 이력번호 추출을 위한 OCR 엔진 및 이미지 전처리
"""

import re
import cv2
import numpy as np
from PIL import Image, ImageOps
import easyocr
from typing import Optional, Tuple, List, Dict
import logging
import itertools

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine:
    """OCR 엔진 클래스 - EasyOCR 기반"""
    
    def __init__(self, gpu: bool = True, languages: List[str] = ['ko', 'en']):
        """
        OCR 엔진 초기화
        
        Args:
            gpu: GPU 사용 여부 (기본값: True)
            languages: 인식할 언어 리스트 (기본값: ['ko', 'en'])
        """
        logger.info(f"OCR 엔진 초기화 중... (GPU: {gpu})")
        try:
            self.reader = easyocr.Reader(languages, gpu=gpu)
            logger.info("✅ OCR 엔진 초기화 완료")
        except Exception as e:
            logger.error(f"❌ OCR 엔진 초기화 실패: {e}")
            raise
    
    def auto_orient_image(self, image: np.ndarray) -> np.ndarray:
        """
        2단계: EXIF 정보를 바탕으로 한 자동 회전
        
        Args:
            image: OpenCV 이미지 (numpy array)
            
        Returns:
            회전된 이미지
        """
        try:
            # OpenCV 이미지를 PIL 이미지로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # EXIF 정보에 따라 자동 회전
            pil_image = ImageOps.exif_transpose(pil_image)
            
            # 다시 OpenCV 형식으로 변환
            oriented_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            logger.info("✅ 이미지 자동 회전 완료")
            return oriented_image
        except Exception as e:
            logger.warning(f"⚠️ EXIF 회전 실패, 원본 반환: {e}")
            return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        3단계: 웹 노이즈 제거 및 전처리
        - Bilateral Filter (노이즈 제거)
        - Denoising
        - Contrast Stretching
        
        Args:
            image: OpenCV 이미지
            
        Returns:
            전처리된 이미지
        """
        # 1. Bilateral Filter (엣지 보존 노이즈 제거)
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. 추가 Denoising (필요시)
        denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)
        
        # 3. 그레이스케일 변환
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # 4. Contrast Stretching (명암 대비 향상)
        # 히스토그램 스트레칭
        min_val = np.min(gray)
        max_val = np.max(gray)
        if max_val > min_val:
            stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            stretched = gray
        
        # 5. 추가 선명화 (선택적)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(stretched, -1, kernel)
        
        logger.info("✅ 이미지 전처리 완료")
        return sharpened

    # ------------------------------
    # 신규: 노란/흰 박스 ROI 탐지 및 ROI 전처리
    # ------------------------------
    def _find_candidate_boxes(self, image: np.ndarray) -> List[np.ndarray]:
        """
        노란/흰 배경 네모 박스를 찾아 ROI 리스트 반환
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 노란색, 흰색 마스크
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([40, 255, 255])
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])

        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_w = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.bitwise_or(mask_y, mask_w)

        # 모폴로지 닫기 연산으로 빈틈 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # 컨투어 탐색
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois: List[np.ndarray] = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 더 느슨한 필터: 작은 박스도 허용
            if w < 50 or h < 20:
                continue
            ratio = w / max(h, 1)
            if ratio < 1.2 or ratio > 10:
                continue
            roi = image[y : y + h, x : x + w]
            rois.append(roi)

        logger.info(f"ROI 후보 탐지: {len(rois)}개")
        return rois

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        ROI에 동일한 전처리 적용 (회전 제외)
        """
        # Bilateral + Denoise
        denoised = cv2.bilateralFilter(roi, 9, 75, 75)
        denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

        min_val = np.min(gray)
        max_val = np.max(gray)
        if max_val > min_val:
            stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            stretched = gray

        # 선택적 선명화
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(stretched, -1, kernel)
        return sharpened

    def _extract_numbers_only(self, text: str) -> List[str]:
        """
        텍스트에서 숫자만 추출하고 12자리로 제한
        """
        numbers = re.findall(r'\d+', text)
        return [n for n in numbers if len(n) == 12]

    def _filter_by_keyword(self, full_text: str, numbers: List[str]) -> List[str]:
        """
        '이력번호' 또는 '묶음번호' 주변에 등장하는 12자리 숫자만 우선 필터링
        """
        if not numbers:
            return []
        keywords = ['이력번호', '묶음번호']
        lowered = full_text
        selected: List[str] = []
        window = 40  # 키워드 앞뒤로 볼 문자 길이

        for kw in keywords:
            idx = lowered.find(kw)
            if idx != -1:
                start = max(0, idx - window)
                end = min(len(lowered), idx + len(kw) + window)
                context = lowered[start:end]
                selected.extend([n for n in numbers if n in context])

        # 중복 제거, 순서 유지
        seen: Dict[str, None] = {}
        for n in selected:
            seen.setdefault(n, None)
        return list(seen.keys())
    
    def extract_numbers(self, text: str) -> List[str]:
        """
        4단계: 정규표현식으로 숫자 추출 및 검증
        
        Args:
            text: OCR로 추출된 텍스트
            
        Returns:
            검증된 숫자 패턴 리스트 (12~15자리)
        """
        # 숫자만 추출 (공백, 하이픈 등 제거)
        numbers = re.findall(r'\d+', text)
        
        # 12~15자리 숫자 패턴 검증
        valid_numbers = []
        for num in numbers:
            if 12 <= len(num) <= 15:
                valid_numbers.append(num)
        
        logger.info(f"추출된 숫자: {valid_numbers}")
        return valid_numbers
    
    def extract_trace_number(self, image_path: str) -> Optional[str]:
        """
        5단계: 통합 함수 - 이력번호 추출
        
        전체 파이프라인:
        1. 이미지 로드
        2. EXIF 자동 회전
        3. ROI 탐색(노란/흰 박스) 후 ROI 단위 전처리 및 OCR
        4. 폴백: 전체 이미지 전처리 후 OCR
        5. 숫자 추출 및 검증 (12자리, 키워드 주변 우선)
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            추출된 이력번호 (12~15자리) 또는 None
        """
        try:
            # 1. 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
                return None
            
            logger.info(f"이미지 로드 완료: {image_path}")
            
            # 2. EXIF 자동 회전
            image = self.auto_orient_image(image)
            
            # 3. ROI 기반 추출 시도
            candidates: List[str] = []
            rois = self._find_candidate_boxes(image)
            if rois:
                logger.info(f"ROI 후보 {len(rois)}개 발견, ROI 기반 추출 시도")
                for roi in rois:
                    processed_roi = self._preprocess_roi(roi)
                    try:
                        roi_results = self.reader.readtext(processed_roi)
                        roi_text = ' '.join([r[1] for r in roi_results])
                        candidates.extend(self._extract_numbers_only(roi_text))
                    except Exception as e:
                        logger.warning(f"ROI OCR 실패: {e}")
            else:
                logger.info("ROI 후보 없음, 전체 이미지 폴백")

            # 4. 폴백: 전체 이미지 전처리 후 OCR
            processed_image = self.preprocess_image(image)
            results = self.reader.readtext(processed_image)
            all_text = ' '.join([result[1] for result in results])
            logger.info(f"OCR 결과: {all_text}")
            candidates.extend(self._extract_numbers_only(all_text))

            # 5. 키워드 근접 필터 (이력번호/묶음번호 주변)
            keyword_filtered = self._filter_by_keyword(all_text, candidates)
            logger.info(f"후보 숫자: {candidates}, 키워드 필터 적용 후: {keyword_filtered}")

            final_candidates = keyword_filtered or candidates
            if final_candidates:
                trace_number = final_candidates[0]
                logger.info(f"✅ 이력번호 추출 성공: {trace_number}")
                return trace_number

            logger.warning("⚠️ 유효한 이력번호를 찾을 수 없습니다")
            return None
                
        except Exception as e:
            logger.error(f"❌ 이력번호 추출 실패: {e}")
            return None


# 사용 예시
if __name__ == "__main__":
    # OCR 엔진 초기화
    ocr = OCREngine(gpu=True)
    
    # 테스트 이미지 경로 (실제 이미지 경로로 변경 필요)
    test_image_path = "test_image.jpg"
    
    # 이력번호 추출
    trace_number = ocr.extract_trace_number(test_image_path)
    
    if trace_number:
        print(f"추출된 이력번호: {trace_number}")
    else:
        print("이력번호를 추출할 수 없습니다.")
