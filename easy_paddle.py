# -*- coding: utf-8 -*-

# [수정] torch를 모든 라이브러리보다 먼저 임포트하여 DLL 충돌 방지

try:

    import torch

except ImportError:

    pass



import os

import re

import cv2

import sys

import glob

import numpy as np

from pathlib import Path

from paddleocr import PaddleOCR

import easyocr



# [환경 설정]

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"



class MeatEyeHybridEngine:

    def __init__(self):

        print("\n[1/3] 하이브리드 엔진 초기화 중 (Paddle + Easy)...")

        try:

            # CPU 환경 명시적 설정

            self.paddle = PaddleOCR(ocr_version='PP-OCRv4', lang="korean", use_gpu=False, show_log=False)

            self.easy = easyocr.Reader(['ko', 'en'], gpu=False)

           

            self.valid_patterns = [

                re.compile(r'^L\d{14}$'),

                re.compile(r'^\d{12}$'),

                re.compile(r'^A\d{20,25}$')

            ]

            print("[2/3] 엔진 준비 완료!")

        except Exception as e:

            print(f"❌ 초기화 실패: {e}")

            sys.exit(1)



    def character_correction(self, text):

        res = re.sub(r"[^A-Z0-9]", "", text.upper())

        c_map = {'O':'0', 'D':'0', 'Q':'0', 'H':'0', 'G':'6', 'I':'1', 'L':'1', 'T':'7', 'S':'5', 'B':'8', 'N':'9'}

        fixed = ""

        for i, char in enumerate(res):

            if i == 0 and char == 'L': fixed += 'L'

            elif char in c_map: fixed += c_map[char]

            else: fixed += char

        if len(fixed) == 14 and fixed.isdigit(): fixed = "L" + fixed

        if len(fixed) == 12 and fixed.startswith('105'): fixed = '9' + fixed[1:]

        return fixed



    def is_valid(self, text):

        return any(p.match(text) for p in self.valid_patterns)



    def _extract_ground_truth(self, file_path):

        """파일명에서 실제 이력번호 추출 (확장자, 괄호, 공백 제거). A, L 패턴 지원."""

        file_name = os.path.basename(str(file_path))

        name_without_ext = os.path.splitext(file_name)[0]

        cleaned = re.sub(r'\s*\([^)]*\)\s*', '', name_without_ext).replace(' ', '').replace('\t', '')

        cleaned = re.sub(r'[^AL0-9]', '', cleaned.upper())

        if re.match(r'^\d{12}$', cleaned):

            return cleaned

        if re.match(r'^L\d{14}$', cleaned):

            return cleaned

        if re.match(r'^A\d{19,29}$', cleaned):

            return cleaned

        if 12 <= len(cleaned) <= 30:

            return cleaned

        return None



    def process(self, img_path):

        try:

            img_array = np.fromfile(str(img_path), np.uint8)

            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

           

            # Step 1: PaddleOCR

            paddle_res = self.paddle.ocr(img, cls=True)

            if paddle_res and paddle_res[0]:

                for line in paddle_res[0]:

                    cleaned = self.character_correction(line[1][0])

                    if self.is_valid(cleaned): return cleaned

           

            # Step 2: EasyOCR

            easy_res = self.easy.readtext(img)

            if easy_res:

                for (bbox, text, prob) in easy_res:

                    cleaned = self.character_correction(text)

                    if self.is_valid(cleaned): return cleaned

           

            return "미인식"

        except Exception as e:

            return f"Error: {e}"



# --- 테스트 실행부 ---

if __name__ == "__main__":

    TEST_DIR = r"C:\Dahila\Projects\meathub\Meat_A_Eye-aimodels\data\raw2"

    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

    image_files = []

    for ext in valid_extensions:

        image_files.extend(glob.glob(os.path.join(TEST_DIR, ext)))



    if not image_files:

        print(f"경로를 다시 확인해주세요: {TEST_DIR}")

    else:

        print(f"[{len(image_files)}개 파일 테스트 시작 - 하이브리드 엔진 (Paddle + EasyOCR)]")

        print("=" * 100)



        engine = MeatEyeHybridEngine()

        correct_count = 0

        total_count = 0

        error_log = []



        for img_path in image_files:

            file_name = os.path.basename(img_path)

            try:

                if not os.path.exists(img_path):

                    print(f"파일명: {file_name:<35} | 에러: 파일이 존재하지 않습니다")

                    continue

                if os.path.getsize(img_path) == 0:

                    print(f"파일명: {file_name:<35} | 에러: 파일 크기가 0입니다")

                    continue



                result = engine.process(img_path)

                ground_truth = engine._extract_ground_truth(img_path)



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

                    print(f"파일명: {file_name:<35} | 이력번호: {result:<25} | (검증 불가)")



            except KeyboardInterrupt:

                print("\n사용자에 의해 중단되었습니다.")

                break

            except Exception as e:

                print(f"파일명: {file_name:<35} | 에러: {e}")



        print("=" * 100)

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
