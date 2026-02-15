"""
Step 4: PaddleOCR Recognition 파인튜닝
======================================
step3 에서 생성된 OCR/rec_dataset/ 을 사용하여
PP-OCRv4 rec 모델을 파인튜닝합니다.

전제:
  - PaddlePaddle 이 설치된 Python 이 필요합니다 (venv 또는 글로벌)
  - PaddleOCR GitHub 소스가 없으면 자동 clone 합니다
  - 사전학습 모델은 자동 다운로드됩니다

사용법:
  python step4_train_paddle_rec.py                 # 기본 설정으로 학습
  python step4_train_paddle_rec.py --epochs 200    # 에포크 지정
  python step4_train_paddle_rec.py --gpu            # GPU 사용
  python step4_train_paddle_rec.py --config-only    # 설정만 생성

생성물 (모두 OCR/ 폴더 내):
  rec_output/                <- 학습된 모델
  rec_train_config.yml       <- 학습 설정
  pretrained/                <- 사전학습 가중치
  PaddleOCR/                 <- GitHub clone (자동)
"""

import os
import sys
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path


# ── 경로 설정 (모두 OCR/ 폴더 기준) ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # ai-server/OCR/

# 학습 데이터 (step3 결과물) — OCR/ 안
REC_DIR = SCRIPT_DIR / "rec_dataset"

# 출력: 모두 OCR/ 안
OUTPUT_DIR = SCRIPT_DIR / "rec_output"
PRETRAINED_DIR = SCRIPT_DIR / "pretrained"
CONFIG_PATH = SCRIPT_DIR / "rec_train_config.yml"
PADDLEOCR_DIR = SCRIPT_DIR / "PaddleOCR"

# en_PP-OCRv4 rec 사전학습 모델 URL (영어/숫자 — 95자 dict, 0-9 A-Z 포함)
PRETRAINED_MODEL_URL = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar"
)
PRETRAINED_SUBDIR = "en_PP-OCRv4_rec_train"


def find_paddle_python() -> str:
    """PaddlePaddle 이 설치된 Python 경로를 찾습니다."""
    # 1) 현재 Python
    try:
        import paddle  # noqa: F401
        return sys.executable
    except ImportError:
        pass

    # 2) 글로벌 Python 후보
    candidates = [
        r"C:\Users\pak10\AppData\Local\Programs\Python\Python312\python.exe",
        shutil.which("python"),
        shutil.which("python3"),
    ]

    for py in candidates:
        if py and Path(py).exists() and str(py) != sys.executable:
            try:
                r = subprocess.run(
                    [py, "-c", "import paddle; print(paddle.__version__)"],
                    capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0:
                    ver = r.stdout.strip()
                    print(f"  PaddlePaddle 발견: {py} (v{ver})")
                    return py
            except Exception:
                continue

    return ""


def clone_paddleocr():
    """PaddleOCR GitHub 소스를 OCR/PaddleOCR/ 에 clone"""
    if PADDLEOCR_DIR.exists() and (PADDLEOCR_DIR / "tools" / "train.py").exists():
        print(f"  PaddleOCR 소스 이미 존재: {PADDLEOCR_DIR}")
        return True

    print("  PaddleOCR GitHub 소스 clone 중... (--depth 1)")
    cmd = [
        "git", "clone", "--depth", "1",
        "https://github.com/PaddlePaddle/PaddleOCR.git",
        str(PADDLEOCR_DIR),
    ]
    try:
        result = subprocess.run(cmd, timeout=300)
        if result.returncode == 0 and (PADDLEOCR_DIR / "tools" / "train.py").exists():
            print(f"  -> clone 완료: {PADDLEOCR_DIR}")
            return True
        else:
            print("  [ERROR] clone 실패 또는 tools/train.py 없음")
            return False
    except FileNotFoundError:
        print("  [ERROR] git 이 설치되어 있지 않습니다.")
        return False
    except subprocess.TimeoutExpired:
        print("  [ERROR] clone 타임아웃 (네트워크 확인)")
        return False


def patch_paddleocr_imports():
    """
    PaddlePaddle + PyTorch DLL 충돌 우회 패치.

    문제: paddle 로드 후 torch 의 shm.dll 로드 실패 (Windows DLL 충돌)
    경로: PaddleOCR -> ppocr.data.imaug.iaa_augment -> albumentations -> torch
    해결: iaa_augment.py 전체를 패치하여 albumentations 없이도 동작하게 함.
    참고: rec 학습은 IaaAugment 를 사용하지 않으므로 안전함.
    """
    iaa_path = PADDLEOCR_DIR / "ppocr" / "data" / "imaug" / "iaa_augment.py"
    if not iaa_path.exists():
        return

    content = iaa_path.read_text(encoding="utf-8")

    # 이미 패치되었는지 확인
    if "# [PATCHED]" in content:
        print("  iaa_augment.py 이미 패치 적용됨")
        return

    # 전체 파일을 안전한 버전으로 교체
    patched = '''# [PATCHED] paddle+torch DLL 충돌 우회 - albumentations optional
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np

try:
    import albumentations as A
    from albumentations.core.transforms_interface import DualTransform
    from albumentations.augmentations.geometric import functional as fgeometric
    from packaging import version
    ALBU_VERSION = version.parse(A.__version__)
    IS_ALBU_NEW_VERSION = ALBU_VERSION >= version.parse("1.4.15")
    HAS_ALBU = True
except (ImportError, OSError):
    A = None
    DualTransform = object
    fgeometric = None
    IS_ALBU_NEW_VERSION = False
    HAS_ALBU = False


class ImgaugLikeResize(DualTransform):
    def __init__(self, scale_range=(0.5, 3.0), interpolation=1, p=1.0):
        if HAS_ALBU:
            super().__init__(p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    def apply(self, img, scale=1.0, **params):
        import cv2
        h, w = img.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        return cv2.resize(img, (nw, nh), interpolation=self.interpolation)

    def get_params(self):
        return {"scale": np.random.uniform(*self.scale_range)}


class AugmenterBuilder(object):
    def __init__(self):
        self.imgaug_to_albu = {"Fliplr": "HorizontalFlip", "Flipud": "VerticalFlip", "Affine": "Affine"}

    def build(self, args, root=True):
        if not HAS_ALBU:
            return None
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.build(v, root=False) for v in args]
                return A.Compose(sequence, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
            else:
                t = args[0]
                a = args[1] if len(args) > 1 else {}
                a = self.map_arguments(t, a)
                t = self.imgaug_to_albu.get(t, t)
                if t == "Resize":
                    return ImgaugLikeResize(**a)
                return getattr(A, t)(**{k: tuple(v) if isinstance(v, list) else v for k, v in a.items()})
        elif isinstance(args, dict):
            t = args["type"]
            a = self.map_arguments(t, args.get("args", {}))
            t = self.imgaug_to_albu.get(t, t)
            if t == "Resize":
                return ImgaugLikeResize(**a)
            return getattr(A, t)(**{k: tuple(v) if isinstance(v, list) else v for k, v in a.items()})
        raise RuntimeError("Unknown augmenter arg: " + str(args))

    def map_arguments(self, t, a):
        a = a.copy()
        if t == "Resize":
            s = a.get("size", [1.0, 1.0])
            return {"scale_range": tuple(s), "interpolation": 1, "p": 1.0}
        elif t == "Affine":
            r = a.get("rotate", 0)
            a["rotate"] = tuple(r) if isinstance(r, list) else (float(r), float(r))
            a["p"] = 1.0
            return a
        a["p"] = a.get("p", 1.0)
        return a


class IaaAugment:
    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            augmenter_args = [
                {"type": "Fliplr", "args": {"p": 0.5}},
                {"type": "Affine", "args": {"rotate": [-10, 10]}},
                {"type": "Resize", "args": {"size": [0.5, 3]}},
            ]
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        image = data["image"]
        if self.augmenter:
            keypoints = []
            lengths = []
            for poly in data["polys"]:
                keypoints.extend([tuple(p) for p in poly])
                lengths.append(len(poly))
            transformed = self.augmenter(image=image, keypoints=keypoints)
            data["image"] = transformed["image"]
            new_polys, idx = [], 0
            for l in lengths:
                new_polys.append(np.array([kp[:2] for kp in transformed["keypoints"][idx:idx+l]]))
                idx += l
            data["polys"] = np.array(new_polys)
        return data
'''
    iaa_path.write_text(patched, encoding="utf-8")
    print("  iaa_augment.py 패치 적용 완료 (albumentations/torch 충돌 우회)")

    # __init__.py 에서 latex_ocr_aug, unimernet_aug 임포트도 try/except 로 감싸기
    init_path = PADDLEOCR_DIR / "ppocr" / "data" / "imaug" / "__init__.py"
    if init_path.exists():
        init_content = init_path.read_text(encoding="utf-8")
        if "# [PATCHED-INIT]" not in init_content:
            init_content = init_content.replace(
                "from .latex_ocr_aug import *\nfrom .unimernet_aug import *",
                "# [PATCHED-INIT] paddle+torch DLL 충돌 우회\n"
                "try:\n    from .latex_ocr_aug import *\nexcept (ImportError, OSError):\n    pass\n"
                "try:\n    from .unimernet_aug import *\nexcept (ImportError, OSError):\n    pass"
            )
            init_path.write_text(init_content, encoding="utf-8")
            print("  __init__.py 패치 적용 완료 (latex/unimernet aug 충돌 우회)")

    # program.py 에서 ParallelEnv().dev_id AttributeError 수정 (PaddlePaddle 3.x 호환)
    prog_path = PADDLEOCR_DIR / "tools" / "program.py"
    if prog_path.exists():
        prog_content = prog_path.read_text(encoding="utf-8")
        old_dev = 'device = "gpu:{}".format(dist.ParallelEnv().dev_id) if use_gpu else "cpu"'
        if old_dev in prog_content and "# [PATCHED-PROG]" not in prog_content:
            new_dev = (
                "# [PATCHED-PROG] PaddlePaddle 3.x 호환\n"
                "        if use_gpu:\n"
                "            try:\n"
                '                device = "gpu:{}".format(dist.ParallelEnv().dev_id)\n'
                "            except AttributeError:\n"
                '                device = "gpu:0"\n'
                "        else:\n"
                '            device = "cpu"'
            )
            prog_content = prog_content.replace(old_dev, new_dev)
            prog_path.write_text(prog_content, encoding="utf-8")
            print("  program.py 패치 적용 완료 (ParallelEnv 호환성)")


def download_pretrained():
    """en_PP-OCRv4 rec 사전학습 모델 다운로드 및 압축 해제"""
    tar_path = PRETRAINED_DIR / f"{PRETRAINED_SUBDIR}.tar"
    model_dir = PRETRAINED_DIR / PRETRAINED_SUBDIR

    if model_dir.exists() and any(model_dir.glob("*.pdparams")):
        print(f"  사전학습 모델 이미 존재: {model_dir}")
        return model_dir

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        print(f"  사전학습 모델 다운로드 중...")
        import urllib.request
        urllib.request.urlretrieve(PRETRAINED_MODEL_URL, str(tar_path))
        print(f"  -> 다운로드 완료")

    print("  압축 해제 중...")
    import tarfile
    with tarfile.open(str(tar_path), "r") as tar:
        tar.extractall(str(PRETRAINED_DIR))
    print(f"  -> 완료: {model_dir}")

    return model_dir


def generate_config(pretrained_dir: Path, epochs: int, use_gpu: bool) -> Path:
    """PaddleOCR rec 학습 설정 YAML 생성 — en_PP-OCRv4 아키텍처 기반"""

    # 영어 dict 사용 (95 chars: 0-9, A-Z, a-z, 특수문자)
    # → 사전학습 모델의 Head 차원과 완전 일치하여 전체 가중치 로드 가능
    en_dict_path = PADDLEOCR_DIR / "ppocr" / "utils" / "en_dict.txt"
    if not en_dict_path.exists():
        print(f"[ERROR] {en_dict_path} 없음. PaddleOCR clone 을 먼저 실행하세요.")
        sys.exit(1)

    # pretrained 모델의 .pdparams 파일명 자동 탐지
    pretrained_model = None
    if pretrained_dir.exists():
        for candidate in ["best_accuracy", "student", "best_model"]:
            if (pretrained_dir / f"{candidate}.pdparams").exists():
                pretrained_model = str(pretrained_dir / candidate)
                break
        # 위 후보에 없으면 아무 .pdparams 찾기
        if pretrained_model is None:
            for p in pretrained_dir.glob("*.pdparams"):
                pretrained_model = str(pretrained_dir / p.stem)
                break
    if pretrained_model:
        print(f"  사전학습 가중치: {pretrained_model}")
    else:
        print("  [WARN] 사전학습 가중치 없음 — 처음부터 학습")

    config = {
        "Global": {
            "debug": False,
            "use_gpu": use_gpu,
            "epoch_num": epochs,
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "save_model_dir": str(OUTPUT_DIR),
            "save_epoch_step": 5,
            "eval_batch_step": [0, 50],
            "cal_metric_during_train": True,
            "pretrained_model": pretrained_model,
            "checkpoints": None,
            "save_inference_dir": str(OUTPUT_DIR / "inference"),
            "use_visualdl": False,
            "infer_img": None,
            "character_dict_path": str(en_dict_path),
            "max_text_length": 25,
            "infer_mode": False,
            "use_space_char": True,
            "distributed": False,
        },
        "Optimizer": {
            "name": "Adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": {
                "name": "Cosine",
                "learning_rate": 0.0005,
                "warmup_epoch": 2,
            },
            "regularizer": {
                "name": "L2",
                "factor": 3.0e-05,
            },
        },
        # en_PP-OCRv4 아키텍처 (PPLCNetV3 + NRTR)
        "Architecture": {
            "model_type": "rec",
            "algorithm": "SVTR_LCNet",
            "Transform": None,
            "Backbone": {
                "name": "PPLCNetV3",
                "scale": 0.95,
            },
            "Head": {
                "name": "MultiHead",
                "head_list": [
                    {
                        "CTCHead": {
                            "Neck": {
                                "name": "svtr",
                                "dims": 120,
                                "depth": 2,
                                "hidden_dims": 120,
                                "kernel_size": [1, 3],
                                "use_guide": True,
                            },
                            "Head": {
                                "fc_decay": 1.0e-05,
                            },
                        }
                    },
                    {
                        "NRTRHead": {
                            "nrtr_dim": 384,
                            "max_text_length": 25,
                        }
                    },
                ],
            },
        },
        "Loss": {
            "name": "MultiLoss",
            "loss_config_list": [
                {"CTCLoss": None},
                {"NRTRLoss": None},
            ],
        },
        "PostProcess": {
            "name": "CTCLabelDecode",
        },
        "Metric": {
            "name": "RecMetric",
            "main_indicator": "acc",
            "ignore_space": False,
        },
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": str(REC_DIR),
                "ext_op_transform_idx": 1,
                "label_file_list": [str(REC_DIR / "rec_train.txt")],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"RecConAug": {
                        "prob": 0.5,
                        "ext_data_num": 2,
                        "image_shape": [48, 320, 3],
                        "max_text_length": 25,
                    }},
                    {"RecAug": None},
                    {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                    {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                    {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}},
                ],
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": 32,
                "drop_last": True,
                "num_workers": 4,
            },
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": str(REC_DIR),
                "label_file_list": [str(REC_DIR / "rec_val.txt")],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                    {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                    {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}},
                ],
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": 128,
                "num_workers": 4,
            },
        },
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  설정 파일 생성: {CONFIG_PATH}")
    return CONFIG_PATH


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR rec 파인튜닝")
    parser.add_argument("--epochs", type=int, default=20, help="학습 에포크 수")
    parser.add_argument("--no-gpu", action="store_true", help="GPU 사용 안 함 (기본: GPU 사용)")
    parser.add_argument("--config-only", action="store_true", help="설정 파일만 생성")
    args = parser.parse_args()
    use_gpu = not args.no_gpu

    print("=" * 70)
    print("Step 4: PaddleOCR Recognition 파인튜닝")
    print(f"  작업 경로: {SCRIPT_DIR}")
    print("=" * 70)

    # 데이터 확인
    train_label = REC_DIR / "rec_train.txt"
    if not train_label.exists():
        print("[ERROR] rec_dataset/rec_train.txt 가 없습니다. step3 을 먼저 실행하세요.")
        sys.exit(1)

    with open(train_label, "r", encoding="utf-8") as f:
        train_cnt = len([l for l in f.read().strip().split("\n") if l.strip()])
    print(f"  학습 데이터: {train_cnt}장")
    print(f"  GPU 사용: {use_gpu}")

    # 0) PaddlePaddle Python 찾기
    print("\n[0/4] PaddlePaddle 환경 확인")
    paddle_python = find_paddle_python()
    if not paddle_python:
        print("[ERROR] PaddlePaddle 이 설치된 Python 을 찾을 수 없습니다.")
        print("  설치: pip install paddlepaddle")
        sys.exit(1)
    print(f"  사용할 Python: {paddle_python}")

    # 1) PaddleOCR 소스 clone
    print("\n[1/4] PaddleOCR 소스 준비")
    if not clone_paddleocr():
        print("[ERROR] PaddleOCR 소스를 가져올 수 없습니다.")
        sys.exit(1)

    # 2) 사전학습 모델 다운로드
    print("\n[2/4] 사전학습 모델 준비")
    pretrained_dir = download_pretrained()

    # 3) 학습 설정 생성
    print("\n[3/5] 학습 설정 생성")
    config_path = generate_config(pretrained_dir, args.epochs, use_gpu)

    # 4) paddle+torch DLL 충돌 패치
    print("\n[4/5] PaddleOCR 소스 패치")
    patch_paddleocr_imports()

    if args.config_only:
        print("\n  --config-only: 설정 파일만 생성됨")
        print(f"  수동 실행:")
        print(f"    cd {PADDLEOCR_DIR}")
        print(f"    {paddle_python} tools/train.py -c \"{config_path}\"")
        return

    # 5) 학습 실행
    print(f"\n[5/5] 학습 시작 (epochs={args.epochs}, gpu={use_gpu})")
    print("-" * 70)

    train_script = PADDLEOCR_DIR / "tools" / "train.py"
    cmd = [paddle_python, str(train_script), "-c", str(config_path)]

    print(f"  명령: {' '.join(cmd)}")
    print()

    # PYTHONPATH 에 PaddleOCR 소스 추가 (cwd 변경 대신)
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PADDLEOCR_DIR) + (";" + pp if pp else "")

    try:
        result = subprocess.run(cmd, cwd=str(PADDLEOCR_DIR), env=env)
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("[완료] 학습 성공!")
            print(f"  모델: {OUTPUT_DIR}")
            print(f"  inference: {OUTPUT_DIR / 'inference'}")
            print("=" * 70)
        else:
            print(f"\n[오류] 학습 실패 (exit code: {result.returncode})")
            print(f"\n  수동 실행:")
            print(f"    cd {PADDLEOCR_DIR}")
            print(f"    {paddle_python} tools/train.py -c \"{config_path}\"")
    except FileNotFoundError:
        print(f"\n[ERROR] 실행 실패: {paddle_python}")


if __name__ == "__main__":
    main()
