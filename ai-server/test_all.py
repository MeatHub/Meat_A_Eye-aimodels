"""
전체 테스트 데이터셋 통합 평가 스크립트
train_dataset_1~N/test 폴더를 모두 합쳐서 한 번에 테스트.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import glob

# ===== 설정 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\b2_imagenet_beef_100-v3.pth"
DATA_ROOT = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\data"
RESULT_DIR = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\test_results_all"

CLASS_NAMES = ['Beef_Brisket', 'Beef_Chuck', 'Beef_Rib', 'Beef_Ribeye', 'Beef_Round',
               'Beef_Shank', 'Beef_Shoulder', 'Beef_Sirloin', 'Beef_Tenderloin']
CLASS_MERGE_MAP = {"Beef_BottomRound": "Beef_Round"}
IMAGE_SIZE = 260
SAVE_REPORTS = True   # False로 하면 이미지 저장 없이 수치만 출력 (빠름)


def find_test_dirs(data_root):
    """train_dataset_*/test 폴더를 자동 탐색."""
    dirs = []
    for entry in sorted(os.listdir(data_root)):
        test_dir = os.path.join(data_root, entry, "test")
        if entry.startswith("train_dataset_") and os.path.isdir(test_dir):
            dirs.append((entry, test_dir))
    return dirs


def collect_images(test_dir, dataset_name):
    """한 test 폴더에서 이미지 수집. (경로, 정답, 출처 데이터셋)"""
    images = []
    # 9클래스 + BottomRound 폴더도 탐색
    scan_classes = CLASS_NAMES + ["Beef_BottomRound"]
    for class_name in scan_classes:
        mapped = CLASS_MERGE_MAP.get(class_name, class_name)
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            for img_path in glob.glob(os.path.join(class_dir, ext)):
                images.append((img_path, mapped, dataset_name))
    return images


# ===== Grad-CAM =====
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = self.output = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'output', o))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[0, class_idx].backward()
        w = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        hm = torch.sum(w * self.output, dim=1).squeeze()
        hm = np.maximum(hm.detach().cpu().numpy(), 0)
        hm /= (hm.max() + 1e-8)
        return hm


def load_model(num_classes):
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def run_all_tests():
    # 모델 로드
    model = load_model(len(CLASS_NAMES))
    grad_cam = GradCAM(model, model.features[-1])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # test 폴더 탐색
    test_dirs = find_test_dirs(DATA_ROOT)
    if not test_dirs:
        print("❌ test 폴더를 찾을 수 없습니다.")
        return

    print(f"\n{'='*100}")
    print(f"🔍 통합 테스트 — 모델: {os.path.basename(MODEL_PATH)}")
    print(f"{'='*100}")
    print(f"  발견된 테스트 폴더:")
    for name, path in test_dirs:
        print(f"    📂 {name}/test")

    # 전체 이미지 수집
    all_images = []
    for name, path in test_dirs:
        imgs = collect_images(path, name)
        all_images.extend(imgs)
        print(f"    → {name}: {len(imgs)}개")
    print(f"  총 이미지: {len(all_images)}개\n")

    # 결과 폴더 생성
    if SAVE_REPORTS:
        os.makedirs(RESULT_DIR, exist_ok=True)
        for cn in CLASS_NAMES:
            os.makedirs(os.path.join(RESULT_DIR, cn), exist_ok=True)
        os.makedirs(os.path.join(RESULT_DIR, "_wrong"), exist_ok=True)

    # ── 통계용 구조 ──
    # 전체 통계
    total_correct, total_count = 0, 0
    class_stats = {n: {"correct": 0, "total": 0, "wrong_preds": []} for n in CLASS_NAMES}

    # 데이터셋별 통계
    ds_stats = {}
    for name, _ in test_dirs:
        ds_stats[name] = {
            "correct": 0, "total": 0,
            "class_stats": {n: {"correct": 0, "total": 0, "wrong_preds": []} for n in CLASS_NAMES}
        }

    # ── 추론 ──
    print("-" * 100)
    print(f"{'데이터셋':<18} | {'파일명':<30} | {'정답':<18} | {'예측':<18} | {'신뢰도':<8} | {'결과'}")
    print("-" * 100)

    for img_path, ground_truth, ds_name in all_images:
        filename = os.path.basename(img_path)
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue

        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(raw_img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(DEVICE)

        with torch.set_grad_enabled(SAVE_REPORTS):
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)

            # 10클래스 모델이면 병합
            if len(CLASS_NAMES) == 10:
                prob[0, 5] += prob[0, 0]
                prob[0, 0] = 0

            conf, pred = torch.max(prob, 1)
            class_idx = pred.item()
            pred_label = CLASS_NAMES[class_idx]
            pred_label = CLASS_MERGE_MAP.get(pred_label, pred_label)
            confidence = conf.item()

        is_correct = pred_label == ground_truth
        mark = "✅" if is_correct else "❌"

        # 전체 통계
        total_count += 1
        if is_correct:
            total_correct += 1
            class_stats[ground_truth]["correct"] += 1
        else:
            class_stats[ground_truth]["wrong_preds"].append((filename, pred_label, confidence, ds_name))
        class_stats[ground_truth]["total"] += 1

        # 데이터셋별 통계
        ds = ds_stats[ds_name]
        ds["total"] += 1
        if is_correct:
            ds["correct"] += 1
            ds["class_stats"][ground_truth]["correct"] += 1
        else:
            ds["class_stats"][ground_truth]["wrong_preds"].append((filename, pred_label, confidence))
        ds["class_stats"][ground_truth]["total"] += 1

        # 리포트 이미지 저장
        if SAVE_REPORTS:
            heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)
            heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
            hmc = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            ov = cv2.addWeighted(img_resized, 0.6, hmc, 0.4, 0)
            combined = np.hstack((img_resized, hmc, ov))

            bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
            txt = f"[{ds_name}] True: {ground_truth} | Pred: {pred_label} ({confidence*100:.1f}%)"
            cv2.putText(bar, txt, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if is_correct else (255, 0, 0), 2)
            final = np.vstack((bar, combined))
            final_bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(RESULT_DIR, ground_truth, f"{ds_name}_{filename}"), final_bgr)
            if not is_correct:
                cv2.imwrite(os.path.join(RESULT_DIR, "_wrong",
                            f"{ds_name}_{ground_truth}_to_{pred_label}_{filename}"), final_bgr)

        print(f"{ds_name:<18} | {filename[:30]:<30} | {ground_truth:<18} | {pred_label:<18} | {confidence*100:>6.1f}% | {mark}")

    # ════════════════════════════════════════════════
    # 결과 요약
    # ════════════════════════════════════════════════

    # 1) 데이터셋별 정확도 비교
    print("\n" + "=" * 100)
    print("📊 [데이터셋별 정확도 비교]")
    print("=" * 100)
    print(f"  {'데이터셋':<22} | {'맞춤':>6} | {'전체':>6} | {'정확도':>10}")
    print(f"  {'-'*22}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}")

    for name, _ in test_dirs:
        ds = ds_stats[name]
        acc = ds["correct"] / ds["total"] * 100 if ds["total"] else 0
        bar = "█" * int(acc // 10) + "░" * (10 - int(acc // 10))
        print(f"  {name:<22} | {ds['correct']:>6} | {ds['total']:>6} | {acc:>6.1f}% {bar}")

    total_acc = total_correct / total_count * 100 if total_count else 0
    print(f"  {'-'*22}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}")
    print(f"  {'합계':<22} | {total_correct:>6} | {total_count:>6} | {total_acc:>6.1f}%")

    # 2) 데이터셋별 세부 클래스 테이블
    for name, _ in test_dirs:
        ds = ds_stats[name]
        ds_acc = ds["correct"] / ds["total"] * 100 if ds["total"] else 0
        print(f"\n  ── {name} (정확도: {ds_acc:.1f}%) ──")
        print(f"  {'클래스':<22} | {'맞춤':>5} | {'전체':>5} | {'정확도':>8} | {'주요 오분류'}")
        for cn in CLASS_NAMES:
            s = ds["class_stats"][cn]
            a = s["correct"] / s["total"] * 100 if s["total"] else 0
            ws = ""
            if s["wrong_preds"]:
                wc = {}
                for _, wp, _, *_ in s["wrong_preds"]:
                    wc[wp] = wc.get(wp, 0) + 1
                ws = ", ".join(f"{k}({v})" for k, v in sorted(wc.items(), key=lambda x: -x[1])[:2])
            bar = "█" * int(a // 10) + "░" * (10 - int(a // 10))
            print(f"  {cn:<22} | {s['correct']:>5} | {s['total']:>5} | {a:>5.1f}% {bar} | {ws}")

    # 3) 전체 클래스별 통합 정확도
    print(f"\n{'='*100}")
    print("📊 [전체 통합 — 클래스별 정확도]")
    print("=" * 100)
    print(f"{'클래스':<22} | {'맞춤':>6} | {'전체':>6} | {'정확도':>10} | {'주요 오분류'}")
    print("-" * 100)

    for cn in CLASS_NAMES:
        s = class_stats[cn]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        ws = ""
        if s["wrong_preds"]:
            wc = {}
            for _, wp, _, *_ in s["wrong_preds"]:
                wc[wp] = wc.get(wp, 0) + 1
            ws = ", ".join(f"{k}({v})" for k, v in sorted(wc.items(), key=lambda x: -x[1])[:3])
        bar = "█" * int(acc // 10) + "░" * (10 - int(acc // 10))
        print(f"{cn:<22} | {s['correct']:>6} | {s['total']:>6} | {acc:>6.1f}% {bar} | {ws}")

    print("=" * 100)
    print(f"🎯 최종 통합 정확도: {total_acc:.2f}% ({total_correct}/{total_count})")
    print(f"   모델: {os.path.basename(MODEL_PATH)}")
    if SAVE_REPORTS:
        print(f"   📂 리포트: {RESULT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    run_all_tests()
