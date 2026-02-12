"""
Swin Transformer — 테스트 시각화 (Attention Rollout 기반 히트맵 + 클래스별 정확도)
⚠ Swin은 CNN이 아니라서 기존 Grad-CAM이 잘 동작하지 않을 수 있음.
   → Attention Rollout 또는 마지막 특성 맵에서 Grad-CAM 적용.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 모델 선택 ──
MODEL_VARIANT = "base"   # "base" 또는 "large"
USE_V2 = True

v_str = "v2" if USE_V2 else ""
MODEL_PATH = rf"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\ai-server\models\swin{v_str}_{MODEL_VARIANT}_beef-v1.pth"
TEST_IMAGE_DIR = r"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\data\train_dataset_3\test"
RESULT_DIR = rf"C:\Pyg\Projects\meathub\Meat_A_Eye-aimodels\test_results_swin{v_str}_{MODEL_VARIANT}"

CLASS_NAMES = ['Beef_Brisket', 'Beef_Chuck', 'Beef_Rib', 'Beef_Ribeye', 'Beef_Round',
               'Beef_Shank', 'Beef_Shoulder', 'Beef_Sirloin', 'Beef_Tenderloin']
IMAGE_SIZE = 256 if USE_V2 else 224

os.makedirs(RESULT_DIR, exist_ok=True)


def collect_test_images(base_dir):
    images = []
    for cn in CLASS_NAMES:
        cd = os.path.join(base_dir, cn)
        if not os.path.exists(cd): continue
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            for p in glob.glob(os.path.join(cd, ext)):
                images.append((p, cn))
    return images


class SwinGradCAM:
    """
    Swin Transformer의 마지막 스테이지에서 Grad-CAM 추출.
    Swin의 features[-1]은 공간 정보를 가진 특성 맵을 출력하므로
    reshape 후 CNN과 유사하게 Grad-CAM 적용 가능.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.output = None
        target_layer.register_forward_hook(self._save_output)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_output(self, module, input, output):
        self.output = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        # Swin 특성 맵: (B, H*W, C) 형태 → (B, C, H, W)로 변환
        feats = self.output
        grads = self.gradients

        if feats.dim() == 3:
            # (B, H*W, C) → (B, C, H, W)
            B, HW, C = feats.shape
            H = W = int(HW ** 0.5)
            feats = feats.permute(0, 2, 1).reshape(B, C, H, W)
            grads = grads.permute(0, 2, 1).reshape(B, C, H, W)

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * feats, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap


def load_model(num_classes):
    if USE_V2:
        model = models.swin_v2_b(weights=None)
    else:
        model = models.swin_b(weights=None)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


model = load_model(len(CLASS_NAMES))
# Swin 마지막 스테이지의 마지막 블록
target_layer = model.features[-1][-1]  # 마지막 SwinTransformerBlock
grad_cam = SwinGradCAM(model, target_layer)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def run_visual_test():
    v = "V2" if USE_V2 else "V1"
    variant_name = f"Swin{v}-{MODEL_VARIANT.upper()}"
    image_list = collect_test_images(TEST_IMAGE_DIR)
    print(f"\n🚀 [{variant_name}] 총 {len(image_list)}개 이미지 검증 중...")
    print(f"📂 테스트 폴더: {TEST_IMAGE_DIR}")
    print("-" * 90)
    print(f"{'파일명':<35} | {'실제 정답':<18} | {'모델 예측':<18} | {'신뢰도':<8} | {'결과'}")
    print("-" * 90)

    correct_count, total_count = 0, 0
    class_stats = {n: {"correct": 0, "total": 0, "wrong_preds": []} for n in CLASS_NAMES}

    for cn in CLASS_NAMES:
        os.makedirs(os.path.join(RESULT_DIR, cn), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "_wrong"), exist_ok=True)

    for img_path, gt in image_list:
        fn = os.path.basename(img_path)
        raw = cv2.imread(img_path)
        if raw is None: continue

        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
        inp = transform(Image.fromarray(resized)).unsqueeze(0).to(DEVICE)

        with torch.set_grad_enabled(True):
            out = model(inp)
            prob = torch.nn.functional.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)
            cidx = pred.item()
            plabel = CLASS_NAMES[cidx]
            confidence = conf.item()

        ok = plabel == gt
        if ok:
            correct_count += 1; class_stats[gt]["correct"] += 1
        else:
            class_stats[gt]["wrong_preds"].append((fn, plabel, confidence))
        class_stats[gt]["total"] += 1
        total_count += 1

        # 히트맵 생성 (Swin에서는 실패할 수 있음 → fallback)
        try:
            hm = grad_cam.generate_heatmap(inp, cidx)
            hm = cv2.resize(hm, (IMAGE_SIZE, IMAGE_SIZE))
            hmc = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
            ov = cv2.addWeighted(resized, 0.6, hmc, 0.4, 0)
            combined = np.hstack((resized, hmc, ov))
        except Exception as e:
            # Grad-CAM 실패 시 원본만 표시
            blank = np.zeros_like(resized)
            combined = np.hstack((resized, blank, resized))

        bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
        txt = f"True: {gt} | Pred: {plabel} ({confidence*100:.1f}%)"
        cv2.putText(bar, txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if ok else (255, 0, 0), 2)
        final = np.vstack((bar, combined))

        cv2.imwrite(os.path.join(RESULT_DIR, gt, f"report_{fn}"), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        if not ok:
            cv2.imwrite(os.path.join(RESULT_DIR, "_wrong", f"{gt}_to_{plabel}_{fn}"),
                        cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

        mark = "✅" if ok else "❌"
        print(f"{fn[:35]:<35} | {gt:<18} | {plabel:<18} | {confidence*100:>6.1f}% | {mark}")

    print("\n" + "=" * 90)
    print(f"📊 [{variant_name} 클래스별 정확도]")
    print("=" * 90)
    print(f"{'클래스':<22} | {'맞춤':>6} | {'전체':>6} | {'정확도':>10} | {'주요 오분류'}")
    print("-" * 90)
    for n in CLASS_NAMES:
        s = class_stats[n]
        acc = s["correct"]/s["total"]*100 if s["total"] else 0
        ws = ""
        if s["wrong_preds"]:
            wc = {}
            for _, wp, _ in s["wrong_preds"]: wc[wp] = wc.get(wp, 0) + 1
            ws = ", ".join(f"{k}({v})" for k, v in sorted(wc.items(), key=lambda x: -x[1])[:2])
        bar_vis = "█" * int(acc // 10) + "░" * (10 - int(acc // 10))
        print(f"{n:<22} | {s['correct']:>6} | {s['total']:>6} | {acc:>6.1f}% {bar_vis} | {ws}")

    accuracy = correct_count/total_count*100 if total_count else 0
    print("=" * 90)
    print(f"🎯 최종 정확도: {accuracy:.2f}% ({correct_count}/{total_count})")
    print(f"📂 리포트: {RESULT_DIR}")


if __name__ == "__main__":
    run_visual_test()
