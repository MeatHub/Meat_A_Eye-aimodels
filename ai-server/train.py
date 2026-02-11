import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
from collections import Counter
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# ===== 설정 =====
DATA_ROOT = Path(__file__).parent.parent / "data" / "dataset_final"
CONFIG = {
    # ── 데이터 경로 (train/val/test 폴더가 이미 분리되어 있음) ──
    "train_dir": DATA_ROOT / "train",
    "val_dir":   DATA_ROOT / "val",
    "test_dir":  DATA_ROOT / "test",
    # ── 모델 저장 ──
    "model_save_path": Path(__file__).parent / "models" / "b2_imagenet_beef.pth",
    "checkpoint_dir":  Path(__file__).parent / "models" / "checkpoints",
    # ── 학습 하이퍼파라미터 ──
    "num_epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-4,         # Backbone (features) 학습률
    "head_learning_rate": 1e-3,    # Classifier (head) 학습률
    "image_size": 260,
    "num_workers": 8,
    "mixup_alpha": 0.2,
    "label_smoothing": 0.1,
    "weight_decay": 1e-2,
    "grad_clip_max_norm": 1.0,     # Gradient clipping
    # ── Early Stopping ──
    "patience": 10,                # val_acc 개선 없으면 N epoch 후 중단
    # ── LR Warmup ──
    "warmup_epochs": 3,            # 선형 warmup 에폭 수
    # ── TTA (Test Time Augmentation) ──
    "tta_transforms": 5,           # 테스트 시 증강 횟수 (1 = 증강 없음)
    # ── Class Weighting ──
    "use_weighted_sampler": True,  # 클래스 불균형 보정
    # ── ImageNet 사전학습 (선택) ──
    "imagenet_dataset_id": "ILSVRC/imagenet-1k",
    "imagenet_pretrain_epochs": 0,
    "imagenet_batch_size": 64,
    "imagenet_max_samples": 100_000,
    "imagenet_model_path": Path(__file__).parent / "models" / "efficientnet_b2_imagenet.pth",
    "hf_token": None,
}

# ===== [핵심 1] 증강 전략 — 소고기 부위 질감·색상·마블링 특화 =====
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(0.02, 0.1),
        hole_width_range=(0.02, 0.1),
        p=0.5,
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# TTA용 경량 증강 (테스트 시 여러 번 적용 후 평균)
tta_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ===== [핵심 2] Mixup =====
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===== Dataset 래퍼 =====
class AlbumentationsDataset(torch.utils.data.Dataset):
    """ImageFolder 결과를 Albumentations 증강과 연결."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


class ImageNetAlbumentationsDataset(torch.utils.data.Dataset):
    """HuggingFace ImageNet → Albumentations 호환 래퍼."""
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        image = np.array(row["image"].convert("RGB"))
        label = row["label"]
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


# ===== 모델 생성 =====
def create_model_b2(num_classes: int, pretrained_path=None):
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    if pretrained_path and Path(pretrained_path).exists():
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith("classifier")}
        model.load_state_dict(state, strict=False)
        print(f"  ✓ Backbone 로드: {pretrained_path}")
    return model


def create_model_imagenet():
    return models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)


# ===== WeightedRandomSampler 생성 =====
def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """클래스 불균형 보정을 위한 가중 샘플러 생성."""
    targets = dataset.targets
    class_counts = Counter(targets)
    num_samples = len(targets)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


# ===== Early Stopping =====
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
            return False  # 계속 학습
        self.counter += 1
        if self.counter >= self.patience:
            print(f"  ⏹ Early Stopping 발동 (patience={self.patience}, best={self.best_score:.4f})")
            return True  # 학습 중단
        return False


# ===== LR Warmup + CosineAnnealing 스케줄러 =====
class WarmupCosineScheduler:
    """Linear warmup → CosineAnnealingWarmRestarts."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(total_epochs - warmup_epochs, 1), T_mult=1
        )
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        else:
            self.cosine.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ===== 평가 함수 =====
def evaluate(model, loader, device, class_names):
    """Validation/Test 세트 평가 — accuracy + per-class precision/recall/F1."""
    model.eval()
    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)  # [pred, true]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            for p, t in zip(preds.cpu(), labels.cpu()):
                confusion[p, t] += 1

    # Overall accuracy
    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    accuracy = correct / total if total > 0 else 0.0

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        tp = confusion[i, i].item()
        fp = confusion[i, :].sum().item() - tp
        fn = confusion[:, i].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}

    return accuracy, per_class, confusion


def evaluate_with_tta(model, dataset_raw, device, class_names, num_augments=5):
    """Test Time Augmentation — 여러 번 증강 후 소프트맥스 평균으로 예측."""
    model.eval()
    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for idx in range(len(dataset_raw)):
        image, label = dataset_raw[idx]
        img_np = np.array(image)

        # 원본 (val_transform) + N-1 번 tta_transform
        logits_sum = None
        for i in range(num_augments):
            tf = val_transform if i == 0 else tta_transform
            aug = tf(image=img_np)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(aug)
                probs = torch.softmax(out, dim=1)
            logits_sum = probs if logits_sum is None else logits_sum + probs

        pred = logits_sum.argmax(dim=1).item()
        confusion[pred, label] += 1

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    for i, name in enumerate(class_names):
        tp = confusion[i, i].item()
        fp = confusion[i, :].sum().item() - tp
        fn = confusion[:, i].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}

    return accuracy, per_class, confusion


def print_metrics(accuracy, per_class, class_names, title="Evaluation"):
    """평가 결과를 보기 좋게 출력."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'─'*60}")
    print(f"  {'Class':<22} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7}")
    macro_p, macro_r, macro_f1 = 0, 0, 0
    for name in class_names:
        m = per_class[name]
        print(f"  {name:<22} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")
        macro_p += m["precision"]
        macro_r += m["recall"]
        macro_f1 += m["f1"]
    n = len(class_names)
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  {'Macro Avg':<22} {macro_p/n:>7.4f} {macro_r/n:>7.4f} {macro_f1/n:>7.4f}")
    print(f"{'='*60}\n")


# ===== 모델 저장 (메타데이터 포함) =====
def save_checkpoint(model, class_to_idx, epoch, val_acc, path):
    """모델 가중치 + 메타데이터를 함께 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # state_dict만 저장 (predict_b2.py 호환)
    torch.save(model.state_dict(), path)

    # 메타데이터를 별도 JSON에 저장
    meta = {
        "epoch": epoch,
        "val_acc": val_acc,
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "image_size": CONFIG["image_size"],
    }
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  💾 모델 저장: {path.name}  |  메타: {meta_path.name}")


# ===== 메인 =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    pretrained_path = None

    # ────────────────────────────────────────────────
    # [Phase 1] ImageNet 사전학습 (선택, epochs=0이면 스킵)
    # ────────────────────────────────────────────────
    if CONFIG["imagenet_pretrain_epochs"] > 0:
        print("\n[Phase 1] ImageNet 사전학습 시작")
        hf_token = CONFIG.get("hf_token") or os.environ.get("HF_TOKEN")
        if not hf_token:
            print("  ⚠ ImageNet(gated) 접근을 위해 HF 토큰이 필요합니다.")
            print("  CONFIG['hf_token'] 또는 .env HF_TOKEN 설정 후 재실행.")
            raise RuntimeError("HF_TOKEN not set.")
        try:
            imagenet_dataset = load_dataset(
                CONFIG["imagenet_dataset_id"], split="train", token=hf_token,
            )
        except Exception as e:
            if "gated" in str(e).lower() or "DatasetNotFoundError" in type(e).__name__:
                print("  ⚠ ImageNet은 gated 데이터셋입니다. HF 이용약관 동의 필요.")
            raise
        if CONFIG["imagenet_max_samples"] is not None:
            n = min(len(imagenet_dataset), CONFIG["imagenet_max_samples"])
            imagenet_dataset = imagenet_dataset.select(range(n))
            print(f"  ImageNet 서브셋: {n:,} 샘플")
        ds = ImageNetAlbumentationsDataset(imagenet_dataset, val_transform)
        il = DataLoader(ds, batch_size=CONFIG["imagenet_batch_size"],
                        shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
        model_imagenet = create_model_imagenet().to(device)
        opt = optim.AdamW(model_imagenet.parameters(), lr=1e-4, weight_decay=1e-2)
        crit = nn.CrossEntropyLoss()
        for ep in range(CONFIG["imagenet_pretrain_epochs"]):
            model_imagenet.train()
            running, correct, total = 0.0, 0, 0
            for x, y in il:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model_imagenet(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                running += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
            print(f"  ImageNet Epoch [{ep+1}/{CONFIG['imagenet_pretrain_epochs']}] "
                  f"Loss: {running/len(il):.4f}  Acc: {correct/total:.4f}")
        CONFIG["imagenet_model_path"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_imagenet.state_dict(), CONFIG["imagenet_model_path"])
        pretrained_path = CONFIG["imagenet_model_path"]
        del model_imagenet, il, ds, imagenet_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ────────────────────────────────────────────────
    # [Phase 2] 소고기 부위 분류 Fine-tuning
    # ────────────────────────────────────────────────
    print("\n[Phase 2] 소고기 부위 분류 Fine-tuning")

    # 데이터 로드 (실제 train / val / test 디렉토리 사용)
    for split, d in [("train", CONFIG["train_dir"]), ("val", CONFIG["val_dir"]), ("test", CONFIG["test_dir"])]:
        if not Path(d).exists():
            raise FileNotFoundError(f"{split} 경로가 없습니다: {d}")

    train_dataset = datasets.ImageFolder(root=str(CONFIG["train_dir"]))
    val_dataset   = datasets.ImageFolder(root=str(CONFIG["val_dir"]))
    test_dataset  = datasets.ImageFolder(root=str(CONFIG["test_dir"]))

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_names)

    # 클래스 분포 출력
    train_counts = Counter(train_dataset.targets)
    print(f"  클래스 수: {num_classes}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"  클래스 분포 (train):")
    for idx in sorted(train_counts.keys()):
        name = class_names[idx]
        count = train_counts[idx]
        bar = "█" * (count // 10)
        print(f"    {name:<22} {count:>4}  {bar}")

    # DataLoader 구성
    if CONFIG["use_weighted_sampler"]:
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            AlbumentationsDataset(train_dataset, train_transform),
            batch_size=CONFIG["batch_size"],
            sampler=sampler,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
        )
        print("  ✓ WeightedRandomSampler 적용 (클래스 불균형 보정)")
    else:
        train_loader = DataLoader(
            AlbumentationsDataset(train_dataset, train_transform),
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
        )

    val_loader = DataLoader(
        AlbumentationsDataset(val_dataset, val_transform),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    # 모델
    model = create_model_b2(num_classes, pretrained_path=pretrained_path).to(device)

    # 차등 학습률: Backbone 느리게, Head 빠르게
    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": CONFIG["learning_rate"]},
        {"params": model.classifier.parameters(), "lr": CONFIG["head_learning_rate"]},
    ], weight_decay=CONFIG["weight_decay"])

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    scheduler = WarmupCosineScheduler(optimizer, CONFIG["warmup_epochs"], CONFIG["num_epochs"])
    early_stopping = EarlyStopping(patience=CONFIG["patience"])

    best_val_acc = 0.0
    CONFIG["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    print(f"\n  학습 시작: {CONFIG['num_epochs']} epochs, batch={CONFIG['batch_size']}, "
          f"warmup={CONFIG['warmup_epochs']}, patience={CONFIG['patience']}")
    print(f"  LR backbone={CONFIG['learning_rate']}, head={CONFIG['head_learning_rate']}")
    print(f"{'─'*70}")

    for epoch in range(CONFIG["num_epochs"]):
        epoch_start = time.time()

        # === Training ===
        model.train()
        train_loss, train_correct, train_total = 0.0, 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, CONFIG["mixup_alpha"])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_max_norm"])

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (lam * (outputs.argmax(1) == labels_a).float().sum()
                              + (1 - lam) * (outputs.argmax(1) == labels_b).float().sum()).item()
            train_total += inputs.size(0)

        # === Validation ===
        val_acc, val_per_class, _ = evaluate(model, val_loader, device, class_names)

        scheduler.step()
        elapsed = time.time() - epoch_start
        t_loss = train_loss / train_total
        t_acc  = train_correct / train_total
        lrs = scheduler.get_last_lr()

        print(f"  Epoch [{epoch+1:>3}/{CONFIG['num_epochs']}]  "
              f"Train Loss: {t_loss:.4f}  Train Acc: {t_acc:.4f}  |  "
              f"Val Acc: {val_acc:.4f}  |  "
              f"LR: {lrs[0]:.2e}/{lrs[1]:.2e}  |  {elapsed:.1f}s")

        # Best 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, class_to_idx, epoch + 1, val_acc, CONFIG["model_save_path"])
            print(f"  ⭐ Best Model Updated! (Val Acc: {val_acc:.4f})")

        # 주기적 체크포인트 (10 에폭마다)
        if (epoch + 1) % 10 == 0:
            ckpt_path = CONFIG["checkpoint_dir"] / f"epoch_{epoch+1:03d}.pth"
            save_checkpoint(model, class_to_idx, epoch + 1, val_acc, ckpt_path)

        # Early Stopping 확인
        if early_stopping(val_acc):
            print(f"  → {epoch+1} epoch에서 학습 종료 (Early Stopping)")
            break

    # ────────────────────────────────────────────────
    # [Phase 3] Test 세트 최종 평가
    # ────────────────────────────────────────────────
    print("\n[Phase 3] Test 세트 최종 평가")

    # Best 모델 다시 로드
    best_state = torch.load(CONFIG["model_save_path"], map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    model.eval()

    # 일반 평가
    test_loader = DataLoader(
        AlbumentationsDataset(test_dataset, val_transform),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    test_acc, test_per_class, test_confusion = evaluate(model, test_loader, device, class_names)
    print_metrics(test_acc, test_per_class, class_names, title="Test Set — 일반 평가")

    # TTA 평가
    if CONFIG["tta_transforms"] > 1:
        print(f"  TTA 평가 중 (augments={CONFIG['tta_transforms']})...")
        tta_acc, tta_per_class, tta_confusion = evaluate_with_tta(
            model, test_dataset, device, class_names, CONFIG["tta_transforms"]
        )
        print_metrics(tta_acc, tta_per_class, class_names, title=f"Test Set — TTA (x{CONFIG['tta_transforms']})")

    # Confusion Matrix 출력
    print("  Confusion Matrix (rows=predicted, cols=actual):")
    header = "          " + " ".join(f"{n[:6]:>6}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join(f"{test_confusion[i, j].item():>6}" for j in range(num_classes))
        print(f"  {name[:8]:>8}  {row}")

    print(f"\n✅ 학습 완료! Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"   모델: {CONFIG['model_save_path']}")
    print(f"   메타: {CONFIG['model_save_path'].with_suffix('.json')}")


if __name__ == "__main__":
    main()