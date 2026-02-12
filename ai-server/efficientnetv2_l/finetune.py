"""
EfficientNetV2-L 소고기 부위 분류 — 학습 & 파인튜닝
기존 EfficientNet-B2 파이프라인과 동일한 구조, 모델만 V2-L로 교체.
"""
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
from collections import Counter

# ===== 설정 =====
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
CONFIG = {
    "train_dir": DATA_ROOT / "train_dataset_1" / "train",
    "val_dir":   DATA_ROOT / "train_dataset_1" / "val",
    "test_dir":  DATA_ROOT / "train_dataset_1" / "test",

    # ── 모델 저장 ──
    "model_save_path": Path(__file__).parent.parent / "models" / "v2l_beef_100-v1.pth",
    "checkpoint_dir":  Path(__file__).parent.parent / "models" / "checkpoints_v2l",
    "history_path":    Path(__file__).parent.parent / "models" / "v2l_training_history.json",

    # ── 파인튜닝 (기존 모델에서 이어서 학습할 때) ──
    "finetune_from": None,  # 예: Path(...) / "v2l_beef_100-v1.pth"

    "freeze_backbone_epochs": 3,   # 초기 N 에폭 동안 backbone 동결 (V2-L은 크므로 권장)

    # ── 학습 하이퍼파라미터 ──
    "num_epochs": 30,
    "batch_size": 8,               # V2-L은 매우 큼 → 배치 줄여야 함
    "learning_rate": 5e-5,         # Backbone (features) 학습률 — 큰 모델은 작게
    "head_learning_rate": 5e-4,    # Classifier (head) 학습률
    "image_size": 480,             # EfficientNetV2-L 기본 입력 크기
    "num_workers": 4,
    "mixup_alpha": 0.2,
    "label_smoothing": 0.1,
    "weight_decay": 1e-2,
    "grad_clip_max_norm": 1.0,
    "patience": 10,
    "warmup_epochs": 3,
    "tta_transforms": 5,
    "use_weighted_sampler": True,
}

# ===== 증강 =====
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
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.02, 0.1), hole_width_range=(0.02, 0.1), p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

tta_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ===== Mixup =====
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===== Dataset 래퍼 =====
class AlbumentationsDataset(torch.utils.data.Dataset):
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


# ===== 모델 생성 — EfficientNetV2-L =====
def create_model(num_classes: int):
    """
    EfficientNetV2-L: 118M 파라미터, 입력 480×480
    B2 대비 약 10배 큰 모델 — 더 깊은 특징 추출 가능
    """
    model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
    # classifier: Sequential(Dropout, Linear(1280, 1000))
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ===== WeightedRandomSampler =====
def make_weighted_sampler(dataset):
    targets = dataset.targets
    class_counts = Counter(targets)
    num_samples = len(targets)
    class_weights = {c: num_samples / cnt for c, cnt in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(sample_weights, num_samples, replacement=True)


# ===== Early Stopping =====
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_score = 0, None

    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"  ⏹ Early Stopping (patience={self.patience}, best={self.best_score:.4f})")
            return True
        return False


# ===== LR Scheduler =====
class WarmupCosineScheduler:
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
            alpha = self.current_epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        else:
            self.cosine.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ===== 평가 =====
def evaluate(model, loader, device, class_names, criterion=None):
    model.eval()
    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion:
                total_loss += criterion(outputs, labels).item() * inputs.size(0)
                total_samples += inputs.size(0)
            preds = outputs.argmax(dim=1)
            for p, t in zip(preds.cpu(), labels.cpu()):
                confusion[p, t] += 1

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    per_class = {}
    for i, name in enumerate(class_names):
        tp = confusion[i, i].item()
        fp = confusion[i, :].sum().item() - tp
        fn = confusion[:, i].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}

    return accuracy, avg_loss, per_class, confusion


def evaluate_with_tta(model, dataset_raw, device, class_names, num_augments=5):
    model.eval()
    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for idx in range(len(dataset_raw)):
        image, label = dataset_raw[idx]
        img_np = np.array(image)
        logits_sum = None
        for i in range(num_augments):
            tf = val_transform if i == 0 else tta_transform
            aug = tf(image=img_np)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(aug), dim=1)
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

    return accuracy, 0.0, per_class, confusion


def print_metrics(accuracy, per_class, class_names, title="Evaluation"):
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
        macro_p += m["precision"]; macro_r += m["recall"]; macro_f1 += m["f1"]
    n = len(class_names)
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  {'Macro Avg':<22} {macro_p/n:>7.4f} {macro_r/n:>7.4f} {macro_f1/n:>7.4f}")
    print(f"{'='*60}\n")


def save_checkpoint(model, class_to_idx, epoch, val_acc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    meta = {
        "model": "EfficientNetV2-L",
        "epoch": epoch, "val_acc": val_acc,
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "image_size": CONFIG["image_size"],
    }
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  💾 모델 저장: {path.name}")


# ===== 메인 =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print("\n[EfficientNetV2-L] 소고기 부위 분류 Fine-tuning")
    print(f"  ⚠ V2-L은 약 118M 파라미터 — VRAM 8GB 이상 권장 (batch_size={CONFIG['batch_size']})")

    # 데이터 로드
    for split, d in [("train", CONFIG["train_dir"]), ("val", CONFIG["val_dir"]), ("test", CONFIG["test_dir"])]:
        if not Path(d).exists():
            raise FileNotFoundError(f"{split} 경로가 없습니다: {d}")

    train_dataset = datasets.ImageFolder(root=str(CONFIG["train_dir"]))
    val_dataset   = datasets.ImageFolder(root=str(CONFIG["val_dir"]))
    test_dataset  = datasets.ImageFolder(root=str(CONFIG["test_dir"]))

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_names)

    train_counts = Counter(train_dataset.targets)
    print(f"  클래스 수: {num_classes}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    for idx in sorted(train_counts.keys()):
        print(f"    {class_names[idx]:<22} {train_counts[idx]:>4}")

    # DataLoader
    if CONFIG["use_weighted_sampler"]:
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(AlbumentationsDataset(train_dataset, train_transform),
                                  batch_size=CONFIG["batch_size"], sampler=sampler,
                                  num_workers=CONFIG["num_workers"], pin_memory=True)
    else:
        train_loader = DataLoader(AlbumentationsDataset(train_dataset, train_transform),
                                  batch_size=CONFIG["batch_size"], shuffle=True,
                                  num_workers=CONFIG["num_workers"], pin_memory=True)

    val_loader = DataLoader(AlbumentationsDataset(val_dataset, val_transform),
                            batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=CONFIG["num_workers"], pin_memory=True)

    # 모델
    model = create_model(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  파라미터: {total_params:.1f}M")

    # 파인튜닝 로드
    if CONFIG["finetune_from"] and Path(CONFIG["finetune_from"]).exists():
        state = torch.load(CONFIG["finetune_from"], map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"  ✓ 파인튜닝 모델 로드: {Path(CONFIG['finetune_from']).name}")

    # Backbone 동결 (초기 에폭)
    freeze_until = CONFIG["freeze_backbone_epochs"]

    # 차등 학습률
    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": CONFIG["learning_rate"]},
        {"params": model.classifier.parameters(), "lr": CONFIG["head_learning_rate"]},
    ], weight_decay=CONFIG["weight_decay"])

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    scheduler = WarmupCosineScheduler(optimizer, CONFIG["warmup_epochs"], CONFIG["num_epochs"])
    early_stopping = EarlyStopping(patience=CONFIG["patience"])

    best_val_acc = 0.0
    CONFIG["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    history = {"config": {"model": "EfficientNetV2-L", **{k: str(v) for k, v in CONFIG.items() if k in ("model_save_path", "finetune_from", "num_epochs", "batch_size", "learning_rate")}}, "epochs": []}

    print(f"\n  학습 시작: {CONFIG['num_epochs']} epochs, batch={CONFIG['batch_size']}")
    print(f"  Backbone freeze: {freeze_until} epochs")
    print(f"{'─'*70}")

    for epoch in range(CONFIG["num_epochs"]):
        epoch_start = time.time()

        # Backbone 동결/해제
        if epoch < freeze_until:
            for p in model.features.parameters():
                p.requires_grad = False
        elif epoch == freeze_until and freeze_until > 0:
            for p in model.features.parameters():
                p.requires_grad = True
            print(f"  🔓 Backbone 해제 (epoch {epoch+1})")

        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, CONFIG["mixup_alpha"])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_max_norm"])
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (lam * (outputs.argmax(1) == labels_a).float().sum()
                              + (1 - lam) * (outputs.argmax(1) == labels_b).float().sum()).item()
            train_total += inputs.size(0)

        # Validation
        val_acc, val_loss, _, _ = evaluate(model, val_loader, device, class_names, criterion)
        scheduler.step()
        elapsed = time.time() - epoch_start
        t_loss = train_loss / train_total
        t_acc = train_correct / train_total
        lrs = scheduler.get_last_lr()

        print(f"  Epoch [{epoch+1:>3}/{CONFIG['num_epochs']}]  "
              f"Train Loss: {t_loss:.4f}  Train Acc: {t_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  |  "
              f"LR: {lrs[0]:.2e}/{lrs[1]:.2e}  |  {elapsed:.1f}s")

        history["epochs"].append({"epoch": epoch+1, "train_loss": round(t_loss, 4), "train_acc": round(t_acc, 4),
                                   "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4), "elapsed": round(elapsed, 1)})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, class_to_idx, epoch+1, val_acc, CONFIG["model_save_path"])
            print(f"  ⭐ Best! (Val Acc: {val_acc:.4f})")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, class_to_idx, epoch+1, val_acc, CONFIG["checkpoint_dir"] / f"epoch_{epoch+1:03d}.pth")

        if early_stopping(val_acc):
            print(f"  → {epoch+1} epoch에서 학습 종료")
            break

    # 히스토리 저장
    history["best_val_acc"] = best_val_acc
    with open(CONFIG["history_path"], "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # Test 평가
    print("\n[Test 세트 최종 평가]")
    best_state = torch.load(CONFIG["model_save_path"], map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    test_loader = DataLoader(AlbumentationsDataset(test_dataset, val_transform),
                             batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=CONFIG["num_workers"], pin_memory=True)
    test_acc, test_loss, test_per_class, test_confusion = evaluate(model, test_loader, device, class_names, criterion)
    print_metrics(test_acc, test_per_class, class_names, title="EfficientNetV2-L — Test")

    if CONFIG["tta_transforms"] > 1:
        tta_acc, _, tta_per_class, _ = evaluate_with_tta(model, test_dataset, device, class_names, CONFIG["tta_transforms"])
        print_metrics(tta_acc, tta_per_class, class_names, title=f"EfficientNetV2-L — TTA (x{CONFIG['tta_transforms']})")

    print(f"\n✅ 완료! Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
