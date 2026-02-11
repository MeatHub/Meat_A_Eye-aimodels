"""
Fine-tuning Script for Existing Beef Classification Model

기존 학습된 가중치(b2_imagenet_beef.pth)에서 이어서 학습하는 스크립트입니다.
새로운 데이터를 추가했거나, 특정 클래스를 보강한 후 사용하세요.

Usage:
    python finetune.py
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
from dotenv import load_dotenv

load_dotenv()

# ===== Fine-tuning 설정 =====
DATA_ROOT = Path(__file__).parent.parent / "data" / "train_dataset_v4"

FINETUNE_CONFIG = {
    # ── 데이터 경로 ──
    "train_dir": DATA_ROOT / "train",
    "val_dir":   DATA_ROOT / "val",
    "test_dir":  DATA_ROOT / "test",
    
    # ── 기존 모델 경로 (이어서 학습할 모델) ──
    "pretrained_model": Path(__file__).parent.parent / "models" / "b2_imagenet_beef.pth",
    "pretrained_meta":  Path(__file__).parent.parent / "models" / "b2_imagenet_beef.json",
    
    # ── 새 모델 저장 경로 ──
    "model_save_path": Path(__file__).parent.parent / "models" / "b2_imagenet_beef_finetuned.pth",
    "checkpoint_dir":  Path(__file__).parent.parent / "models" / "checkpoints_finetuned",
    
    # ── Fine-tuning 하이퍼파라미터 (기존보다 낮은 LR 권장) ──
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 5e-5,         # 기존 1e-4 → 더 낮게 (fine-tuning)
    "head_learning_rate": 5e-4,    # 기존 1e-3 → 더 낮게
    "image_size": 260,
    "num_workers": 8,
    "mixup_alpha": 0.15,           # 약간 낮춤 (이미 잘 학습된 모델)
    "label_smoothing": 0.05,       # 약간 낮춤
    "weight_decay": 1e-2,
    "grad_clip_max_norm": 1.0,
    
    # ── Early Stopping ──
    "patience": 8,
    
    # ── LR Warmup (짧게) ──
    "warmup_epochs": 1,
    
    # ── TTA ──
    "tta_transforms": 5,
    
    # ── Class Weighting ──
    "use_weighted_sampler": True,
    
    # ── Focus Classes (낮은 성능 클래스에 추가 가중치) ──
    "focus_classes": ["Beef_BottomRound", "Beef_Shoulder"],
    "focus_weight_multiplier": 1.5,  # 이 클래스들의 샘플 가중치 1.5배
}

# ===== Augmentation (기존과 동일) =====
train_transform = A.Compose([
    A.Resize(FINETUNE_CONFIG["image_size"], FINETUNE_CONFIG["image_size"]),
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
    A.Resize(FINETUNE_CONFIG["image_size"], FINETUNE_CONFIG["image_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

tta_transform = A.Compose([
    A.Resize(FINETUNE_CONFIG["image_size"], FINETUNE_CONFIG["image_size"]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ===== Mixup =====
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


# ===== Dataset =====
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


# ===== Model Loading (기존 가중치 로드) =====
def load_pretrained_model(pretrained_path, meta_path, device):
    """기존 학습된 모델을 로드합니다."""
    
    # 메타데이터 로드
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    num_classes = meta["num_classes"]
    class_to_idx = meta["class_to_idx"]
    
    # 모델 구조 생성
    model = models.efficientnet_b2(weights=None)  # 사전학습 가중치 사용 안함
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    
    # 학습된 가중치 로드
    state = torch.load(pretrained_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    
    print(f"  ✓ 기존 모델 로드 완료: {pretrained_path}")
    print(f"    - Epoch: {meta.get('epoch', 'N/A')}")
    print(f"    - Val Acc: {meta.get('val_acc', 'N/A'):.4f}")
    print(f"    - Classes: {num_classes}")
    
    return model.to(device), class_to_idx, meta


# ===== WeightedRandomSampler with Focus Classes =====
def make_focused_weighted_sampler(dataset, class_to_idx, focus_classes, focus_multiplier=1.5):
    """특정 클래스(focus_classes)에 추가 가중치를 부여하는 샘플러."""
    targets = dataset.targets
    class_counts = Counter(targets)
    num_samples = len(targets)
    
    # 기본 클래스 가중치 계산
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    
    # Focus 클래스에 추가 가중치
    focus_indices = [class_to_idx[c] for c in focus_classes if c in class_to_idx]
    for idx in focus_indices:
        if idx in class_weights:
            class_weights[idx] *= focus_multiplier
            class_name = [k for k, v in class_to_idx.items() if v == idx][0]
            print(f"    ✓ Focus class '{class_name}' weight x{focus_multiplier}")
    
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


# ===== Early Stopping =====
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ===== LR Scheduler =====
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=1e-7):
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


# ===== Evaluation =====
def evaluate(model, loader, device, class_names):
    model.eval()
    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            for p, t in zip(preds.cpu(), labels.cpu()):
                confusion[p, t] += 1

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
        mark = " ⚠️" if m['f1'] < 0.92 else ""
        print(f"  {name:<22} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}{mark}")
        macro_p += m["precision"]
        macro_r += m["recall"]
        macro_f1 += m["f1"]
    n = len(class_names)
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  {'Macro Avg':<22} {macro_p/n:>7.4f} {macro_r/n:>7.4f} {macro_f1/n:>7.4f}")
    print(f"{'='*60}\n")


# ===== Checkpoint Save =====
def save_checkpoint(model, class_to_idx, epoch, val_acc, path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    
    meta = {
        "epoch": epoch,
        "val_acc": val_acc,
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "image_size": config["image_size"],
        "fine_tuned": True,
        "base_model": str(config["pretrained_model"]),
    }
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  💾 모델 저장: {path.name}  |  메타: {meta_path.name}")


# ===== Main =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "="*60)
    print("  Fine-tuning 모드 - 기존 모델에서 이어서 학습")
    print("="*60)

    # ── 기존 모델 로드 ──
    model, class_to_idx, meta = load_pretrained_model(
        FINETUNE_CONFIG["pretrained_model"],
        FINETUNE_CONFIG["pretrained_meta"],
        device
    )
    class_names = list(class_to_idx.keys())
    num_classes = len(class_names)

    # ── 데이터 로드 ──
    print("\n[데이터 로드]")
    train_dataset = datasets.ImageFolder(root=str(FINETUNE_CONFIG["train_dir"]))
    val_dataset   = datasets.ImageFolder(root=str(FINETUNE_CONFIG["val_dir"]))
    test_dataset  = datasets.ImageFolder(root=str(FINETUNE_CONFIG["test_dir"]))

    # 클래스 일치 확인
    if train_dataset.class_to_idx != class_to_idx:
        print("  ⚠️ 경고: 데이터셋 클래스와 모델 클래스가 다릅니다!")
        print(f"    모델: {list(class_to_idx.keys())}")
        print(f"    데이터: {train_dataset.classes}")
        # 계속 진행 여부 확인
        response = input("  계속 진행하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            return

    # 클래스 분포 출력
    train_counts = Counter(train_dataset.targets)
    print(f"  클래스 수: {num_classes}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"  클래스 분포 (train):")
    for idx in sorted(train_counts.keys()):
        name = class_names[idx]
        count = train_counts[idx]
        focus_mark = " ← FOCUS" if name in FINETUNE_CONFIG["focus_classes"] else ""
        bar = "█" * (count // 10)
        print(f"    {name:<22} {count:>4}  {bar}{focus_mark}")

    # ── DataLoader ──
    if FINETUNE_CONFIG["use_weighted_sampler"]:
        sampler = make_focused_weighted_sampler(
            train_dataset, 
            class_to_idx,
            FINETUNE_CONFIG["focus_classes"],
            FINETUNE_CONFIG["focus_weight_multiplier"]
        )
        train_loader = DataLoader(
            AlbumentationsDataset(train_dataset, train_transform),
            batch_size=FINETUNE_CONFIG["batch_size"],
            sampler=sampler,
            num_workers=FINETUNE_CONFIG["num_workers"],
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            AlbumentationsDataset(train_dataset, train_transform),
            batch_size=FINETUNE_CONFIG["batch_size"],
            shuffle=True,
            num_workers=FINETUNE_CONFIG["num_workers"],
            pin_memory=True,
        )

    val_loader = DataLoader(
        AlbumentationsDataset(val_dataset, val_transform),
        batch_size=FINETUNE_CONFIG["batch_size"],
        shuffle=False,
        num_workers=FINETUNE_CONFIG["num_workers"],
        pin_memory=True,
    )

    # ── Optimizer (낮은 LR로 fine-tuning) ──
    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": FINETUNE_CONFIG["learning_rate"]},
        {"params": model.classifier.parameters(), "lr": FINETUNE_CONFIG["head_learning_rate"]},
    ], weight_decay=FINETUNE_CONFIG["weight_decay"])

    criterion = nn.CrossEntropyLoss(label_smoothing=FINETUNE_CONFIG["label_smoothing"])
    scheduler = WarmupCosineScheduler(
        optimizer, 
        FINETUNE_CONFIG["warmup_epochs"], 
        FINETUNE_CONFIG["num_epochs"]
    )
    early_stopping = EarlyStopping(patience=FINETUNE_CONFIG["patience"])

    # 기존 모델의 val_acc를 초기 best로 설정
    best_val_acc = meta.get("val_acc", 0.0)
    print(f"\n  초기 Best Val Acc (기존 모델): {best_val_acc:.4f}")
    
    FINETUNE_CONFIG["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    print(f"\n  Fine-tuning 시작: {FINETUNE_CONFIG['num_epochs']} epochs")
    print(f"  LR backbone={FINETUNE_CONFIG['learning_rate']}, head={FINETUNE_CONFIG['head_learning_rate']}")
    print(f"{'─'*70}")

    for epoch in range(FINETUNE_CONFIG["num_epochs"]):
        epoch_start = time.time()

        # === Training ===
        model.train()
        train_loss, train_correct, train_total = 0.0, 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, FINETUNE_CONFIG["mixup_alpha"])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FINETUNE_CONFIG["grad_clip_max_norm"])
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

        print(f"  Epoch [{epoch+1:>3}/{FINETUNE_CONFIG['num_epochs']}]  "
              f"Train Loss: {t_loss:.4f}  Train Acc: {t_acc:.4f}  |  "
              f"Val Acc: {val_acc:.4f}  |  "
              f"LR: {lrs[0]:.2e}/{lrs[1]:.2e}  |  {elapsed:.1f}s")

        # Best 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, class_to_idx, epoch + 1, val_acc, 
                          FINETUNE_CONFIG["model_save_path"], FINETUNE_CONFIG)
            print(f"  ⭐ Best Model Updated! (Val Acc: {val_acc:.4f})")

        # 주기적 체크포인트
        if (epoch + 1) % 10 == 0:
            ckpt_path = FINETUNE_CONFIG["checkpoint_dir"] / f"finetune_epoch_{epoch+1:03d}.pth"
            save_checkpoint(model, class_to_idx, epoch + 1, val_acc, ckpt_path, FINETUNE_CONFIG)

        # Early Stopping
        if early_stopping(val_acc):
            print(f"  ⏹ Early Stopping at epoch {epoch+1}")
            break

    # ── Test 평가 ──
    print("\n[Test 세트 최종 평가]")
    
    # Best 모델 로드
    if FINETUNE_CONFIG["model_save_path"].exists():
        best_state = torch.load(FINETUNE_CONFIG["model_save_path"], map_location=device, weights_only=True)
        model.load_state_dict(best_state)
    
    model.eval()

    test_loader = DataLoader(
        AlbumentationsDataset(test_dataset, val_transform),
        batch_size=FINETUNE_CONFIG["batch_size"],
        shuffle=False,
        num_workers=FINETUNE_CONFIG["num_workers"],
        pin_memory=True,
    )
    
    test_acc, test_per_class, test_confusion = evaluate(model, test_loader, device, class_names)
    print_metrics(test_acc, test_per_class, class_names, title="Test Set — Fine-tuned Model")

    # TTA 평가
    if FINETUNE_CONFIG["tta_transforms"] > 1:
        print(f"  TTA 평가 중 (augments={FINETUNE_CONFIG['tta_transforms']})...")
        tta_acc, tta_per_class, _ = evaluate_with_tta(
            model, test_dataset, device, class_names, FINETUNE_CONFIG["tta_transforms"]
        )
        print_metrics(tta_acc, tta_per_class, class_names, 
                     title=f"Test Set — TTA (x{FINETUNE_CONFIG['tta_transforms']})")

    print(f"\n✅ Fine-tuning 완료! Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"   모델: {FINETUNE_CONFIG['model_save_path']}")


if __name__ == "__main__":
    main()
