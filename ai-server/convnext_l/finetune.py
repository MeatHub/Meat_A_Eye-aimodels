"""
ConvNeXt-Large 소고기 부위 분류 — 학습 & 파인튜닝
ConvNeXt: 순수 CNN이지만 Transformer 설계 원칙 적용 (Meta, 2022)
ConvNeXt-Large: 198M 파라미터 | ConvNeXt-XL: 350M 파라미터
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

# ── 모델 선택: "large" 또는 "xlarge" ──
MODEL_VARIANT = "large"   # ← "xlarge"으로 변경하면 ConvNeXt-XL 사용

CONFIG = {
    "train_dir": DATA_ROOT / "train_dataset_3" / "train",
    "val_dir":   DATA_ROOT / "train_dataset_3" / "val",
    "test_dir":  DATA_ROOT / "train_dataset_3" / "test",

    "model_save_path": Path(__file__).parent.parent / "models" / f"convnext_{MODEL_VARIANT}_beef-v1.pth",
    "checkpoint_dir":  Path(__file__).parent.parent / "models" / f"checkpoints_convnext_{MODEL_VARIANT}",
    "history_path":    Path(__file__).parent.parent / "models" / f"convnext_{MODEL_VARIANT}_training_history.json",

    "finetune_from": None,

    "freeze_backbone_epochs": 3,

    "num_epochs": 30,
    "batch_size": 8 if MODEL_VARIANT == "xlarge" else 12,  # XL은 더 작게
    "learning_rate": 5e-5,
    "head_learning_rate": 5e-4,
    "image_size": 224,             # ConvNeXt 기본 입력 크기
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
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset, self.transform = dataset, transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform: image = self.transform(image=image)["image"]
        return image, label


# ===== 모델 생성 — ConvNeXt =====
def create_model(num_classes: int):
    """
    ConvNeXt-Large:  198M params, classifier[-1].in_features = 1536
    ConvNeXt-XLarge: 350M params, classifier[-1].in_features = 2048
    """
    if MODEL_VARIANT == "xlarge":
        model = models.convnext_xlarge(weights=models.ConvNeXt_XLarge_Weights.DEFAULT)
    else:
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)

    # ConvNeXt classifier: Sequential(LayerNorm2d, Flatten, Linear)
    in_features = model.classifier[2].in_features  # Large=1536, XL=2048
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def make_weighted_sampler(dataset):
    targets = dataset.targets
    class_counts = Counter(targets)
    n = len(targets)
    weights = [n / class_counts[t] for t in targets]
    return WeightedRandomSampler(weights, n, replacement=True)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_score = 0, None
    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score, self.counter = val_acc, 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"  ⏹ Early Stopping (best={self.best_score:.4f})")
            return True
        return False


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=1e-6):
        self.optimizer, self.warmup_epochs = optimizer, warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(total_epochs - warmup_epochs, 1))
        self.current_epoch = 0
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            a = self.current_epoch / self.warmup_epochs
            for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = self.warmup_start_lr + a * (blr - self.warmup_start_lr)
        else:
            self.cosine.step()
    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def evaluate(model, loader, device, class_names, criterion=None):
    model.eval()
    nc = len(class_names)
    confusion = torch.zeros(nc, nc, dtype=torch.long)
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for inp, lab in loader:
            inp, lab = inp.to(device), lab.to(device)
            out = model(inp)
            if criterion:
                total_loss += criterion(out, lab).item() * inp.size(0)
                total_samples += inp.size(0)
            for p, t in zip(out.argmax(1).cpu(), lab.cpu()):
                confusion[p, t] += 1
    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    acc = correct / total if total else 0
    avg_loss = total_loss / total_samples if total_samples else 0
    per_class = {}
    for i, name in enumerate(class_names):
        tp = confusion[i, i].item()
        fp = confusion[i, :].sum().item() - tp
        fn = confusion[:, i].sum().item() - tp
        p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        per_class[name] = {"precision": p, "recall": r, "f1": f1}
    return acc, avg_loss, per_class, confusion


def evaluate_with_tta(model, dataset_raw, device, class_names, num_augments=5):
    model.eval()
    nc = len(class_names)
    confusion = torch.zeros(nc, nc, dtype=torch.long)
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
        confusion[logits_sum.argmax(1).item(), label] += 1
    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    acc = correct / total if total else 0
    per_class = {}
    for i, name in enumerate(class_names):
        tp = confusion[i,i].item(); fp = confusion[i,:].sum().item()-tp; fn = confusion[:,i].sum().item()-tp
        p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
        per_class[name] = {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0}
    return acc, 0.0, per_class, confusion


def print_metrics(accuracy, per_class, class_names, title="Evaluation"):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'─'*60}")
    print(f"  {'Class':<22} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    mp, mr, mf = 0, 0, 0
    for name in class_names:
        m = per_class[name]
        print(f"  {name:<22} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")
        mp += m["precision"]; mr += m["recall"]; mf += m["f1"]
    n = len(class_names)
    print(f"  {'Macro Avg':<22} {mp/n:>7.4f} {mr/n:>7.4f} {mf/n:>7.4f}\n{'='*60}\n")


def save_checkpoint(model, class_to_idx, epoch, val_acc, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    meta = {"model": f"ConvNeXt-{MODEL_VARIANT.upper()}", "epoch": epoch, "val_acc": val_acc,
            "num_classes": len(class_to_idx), "class_to_idx": class_to_idx,
            "idx_to_class": {v:k for k,v in class_to_idx.items()}, "image_size": CONFIG["image_size"]}
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  💾 모델 저장: {path.name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant_name = f"ConvNeXt-{MODEL_VARIANT.upper()}"
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    print(f"\n[{variant_name}] 소고기 부위 분류 Fine-tuning")

    for split, d in [("train", CONFIG["train_dir"]), ("val", CONFIG["val_dir"]), ("test", CONFIG["test_dir"])]:
        if not Path(d).exists():
            raise FileNotFoundError(f"{split}: {d}")

    train_dataset = datasets.ImageFolder(root=str(CONFIG["train_dir"]))
    val_dataset   = datasets.ImageFolder(root=str(CONFIG["val_dir"]))
    test_dataset  = datasets.ImageFolder(root=str(CONFIG["test_dir"]))
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_names)

    train_counts = Counter(train_dataset.targets)
    print(f"  클래스 수: {num_classes} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    for idx in sorted(train_counts.keys()):
        print(f"    {class_names[idx]:<22} {train_counts[idx]:>4}")

    if CONFIG["use_weighted_sampler"]:
        train_loader = DataLoader(AlbumentationsDataset(train_dataset, train_transform),
                                  batch_size=CONFIG["batch_size"], sampler=make_weighted_sampler(train_dataset),
                                  num_workers=CONFIG["num_workers"], pin_memory=True)
    else:
        train_loader = DataLoader(AlbumentationsDataset(train_dataset, train_transform),
                                  batch_size=CONFIG["batch_size"], shuffle=True,
                                  num_workers=CONFIG["num_workers"], pin_memory=True)

    val_loader = DataLoader(AlbumentationsDataset(val_dataset, val_transform),
                            batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=CONFIG["num_workers"], pin_memory=True)

    model = create_model(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  파라미터: {total_params:.1f}M")

    if CONFIG["finetune_from"] and Path(CONFIG["finetune_from"]).exists():
        model.load_state_dict(torch.load(CONFIG["finetune_from"], map_location=device, weights_only=True))
        print(f"  ✓ 파인튜닝 로드: {Path(CONFIG['finetune_from']).name}")

    freeze_until = CONFIG["freeze_backbone_epochs"]

    # ConvNeXt: features (backbone), classifier (head)
    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": CONFIG["learning_rate"]},
        {"params": model.classifier.parameters(), "lr": CONFIG["head_learning_rate"]},
    ], weight_decay=CONFIG["weight_decay"])

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    scheduler = WarmupCosineScheduler(optimizer, CONFIG["warmup_epochs"], CONFIG["num_epochs"])
    early_stopping = EarlyStopping(patience=CONFIG["patience"])

    best_val_acc = 0.0
    CONFIG["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    history = {"config": {"model": variant_name}, "epochs": []}

    print(f"\n  학습 시작: {CONFIG['num_epochs']} epochs, batch={CONFIG['batch_size']}")
    print(f"{'─'*70}")

    for epoch in range(CONFIG["num_epochs"]):
        t0 = time.time()

        if epoch < freeze_until:
            for p in model.features.parameters(): p.requires_grad = False
        elif epoch == freeze_until and freeze_until > 0:
            for p in model.features.parameters(): p.requires_grad = True
            print(f"  🔓 Backbone 해제 (epoch {epoch+1})")

        model.train()
        tl, tc, tt = 0.0, 0.0, 0
        for inp, lab in train_loader:
            inp, lab = inp.to(device), lab.to(device)
            inp, la, lb, lam = mixup_data(inp, lab, CONFIG["mixup_alpha"])
            optimizer.zero_grad()
            out = model(inp)
            loss = mixup_criterion(criterion, out, la, lb, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_max_norm"])
            optimizer.step()
            tl += loss.item() * inp.size(0)
            tc += (lam*(out.argmax(1)==la).float().sum() + (1-lam)*(out.argmax(1)==lb).float().sum()).item()
            tt += inp.size(0)

        va, vl, _, _ = evaluate(model, val_loader, device, class_names, criterion)
        scheduler.step()
        lrs = scheduler.get_last_lr()
        elapsed = time.time() - t0

        print(f"  Epoch [{epoch+1:>3}/{CONFIG['num_epochs']}]  "
              f"Train Loss: {tl/tt:.4f}  Train Acc: {tc/tt:.4f}  |  "
              f"Val Loss: {vl:.4f}  Val Acc: {va:.4f}  |  "
              f"LR: {lrs[0]:.2e}/{lrs[1]:.2e}  |  {elapsed:.1f}s")

        history["epochs"].append({"epoch": epoch+1, "train_loss": round(tl/tt,4), "val_acc": round(va,4), "val_loss": round(vl,4)})

        if va > best_val_acc:
            best_val_acc = va
            save_checkpoint(model, class_to_idx, epoch+1, va, CONFIG["model_save_path"])
            print(f"  ⭐ Best! (Val Acc: {va:.4f})")

        if (epoch+1) % 10 == 0:
            save_checkpoint(model, class_to_idx, epoch+1, va, CONFIG["checkpoint_dir"] / f"epoch_{epoch+1:03d}.pth")

        if early_stopping(va):
            print(f"  → {epoch+1} epoch에서 종료"); break

    history["best_val_acc"] = best_val_acc
    with open(CONFIG["history_path"], "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # Test
    print(f"\n[Test 평가]")
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device, weights_only=True))
    test_loader = DataLoader(AlbumentationsDataset(test_dataset, val_transform),
                             batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    ta, tl, tpc, tc = evaluate(model, test_loader, device, class_names, criterion)
    print_metrics(ta, tpc, class_names, title=f"{variant_name} — Test")

    if CONFIG["tta_transforms"] > 1:
        tta_acc, _, tta_pc, _ = evaluate_with_tta(model, test_dataset, device, class_names, CONFIG["tta_transforms"])
        print_metrics(tta_acc, tta_pc, class_names, title=f"{variant_name} — TTA (x{CONFIG['tta_transforms']})")

    print(f"\n✅ 완료! Best Val: {best_val_acc:.4f} | Test: {ta:.4f}")


if __name__ == "__main__":
    main()
