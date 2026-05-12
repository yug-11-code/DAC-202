"""
train.py - Full Baseline Training Pipeline
Brain Tumor Segmentation - BRISC 2025 Dataset

Usage:
    python Code/train.py                # full training (50 epochs)
    python Code/train.py --quick        # quick test (3 epochs, 100 images)
"""

import os, sys, csv, json, time, random, argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (
    discover_pairs, BRISCDataset, get_train_transforms, get_val_transforms,
    DATASET_ROOT, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES, CLASS_COLORS_RGB, NUM_CLASSES, IMG_SIZE
)
from model import get_model, freeze_encoder, unfreeze_encoder
from loss import compute_rmif_weights, get_loss_fn
from metrics import SegmentationMetrics, print_metrics

CONFIG = {
    "seed": 42,
    "epochs": 50,
    "batch_size": 16,
    "lr": 1e-3,
    "encoder_lr_factor": 0.01,
    "unfreeze_epoch": 5,
    "weight_decay": 1e-4,
    "cosine_T0": 10,
    "cosine_T_mult": 2,
    "early_stop_patience": 15,
    "grad_clip_norm": 1.0,
    "val_split": 0.1,
    "num_workers": 2,
    "dice_loss_weight": 0.5,
    "output_dir": "outputs/baseline_focal_dice",
    "viz_every_n": 5,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Monitor tumor Dice (higher=better) instead of val_loss.
    Val_loss is dominated by background pixels and can increase even while
    tumor segmentation improves. Tumor Dice directly measures what we care about."""
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, tumor_dice):
        if self.best_score is None or tumor_dice > self.best_score + self.min_delta:
            self.best_score = tumor_dice
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        with open(filepath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames).writeheader()

    def log(self, row):
        with open(self.filepath, "a", newline="") as f:
            csv.DictWriter(f, self.fieldnames).writerow(row)


def prepare_data(cfg, quick=False):
    """Discover pairs, split train/val/test, build dataloaders."""
    all_pairs = discover_pairs(DATASET_ROOT)

    train_pairs = [(i, m) for i, m in all_pairs if "train" in i.lower()]
    test_pairs = [(i, m) for i, m in all_pairs if "test" in i.lower()]
    print(f"  Raw split: {len(train_pairs)} train, {len(test_pairs)} test")

    if quick:
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:100]
        test_pairs = test_pairs[:30]

    random.shuffle(train_pairs)
    n_val = max(1, int(len(train_pairs) * cfg["val_split"]))
    val_pairs = train_pairs[:n_val]
    train_pairs = train_pairs[n_val:]

    print(f"  Final: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

    train_ds = BRISCDataset(train_pairs, transform=get_train_transforms())
    val_ds = BRISCDataset(val_pairs, transform=get_val_transforms())
    test_ds = BRISCDataset(test_pairs, transform=get_val_transforms())

    pin = torch.cuda.is_available()
    kw = dict(num_workers=cfg["num_workers"], pin_memory=pin)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, **kw)

    return train_loader, val_loader, test_loader


def count_class_pixels(loader):
    counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for _, masks in tqdm(loader, desc="  Counting pixels", leave=False):
        for c in range(NUM_CLASSES):
            counts[c] += (masks == c).sum().item()
    return counts.tolist()


def dice_loss(logits, targets, num_classes=NUM_CLASSES, smooth=1.0):
    """Soft Dice Loss averaged over classes (excluding background).
    Focal Loss optimizes per-pixel classification but doesn't directly
    maximize spatial overlap. Dice Loss fills that gap by pushing the model
    to produce contiguous, correctly-shaped tumor regions."""
    probs = torch.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    total_dice = 0.0
    for c in range(1, num_classes):
        p = probs[:, c]
        t = targets_oh[:, c]
        intersection = (p * t).sum()
        total_dice += (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
    return 1.0 - total_dice / (num_classes - 1)


def check_batch_safety(loss_val, logits, model):
    """Returns list of issues (empty = safe)."""
    issues = []
    if torch.isnan(loss_val):
        issues.append("NaN loss")
    if torch.isinf(loss_val):
        issues.append("Inf loss")
    if loss_val.item() > 100:
        issues.append(f"Exploding loss={loss_val.item():.1f}")
    if torch.isnan(logits).any():
        issues.append("NaN in logits")
    preds = torch.argmax(logits, dim=1)
    if len(torch.unique(preds)) == 1:
        issues.append(f"Class collapse: all preds={preds[0,0,0].item()}")
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any():
            issues.append(f"NaN gradient: {n}")
            break
    return issues


def train_one_epoch(model, loader, loss_fn, optimizer, device, cfg, epoch):
    model.train()
    losses = []
    metrics = SegmentationMetrics()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    pbar = tqdm(loader, desc=f"  Train E{epoch+1}", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            focal = loss_fn(logits, masks)
            dl = dice_loss(logits, masks)
            loss = focal + cfg.get("dice_loss_weight", 0.5) * dl

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()

        issues = check_batch_safety(loss, logits, model)
        if issues:
            print(f"\n  [WARNING] {', '.join(issues)}")
            if "NaN loss" in issues or "NaN in logits" in issues:
                print("  [ABORT] Critical issue. Stopping training.")
                return None, None

        losses.append(loss.item())
        preds = torch.argmax(logits.detach(), dim=1)
        metrics.update(preds, masks)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return np.mean(losses), metrics.get_metrics()


@torch.no_grad()
def validate(model, loader, loss_fn, device, collect_roc=False):
    model.eval()
    losses = []
    metrics = SegmentationMetrics()

    for images, masks in tqdm(loader, desc="  Val", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        focal = loss_fn(logits, masks)
        dl = dice_loss(logits, masks)
        loss = focal + 0.5 * dl
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        metrics.update(preds, masks)

        if collect_roc:
            probs = torch.softmax(logits, dim=1)
            metrics.update_roc(probs, masks)

    return np.mean(losses), metrics.get_metrics()


def plot_confusion_matrix(cm, output_path, class_names=CLASS_NAMES):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({cm[i,j]:,})", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([class_names[i] for i in range(len(class_names))], rotation=45, ha="right")
    ax.set_yticklabels([class_names[i] for i in range(len(class_names))])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_curves(history, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["val_dice"], "g-", label="Mean Dice")
    axes[0, 1].plot(epochs, history["val_dice_tumor"], "m-", label="Tumor Dice")
    axes[0, 1].set_title("Dice Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["val_accuracy"], "c-", label="Accuracy")
    axes[1, 0].plot(epochs, history["val_macro_f1"], "y-", label="Macro F1")
    axes[1, 0].set_title("Accuracy & F1")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["lr"], "k-")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    plt.suptitle("Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _colorize(mask_np):
    colored = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for c, rgb in CLASS_COLORS_RGB.items():
        colored[mask_np == c] = rgb
    return colored


def visualize_preds(model, loader, device, output_path, n=4):
    model.eval()
    images, masks = next(iter(loader))
    images = images[:n].to(device)
    masks = masks[:n]
    with torch.no_grad():
        preds = torch.argmax(model(images), dim=1).cpu()

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    imgs = torch.clamp(images.cpu() * std + mean, 0, 1)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]
    titles = ["Ch1: Original", "Ch2: CLAHE", "GT Mask", "Prediction"]
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontweight="bold")
    for i in range(n):
        axes[i][0].imshow(imgs[i, 0].numpy(), cmap="gray"); axes[i][0].axis("off")
        axes[i][1].imshow(imgs[i, 1].numpy(), cmap="gray"); axes[i][1].axis("off")
        axes[i][2].imshow(_colorize(masks[i].numpy())); axes[i][2].axis("off")
        axes[i][3].imshow(_colorize(preds[i].numpy())); axes[i][3].axis("off")
    patches = [mpatches.Patch(color=np.array(CLASS_COLORS_RGB[c]) / 255.0,
                              label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close()
    model.train()


def train(quick=False):
    cfg = CONFIG.copy()
    if quick:
        cfg["epochs"] = 3
        cfg["unfreeze_epoch"] = 2
        cfg["early_stop_patience"] = 99
        cfg["output_dir"] = "outputs/baseline_focal_dice_quick"
    else:
        cfg["output_dir"] = "outputs/baseline_focal_dice"

    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    with open(os.path.join(cfg["output_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  BASELINE TRAINING - RMIF Focal + Dice Loss")
    print("=" * 65)
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs    : {cfg['epochs']}")
    print(f"  Batch     : {cfg['batch_size']}")
    print(f"  Loss      : RMIF Focal (gamma=2.0) + 0.5 * Dice Loss")
    if device.type == "cpu":
        print("  [NOTE] Training on CPU will be slow. Use GPU for full runs.")

    print("\n--- Data ---")
    train_loader, val_loader, test_loader = prepare_data(cfg, quick=quick)

    print("\n--- Class Weights ---")
    class_counts = count_class_pixels(train_loader)
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:12s}: {class_counts[c]:>14,} pixels")
    rmif_weights = compute_rmif_weights(class_counts, num_classes=NUM_CLASSES, device=device)
    print(f"  RMIF weights: {rmif_weights}")

    print("\n--- Model ---")
    model = get_model(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_p:,}")
    print(f"  Trainable       : {train_p:,} (decoder only, encoder frozen)")

    loss_fn = get_loss_fn(rmif_weights=rmif_weights, gamma=2.0)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["cosine_T0"], T_mult=cfg["cosine_T_mult"]
    )
    early_stop = EarlyStopping(patience=cfg["early_stop_patience"])

    log_fields = ["epoch", "train_loss", "val_loss", "accuracy", "macro_f1",
                  "weighted_f1", "mean_dice", "tumor_dice", "mean_iou", "lr"]
    csv_log = CSVLogger(os.path.join(cfg["output_dir"], "training_log.csv"), log_fields)

    print("\n" + "=" * 65)
    print("  TRAINING")
    print("=" * 65)

    history = defaultdict(list)
    best_dice = 0.0
    best_path = os.path.join(cfg["output_dir"], "best_model.pth")

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()

        if epoch == cfg["unfreeze_epoch"]:
            print(f"\n  >>> Epoch {epoch+1}: Unfreezing encoder <<<")
            unfreeze_encoder(model)
            optimizer = torch.optim.Adam([
                {"params": model.encoder.parameters(), "lr": cfg["lr"] * cfg["encoder_lr_factor"]},
                {"params": model.decoder.parameters(), "lr": cfg["lr"]},
                {"params": model.segmentation_head.parameters(), "lr": cfg["lr"]},
            ], weight_decay=cfg["weight_decay"])
            remaining = cfg["epochs"] - epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=min(cfg["cosine_T0"], remaining),
                T_mult=cfg["cosine_T_mult"]
            )
            tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params now: {tp:,}")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, cfg, epoch
        )
        if train_loss is None:
            print("[ABORT] Training stopped.")
            break

        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_metrics["mean_dice"])
        history["val_dice_tumor"].append(val_metrics["mean_dice_tumor"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["lr"].append(current_lr)

        csv_log.log({
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "accuracy": f"{val_metrics['accuracy']:.4f}",
            "macro_f1": f"{val_metrics['macro_f1']:.4f}",
            "weighted_f1": f"{val_metrics['weighted_f1']:.4f}",
            "mean_dice": f"{val_metrics['mean_dice']:.4f}",
            "tumor_dice": f"{val_metrics['mean_dice_tumor']:.4f}",
            "mean_iou": f"{val_metrics['mean_iou']:.4f}",
            "lr": f"{current_lr:.2e}",
        })

        print(f"\n  Epoch {epoch+1}/{cfg['epochs']}  ({elapsed:.0f}s)  "
              f"LR={current_lr:.2e}")
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f"  Dice: {val_metrics['mean_dice']:.4f} (tumor: {val_metrics['mean_dice_tumor']:.4f})  "
              f"Acc: {val_metrics['accuracy']:.4f}  F1: {val_metrics['macro_f1']:.4f}")

        for c in range(NUM_CLASSES):
            d = val_metrics["per_class"][c]["dice"]
            print(f"    {CLASS_NAMES[c]:12s}: Dice={d:.4f}")

        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(0) / 1e6
            print(f"  GPU: {alloc:.0f} MB allocated")

        tumor_dice = val_metrics["mean_dice_tumor"]
        improved = early_stop.step(tumor_dice)
        if val_metrics["mean_dice"] > best_dice:
            best_dice = val_metrics["mean_dice"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "mean_dice": best_dice,
                "config": cfg,
            }, best_path)
            print(f"  ** Best model saved (Dice={best_dice:.4f}) **")

        if (epoch + 1) % cfg["viz_every_n"] == 0 or epoch == 0:
            vpath = os.path.join(cfg["output_dir"], f"preds_epoch_{epoch+1:03d}.png")
            visualize_preds(model, val_loader, device, vpath)
            print(f"  Predictions saved -> {vpath}")

        if early_stop.should_stop:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

        print("-" * 65)

    print("\n--- Saving training curves ---")
    plot_curves(dict(history), os.path.join(cfg["output_dir"], "training_curves.png"))

    print("\n" + "=" * 65)
    print("  TEST SET EVALUATION")
    print("=" * 65)

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best model from epoch {ckpt['epoch']} (Dice={ckpt['mean_dice']:.4f})")

    test_loss, test_metrics = validate(model, test_loader, loss_fn, device, collect_roc=True)
    print(f"\n  Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics)

    cm_path = os.path.join(cfg["output_dir"], "confusion_matrix.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], cm_path)
    print(f"\n  Confusion matrix saved -> {cm_path}")

    pred_path = os.path.join(cfg["output_dir"], "test_predictions.png")
    visualize_preds(model, test_loader, device, pred_path, n=6)
    print(f"  Test predictions saved -> {pred_path}")

    results = {
        "test_loss": float(test_loss),
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "weighted_f1": float(test_metrics["weighted_f1"]),
        "mean_dice": float(test_metrics["mean_dice"]),
        "mean_dice_tumor": float(test_metrics["mean_dice_tumor"]),
        "mean_iou": float(test_metrics["mean_iou"]),
        "roc_auc": float(test_metrics["roc_auc"]) if test_metrics["roc_auc"] else None,
        "per_class": {CLASS_NAMES[c]: {k: float(v) for k, v in m.items()}
                      for c, m in test_metrics["per_class"].items()},
    }
    with open(os.path.join(cfg["output_dir"], "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print(f"  Best Dice: {best_dice:.4f}")
    print(f"  All outputs in: {cfg['output_dir']}/")
    print("=" * 65)

    return history, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline training (Focal+Dice loss)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (3 epochs, 100 images)")
    args = parser.parse_args()
    train(quick=args.quick)

