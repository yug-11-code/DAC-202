"""
debug_train.py - Lightweight Debug Training Run
Brain Tumor Segmentation - BRISC 2025 Dataset

Purpose:
    Verify the full pipeline (dataset -> model -> loss -> metrics -> viz)
    works correctly BEFORE committing to full-scale training or Optuna tuning.

What this script checks:
    1. Tensor shapes, dtypes, and value ranges
    2. Model forward/backward pass stability
    3. Loss decreases over epochs (model is learning)
    4. Dice and IoU metrics behave correctly
    5. Overfitting test on 5 images (memorization check)
    6. Safety: NaN detection, dead prediction detection, gradient checks
    7. Visualization: input / GT mask / predicted mask after each epoch

Imports from existing project files:
    dataset.py -> discover_pairs, BRISCDataset, get_val_transforms,
                  DATASET_ROOT, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES,
                  CLASS_COLORS_RGB, NUM_CLASSES
    model.py   -> get_model, predict_mask, freeze_encoder, unfreeze_encoder
    loss.py    -> compute_rmif_weights, get_loss_fn
"""

import os
import sys
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    discover_pairs, BRISCDataset, get_val_transforms,
    DATASET_ROOT, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES, CLASS_COLORS_RGB, NUM_CLASSES, IMG_SIZE
)
from model import get_model, predict_mask, freeze_encoder, unfreeze_encoder
from loss import compute_rmif_weights, get_loss_fn


DEBUG_CFG = {
    "num_train":     40,
    "num_val":       10,
    "overfit_count": 5,
    "epochs":        5,
    "batch_size":    4,
    "lr":            1e-3,
    "num_workers":   0,
    "seed":          42,
    "output_dir":    "outputs/debug",
}

MAX_LOSS_THRESHOLD = 50.0
MIN_DICE_AFTER_3   = 0.01


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_dice_iou(preds, targets, num_classes=NUM_CLASSES, smooth=1e-6):
    """
    Compute per-class Dice and IoU between predicted and GT masks.

    Args:
        preds:   (B, H, W) int64 - predicted class labels
        targets: (B, H, W) int64 - ground truth class labels
        num_classes: int
        smooth: float, smoothing constant to avoid division by zero

    Returns:
        dice_per_class: dict {class_idx: dice_score}
        iou_per_class:  dict {class_idx: iou_score}
    """
    dice_per_class = {}
    iou_per_class = {}

    for c in range(num_classes):
        pred_c = (preds == c).float()
        true_c = (targets == c).float()

        intersection = (pred_c * true_c).sum()
        union = pred_c.sum() + true_c.sum() - intersection

        dice = (2.0 * intersection + smooth) / (pred_c.sum() + true_c.sum() + smooth)
        iou = (intersection + smooth) / (union + smooth)

        dice_per_class[c] = dice.item()
        iou_per_class[c] = iou.item()

    return dice_per_class, iou_per_class


def check_tensors(images, masks, batch_idx=0):
    """
    Print and verify tensor shapes, dtypes, and value ranges.
    Called once at the start to catch pipeline bugs early.

    Why each check matters:
        - Shape mismatch -> model crash or silent wrong results
        - Wrong dtype -> loss function fails or gives garbage gradients
        - Unexpected mask values -> loss computes on wrong classes
        - NaN in images -> model immediately produces NaN outputs
    """
    print("\n--- Tensor Sanity Check (batch {}) ---".format(batch_idx))

    print(f"  Image shape : {images.shape}  (expected: (B, 3, 256, 256))")
    print(f"  Image dtype : {images.dtype}  (expected: float32)")
    print(f"  Image min   : {images.min().item():.4f}")
    print(f"  Image max   : {images.max().item():.4f}")
    assert images.ndim == 4, f"Image must be 4D, got {images.ndim}D"
    assert images.shape[1] == 3, f"Image must have 3 channels, got {images.shape[1]}"
    assert images.dtype == torch.float32, f"Image dtype must be float32, got {images.dtype}"
    assert not torch.isnan(images).any(), "NaN detected in image tensor!"

    print(f"  Mask shape  : {masks.shape}  (expected: (B, 256, 256))")
    print(f"  Mask dtype  : {masks.dtype}  (expected: int64)")
    unique_vals = torch.unique(masks).tolist()
    print(f"  Mask unique : {unique_vals}  (expected: subset of {{0,1,2,3}})")
    assert masks.ndim == 3, f"Mask must be 3D (B,H,W), got {masks.ndim}D"
    assert masks.dtype == torch.int64, f"Mask dtype must be int64, got {masks.dtype}"
    valid_classes = {0, 1, 2, 3}
    assert set(int(v) for v in unique_vals) <= valid_classes, \
        f"Mask has invalid values: {unique_vals}"
    assert not torch.isnan(masks.float()).any(), "NaN detected in mask tensor!"

    print("  [OK] All tensor checks passed!")


def check_safety(loss_val, logits, model, epoch, batch_idx):
    """
    Detect training pathologies early:
        - Exploding loss: training has diverged, lr too high or data bug
        - NaN loss: numerical instability in loss computation
        - NaN in logits: model weights have exploded
        - Dead predictions: model predicts same class everywhere
        - NaN gradients: backward pass is broken
    """
    issues = []

    if torch.isnan(loss_val):
        issues.append("[CRITICAL] Loss is NaN!")
    elif torch.isinf(loss_val):
        issues.append("[CRITICAL] Loss is Inf!")
    elif loss_val.item() > MAX_LOSS_THRESHOLD:
        issues.append(f"[WARNING] Loss = {loss_val.item():.2f} > {MAX_LOSS_THRESHOLD} (exploding?)")

    if torch.isnan(logits).any():
        issues.append("[CRITICAL] NaN detected in model logits!")
    if logits.abs().max() > 1000:
        issues.append(f"[WARNING] Logit magnitude = {logits.abs().max().item():.1f} (very large)")

    preds = torch.argmax(logits, dim=1)
    unique_preds = torch.unique(preds)
    if len(unique_preds) == 1:
        issues.append(f"[WARNING] Dead predictions: all pixels = class {unique_preds[0].item()}")

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"[CRITICAL] NaN gradient in: {name}")
                break

    if issues:
        print(f"\n  ** SAFETY ISSUES at epoch {epoch+1}, batch {batch_idx} **")
        for issue in issues:
            print(f"    {issue}")
        if any("[CRITICAL]" in i for i in issues):
            print("    Stopping training due to critical issue!")
            return False
    return True


def count_class_pixels(loader, num_classes=NUM_CLASSES):
    """
    Count total pixels per class across the entire DataLoader.
    Used to compute RMIF weights from the actual training data.
    """
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, masks in loader:
        for c in range(num_classes):
            counts[c] += (masks == c).sum().item()
    return counts.tolist()


def visualize_predictions(model, loader, device, epoch, output_dir, num_samples=3):
    """
    Save a figure showing input channels, GT mask, and predicted mask
    for a few samples. Called after each epoch to visually track learning.

    Why this matters:
        - Numeric metrics can hide subtle issues
        - Visual inspection reveals spatial prediction quality
        - You can spot if the model always predicts background
    """
    model.eval()
    images, masks = next(iter(loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples]

    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu()

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images_denorm = images.cpu() * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]

    col_titles = ["Ch1: Original", "Ch2: CLAHE", "Ch3: Edges",
                  "GT Mask", "Predicted Mask"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(num_samples):
        for ch in range(3):
            axes[i][ch].imshow(images_denorm[i, ch].numpy(), cmap="gray")
            axes[i][ch].axis("off")

        gt_colored = _colorize(masks[i].numpy())
        axes[i][3].imshow(gt_colored)
        axes[i][3].axis("off")

        pred_colored = _colorize(preds[i].numpy())
        axes[i][4].imshow(pred_colored)
        axes[i][4].axis("off")

    patches = [mpatches.Patch(color=np.array(CLASS_COLORS_RGB[c]) / 255.0,
                              label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Debug Training - Epoch {epoch + 1}", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"debug_epoch_{epoch + 1:02d}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"  Visualization saved -> {out_path}")
    model.train()


def _colorize(mask_np):
    """Convert class-label mask (H,W) to RGB (H,W,3)."""
    colored = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for c, rgb in CLASS_COLORS_RGB.items():
        colored[mask_np == c] = rgb
    return colored


def debug_train(overfit_mode=False):
    """
    Run a lightweight debug training session.

    Args:
        overfit_mode: if True, train on only 5 images to test memorization.
            The model should reach near-perfect Dice on those 5 images.
            If it cannot, something is fundamentally broken:
                - loss function is wrong
                - model capacity is insufficient
                - data pipeline is corrupting inputs
                - optimizer is not working
    """
    cfg = DEBUG_CFG.copy()
    if overfit_mode:
        cfg["num_train"] = cfg["overfit_count"]
        cfg["num_val"] = cfg["overfit_count"]
        cfg["epochs"] = 10
        cfg["output_dir"] = "outputs/debug_overfit"
        print("=" * 60)
        print("  OVERFITTING TEST MODE")
        print(f"  Training on {cfg['overfit_count']} images for {cfg['epochs']} epochs")
        print("  Expected: loss -> ~0, Dice -> ~1.0")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  DEBUG TRAINING RUN")
        print(f"  {cfg['num_train']} train, {cfg['num_val']} val, {cfg['epochs']} epochs")
        print("=" * 60)

    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    print("\n--- Loading data (small subset) ---")
    all_pairs = discover_pairs(DATASET_ROOT)

    random.shuffle(all_pairs)
    train_pairs = all_pairs[:cfg["num_train"]]
    val_pairs = all_pairs[cfg["num_train"]:cfg["num_train"] + cfg["num_val"]]

    if overfit_mode:
        val_pairs = train_pairs[:cfg["overfit_count"]]

    train_dataset = BRISCDataset(train_pairs, transform=get_val_transforms())
    val_dataset = BRISCDataset(val_pairs, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples  : {len(val_dataset)}")
    print(f"  Batch size   : {cfg['batch_size']}")

    print("\n--- Initial tensor sanity check ---")
    first_images, first_masks = next(iter(train_loader))
    check_tensors(first_images, first_masks, batch_idx=0)

    print("\n--- Computing RMIF class weights ---")
    class_counts = count_class_pixels(train_loader)
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:12s}: {class_counts[c]:>12,} pixels")

    rmif_weights = compute_rmif_weights(class_counts, num_classes=NUM_CLASSES, device=device)
    print(f"  RMIF weights: {rmif_weights}")
    print(f"  Weights sum : {rmif_weights.sum().item():.4f}")

    print("\n--- Building model ---")
    model = get_model(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen (encoder): {total_params - trainable_params:,}")

    loss_fn = get_loss_fn(rmif_weights=rmif_weights, gamma=2.0)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
    )

    print("\n--- Forward pass check ---")
    test_imgs = first_images[:2].to(device)
    test_logits = model(test_imgs)
    print(f"  Input shape  : {test_imgs.shape}")
    print(f"  Output shape : {test_logits.shape}")
    print(f"  Output dtype : {test_logits.dtype}")
    print(f"  Output range : [{test_logits.min().item():.4f}, {test_logits.max().item():.4f}]")
    assert test_logits.shape == (2, NUM_CLASSES, IMG_SIZE, IMG_SIZE), \
        f"Shape mismatch! Got {test_logits.shape}"
    print("  [OK] Forward pass shape verified!")

    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()

        model.train()
        train_losses = []
        batch_count = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()

            safe = check_safety(loss, logits, model, epoch, batch_idx)
            if not safe:
                print("\n[ABORT] Training stopped due to critical safety issue.")
                return history

            optimizer.step()
            train_losses.append(loss.item())
            batch_count += 1

            if batch_count <= 3 or batch_count % 5 == 0:
                preds_quick = torch.argmax(logits.detach(), dim=1)
                unique_preds = torch.unique(preds_quick).tolist()
                print(f"  Epoch {epoch+1}/{cfg['epochs']} | "
                      f"Batch {batch_count:3d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Pred classes: {unique_preds}")

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_losses = []
        all_dice = defaultdict(list)
        all_iou = defaultdict(list)

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                vloss = loss_fn(logits, masks)
                val_losses.append(vloss.item())

                preds = torch.argmax(logits, dim=1)
                dice, iou = compute_dice_iou(preds, masks)
                for c in range(NUM_CLASSES):
                    all_dice[c].append(dice[c])
                    all_iou[c].append(iou[c])

        avg_val_loss = np.mean(val_losses)
        mean_dice = {c: np.mean(all_dice[c]) for c in range(NUM_CLASSES)}
        mean_iou = {c: np.mean(all_iou[c]) for c in range(NUM_CLASSES)}
        avg_dice = np.mean(list(mean_dice.values()))
        avg_iou = np.mean(list(mean_iou.values()))

        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_dice)
        history["val_iou"].append(avg_iou)

        elapsed = time.time() - epoch_start

        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/{cfg['epochs']} Summary  ({elapsed:.1f}s)")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {avg_val_loss:.4f}")
        print(f"  Mean Dice  : {avg_dice:.4f}")
        print(f"  Mean IoU   : {avg_iou:.4f}")

        print(f"  Per-class Dice:")
        for c in range(NUM_CLASSES):
            print(f"    {CLASS_NAMES[c]:12s}: Dice={mean_dice[c]:.4f}  IoU={mean_iou[c]:.4f}")

        if device.type == "cuda":
            mem_alloc = torch.cuda.memory_allocated(0) / 1e6
            mem_reserved = torch.cuda.memory_reserved(0) / 1e6
            print(f"  GPU Memory : {mem_alloc:.0f} MB allocated, {mem_reserved:.0f} MB reserved")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR         : {current_lr:.6f}")

        if avg_dice < MIN_DICE_AFTER_3 and epoch >= 2:
            print(f"  [WARNING] Dice is still < {MIN_DICE_AFTER_3} after {epoch+1} epochs!")
            print(f"            Model may be predicting all-background.")

        print(f"{'='*60}")

        visualize_predictions(model, val_loader, device, epoch, cfg["output_dir"])

    print("\n" + "=" * 60)
    print("  DEBUG TRAINING COMPLETE")
    print("-" * 60)

    loss_decreased = history["train_loss"][-1] < history["train_loss"][0]
    dice_increased = history["val_dice"][-1] > history["val_dice"][0]

    print(f"  Initial train loss : {history['train_loss'][0]:.4f}")
    print(f"  Final train loss   : {history['train_loss'][-1]:.4f}")
    print(f"  Loss decreased     : {'YES' if loss_decreased else 'NO'}")
    print(f"  Initial val Dice   : {history['val_dice'][0]:.4f}")
    print(f"  Final val Dice     : {history['val_dice'][-1]:.4f}")
    print(f"  Dice improved      : {'YES' if dice_increased else 'NO'}")

    if overfit_mode:
        final_dice = history["val_dice"][-1]
        if final_dice > 0.5:
            print(f"\n  [OK] Overfitting test PASSED (Dice={final_dice:.4f} > 0.5)")
            print(f"       Model can memorize a tiny dataset -> pipeline is correct.")
        else:
            print(f"\n  [FAIL] Overfitting test FAILED (Dice={final_dice:.4f} < 0.5)")
            print(f"         The model cannot memorize 5 images. Possible causes:")
            print(f"           - Loss function bug")
            print(f"           - Data pipeline corrupting inputs")
            print(f"           - Model architecture issue")
            print(f"           - Learning rate too low/high")
    else:
        if loss_decreased and dice_increased:
            print(f"\n  [OK] Pipeline verification PASSED!")
            print(f"       Loss decreases and Dice improves -> ready for full training.")
        elif loss_decreased:
            print(f"\n  [PARTIAL] Loss decreases but Dice didn't improve much.")
            print(f"            This may be normal for only {cfg['epochs']} debug epochs.")
            print(f"            Try running the overfitting test: debug_train(overfit_mode=True)")
        else:
            print(f"\n  [FAIL] Loss did NOT decrease. Something is wrong:")
            print(f"           - Check learning rate")
            print(f"           - Check loss function")
            print(f"           - Check data pipeline")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve (Debug)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["val_dice"], "g-o", label="Mean Dice")
    ax2.plot(epochs_range, history["val_iou"], "m-o", label="Mean IoU")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Metrics (Debug)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(cfg["output_dir"], "debug_curves.png")
    plt.savefig(curve_path, dpi=100)
    plt.close()
    print(f"\n  Loss/metric curves saved -> {curve_path}")

    print("=" * 60)
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug training for brain tumor segmentation")
    parser.add_argument("--overfit", action="store_true",
                        help="Run overfitting test on 5 images instead of normal debug")
    args = parser.parse_args()

    history = debug_train(overfit_mode=args.overfit)
