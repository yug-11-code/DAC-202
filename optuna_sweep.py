"""
optuna_sweep.py - Optuna Hyperparameter Optimization
Brain Tumor Segmentation - BRISC 2025 Dataset

Searches over: lr, gamma, unfreeze_epoch, batch_size, weight_decay, dice_loss_weight
Maximizes: mean_dice_tumor (avg Dice of glioma + meningioma + pituitary)

Usage:
    python Code/optuna_sweep.py                     # local quick test (3 trials, 2 epochs)
    python Code/optuna_sweep.py --trials 20 --epochs 10  # full sweep

Kaggle usage:
    import os, sys
    os.environ["DATASET_ROOT"] = "/kaggle/input/.../brisc2025"
    os.environ["OUTPUT_DIR"]   = "/kaggle/working/outputs"
    sys.path.insert(0, "/kaggle/working/project")
    from optuna_sweep import optuna_sweep
    best = optuna_sweep(n_trials=20, epochs_per_trial=10, quick=False)
"""

import os, sys, csv, json, time, random, argparse, gc
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (
    discover_pairs, BRISCDataset, get_train_transforms, get_val_transforms,
    DATASET_ROOT, NUM_CLASSES, IMG_SIZE, CLASS_NAMES
)
from model import get_model, freeze_encoder, unfreeze_encoder
from loss import compute_rmif_weights, get_loss_fn
from metrics import SegmentationMetrics

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits, targets, num_classes=NUM_CLASSES, smooth=1.0):
    """Soft Dice Loss averaged over tumor classes (excluding background)."""
    probs = torch.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    total_dice = 0.0
    for c in range(1, num_classes):
        p = probs[:, c]
        t = targets_oh[:, c]
        intersection = (p * t).sum()
        total_dice += (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
    return 1.0 - total_dice / (num_classes - 1)


_DATA_CACHE = {}


def get_data_pairs(quick=False):
    """Discover and split data. Cached so it's only done once."""
    key = f"quick={quick}"
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    all_pairs = discover_pairs(DATASET_ROOT)
    train_pairs = [(i, m) for i, m in all_pairs if "train" in i.lower()]
    test_pairs = [(i, m) for i, m in all_pairs if "test" in i.lower()]

    if quick:
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:100]
        test_pairs = test_pairs[:30]

    random.shuffle(train_pairs)
    n_val = max(1, int(len(train_pairs) * 0.1))
    val_pairs = train_pairs[:n_val]
    train_pairs = train_pairs[n_val:]

    result = (train_pairs, val_pairs, test_pairs)
    _DATA_CACHE[key] = result
    print(f"  Data: {len(train_pairs)} train, {len(val_pairs)} val")
    return result


def build_loaders(train_pairs, val_pairs, batch_size, num_workers=2):
    """Build DataLoaders with given batch size."""
    train_ds = BRISCDataset(train_pairs, transform=get_train_transforms())
    val_ds = BRISCDataset(val_pairs, transform=get_val_transforms())

    pin = torch.cuda.is_available()
    kw = dict(num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader


def count_class_pixels(loader):
    """Count pixels per class in the training set."""
    counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for _, masks in loader:
        for c in range(NUM_CLASSES):
            counts[c] += (masks == c).sum().item()
    return counts.tolist()


def objective(trial, train_pairs, val_pairs, epochs_per_trial, device):
    """Optuna objective: train for epochs_per_trial, return best tumor Dice."""

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 1.0, 4.0)
    unfreeze_epoch = trial.suggest_int("unfreeze_epoch", 2, 8)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dice_weight = trial.suggest_float("dice_weight", 0.0, 1.0)
    encoder_lr_factor = 0.01

    train_loader, val_loader = build_loaders(train_pairs, val_pairs, batch_size)

    class_counts = count_class_pixels(train_loader)
    rmif_weights = compute_rmif_weights(class_counts, num_classes=NUM_CLASSES, device=device)

    model = get_model(device)

    focal_fn = get_loss_fn(rmif_weights=rmif_weights, gamma=gamma)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_tumor_dice = 0.0

    for epoch in range(epochs_per_trial):
        if epoch == unfreeze_epoch:
            unfreeze_encoder(model)
            optimizer = torch.optim.Adam([
                {"params": model.encoder.parameters(), "lr": lr * encoder_lr_factor},
                {"params": model.decoder.parameters(), "lr": lr},
                {"params": model.segmentation_head.parameters(), "lr": lr},
            ], weight_decay=weight_decay)

        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = focal_fn(logits, masks)
                if dice_weight > 0:
                    loss = loss + dice_weight * dice_loss(logits, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(loss):
                raise optuna.exceptions.TrialPruned()

        model.eval()
        metrics = SegmentationMetrics()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                metrics.update(preds, masks)

        val_metrics = metrics.get_metrics()
        tumor_dice = val_metrics["mean_dice_tumor"]
        best_tumor_dice = max(best_tumor_dice, tumor_dice)

        trial.report(tumor_dice, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    del model, optimizer, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return best_tumor_dice


def optuna_sweep(n_trials=20, epochs_per_trial=10, quick=False):
    """Run Optuna hyperparameter sweep.

    Args:
        n_trials: number of Optuna trials
        epochs_per_trial: training epochs per trial
        quick: if True, use only 100 train samples for fast iteration

    Returns:
        dict of best hyperparameters
    """
    out_dir = os.path.join(OUTPUT_DIR, "optuna")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print("=" * 65)
    print("  OPTUNA HYPERPARAMETER SWEEP")
    print("=" * 65)
    print(f"  Device         : {device}")
    if device.type == "cuda":
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  Trials         : {n_trials}")
    print(f"  Epochs/trial   : {epochs_per_trial}")
    print(f"  Quick mode     : {quick}")
    print(f"  Metric         : mean_dice_tumor (maximize)")
    print(f"  Output dir     : {out_dir}")

    print("\n--- Loading data ---")
    train_pairs, val_pairs, _ = get_data_pairs(quick=quick)

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="brain_tumor_sweep",
    )

    print("\n--- Starting optimization ---\n")

    def wrapped_objective(trial):
        return objective(trial, train_pairs, val_pairs, epochs_per_trial, device)

    study.optimize(
        wrapped_objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print("\n" + "=" * 65)
    print("  SWEEP COMPLETE")
    print("=" * 65)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    print(f"\n  Top {min(5, len(completed))} trials:")
    print(f"  {'#':<5s} {'Tumor Dice':<12s} {'LR':<10s} {'Gamma':<7s} "
          f"{'Unfreeze':<9s} {'BS':<5s} {'WD':<10s} {'Dice W':<8s}")
    print(f"  {'-'*65}")
    for t in completed[:5]:
        p = t.params
        print(f"  {t.number:<5d} {t.value:<12.4f} {p['lr']:<10.6f} "
              f"{p['gamma']:<7.2f} {p['unfreeze_epoch']:<9d} "
              f"{p['batch_size']:<5d} {p['weight_decay']:<10.6f} "
              f"{p.get('dice_weight', 0.5):<8.3f}")

    best = study.best_params
    best["best_tumor_dice"] = study.best_value
    best["n_trials"] = n_trials
    best["epochs_per_trial"] = epochs_per_trial

    params_path = os.path.join(out_dir, "optuna_best_params.json")
    with open(params_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n  Best params saved -> {params_path}")
    print(f"  Best tumor Dice  : {study.best_value:.4f}")
    print(f"  Best params      : {best}")

    csv_path = os.path.join(out_dir, "optuna_results.csv")
    fields = ["trial_number", "value", "state", "lr", "gamma",
              "unfreeze_epoch", "batch_size", "weight_decay", "dice_weight"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in study.trials:
            row = {
                "trial_number": t.number,
                "value": f"{t.value:.4f}" if t.value is not None else "pruned",
                "state": t.state.name,
            }
            for param in ["lr", "gamma", "unfreeze_epoch", "batch_size",
                          "weight_decay", "dice_weight"]:
                row[param] = t.params.get(param, "")
            writer.writerow(row)
    print(f"  All trials saved -> {csv_path}")

    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )

        fig = plot_optimization_history(study)
        fig.figure.savefig(os.path.join(out_dir, "optuna_history.png"),
                           dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  History plot saved -> {out_dir}/optuna_history.png")

        if len(completed) >= 3:
            fig = plot_param_importances(study)
            fig.figure.savefig(os.path.join(out_dir, "optuna_importances.png"),
                               dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"  Importances plot saved -> {out_dir}/optuna_importances.png")

    except Exception as e:
        print(f"  [NOTE] Visualization skipped: {e}")

    print("\n" + "=" * 65)
    print(f"  All outputs in: {out_dir}/")
    print("=" * 65)

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials (default: 3 for local test)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs per trial (default: 2 for local test)")
    parser.add_argument("--quick", action="store_true",
                        help="Use small data subset for fast iteration")
    args = parser.parse_args()
    optuna_sweep(n_trials=args.trials, epochs_per_trial=args.epochs,
                 quick=args.quick)


