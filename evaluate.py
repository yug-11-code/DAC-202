"""
evaluate.py - Advanced Evaluation Metrics for Brain Tumor Segmentation
Brain Tumor Segmentation - BRISC 2025 Dataset

Computes research-grade metrics on saved model checkpoints:
  - Dice Coefficient (per-class + mean)
  - IoU / Jaccard Index
  - Hausdorff Distance (HD)
  - 95th Percentile Hausdorff Distance (HD95)
  - Average Surface Distance (ASD)
  - Precision, Recall, F1, Specificity
  - Volume Similarity

Evaluates all available trained models and produces a unified report.

Usage:
    python Code/evaluate.py                    # evaluate all available models
    python Code/evaluate.py --model focal_dice # evaluate single model
    python Code/evaluate.py --quick            # small subset for testing
"""

import os, sys, json, argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (
    discover_pairs, BRISCDataset, get_val_transforms,
    DATASET_ROOT, NUM_CLASSES, CLASS_NAMES, IMG_SIZE
)
from model import get_model
from metrics import SegmentationMetrics, print_metrics

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

MODEL_REGISTRY = {
    "focal_dice": {
        "dir": os.path.join(OUTPUT_DIR, "baseline_focal_dice"),
        "ckpt": "best_model.pth",
        "label": "Focal+Dice",
        "type": "single",
    },
    "weighted_ce": {
        "dir": os.path.join(OUTPUT_DIR, "baseline_ce"),
        "ckpt": "best_model.pth",
        "label": "Weighted CE",
        "type": "single",
    },
    "multitask": {
        "dir": os.path.join(OUTPUT_DIR, "multitask"),
        "ckpt": "best_model_multitask.pth",
        "label": "Multi-Task",
        "type": "multitask",
    },
}


def _surface_distances(pred_binary, gt_binary, spacing=(1.0, 1.0)):
    """Compute surface distances between binary prediction and ground truth.

    Uses distance transforms to find the set of surface voxels and compute
    distances between them. This is the standard approach used in BraTS.

    Args:
        pred_binary: (H, W) bool array
        gt_binary:   (H, W) bool array
        spacing:     pixel spacing (row, col)

    Returns:
        (distances_pred_to_gt, distances_gt_to_pred) or (None, None) if empty
    """
    pred_b = pred_binary.astype(bool)
    gt_b = gt_binary.astype(bool)

    if not pred_b.any() and not gt_b.any():
        return np.array([0.0]), np.array([0.0])
    if not pred_b.any() or not gt_b.any():
        return None, None

    from scipy.ndimage import binary_erosion
    struct = np.ones((3, 3), dtype=bool)

    pred_surface = pred_b ^ binary_erosion(pred_b, structure=struct, border_value=0)
    gt_surface = gt_b ^ binary_erosion(gt_b, structure=struct, border_value=0)

    if not pred_surface.any():
        pred_surface = pred_b
    if not gt_surface.any():
        gt_surface = gt_b

    dt_gt = distance_transform_edt(~gt_b, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_b, sampling=spacing)

    dist_pred_to_gt = dt_gt[pred_surface]
    dist_gt_to_pred = dt_pred[gt_surface]

    return dist_pred_to_gt, dist_gt_to_pred


def hausdorff_distance(pred_binary, gt_binary, spacing=(1.0, 1.0)):
    """Hausdorff Distance: max of directed Hausdorff distances."""
    d1, d2 = _surface_distances(pred_binary, gt_binary, spacing)
    if d1 is None:
        return float("nan")
    return max(d1.max(), d2.max())


def hausdorff_distance_95(pred_binary, gt_binary, spacing=(1.0, 1.0)):
    """95th percentile Hausdorff Distance (HD95).
    More robust than HD — ignores the worst 5% of outliers.
    This is the standard metric used in BraTS challenges."""
    d1, d2 = _surface_distances(pred_binary, gt_binary, spacing)
    if d1 is None:
        return float("nan")
    all_dists = np.concatenate([d1, d2])
    return np.percentile(all_dists, 95)


def average_surface_distance(pred_binary, gt_binary, spacing=(1.0, 1.0)):
    """Average Surface Distance (ASD): mean of all surface distances."""
    d1, d2 = _surface_distances(pred_binary, gt_binary, spacing)
    if d1 is None:
        return float("nan")
    all_dists = np.concatenate([d1, d2])
    return all_dists.mean()


def volume_similarity(pred_binary, gt_binary):
    """Volume Similarity: 1 - |V_pred - V_gt| / (V_pred + V_gt).
    Measures how similar the predicted and GT volumes are."""
    vp = pred_binary.sum()
    vg = gt_binary.sum()
    if vp + vg == 0:
        return 1.0
    return 1.0 - abs(vp - vg) / (vp + vg)


def evaluate_image(pred, gt, num_classes=NUM_CLASSES):
    """Compute all metrics for a single image.

    Args:
        pred: (H, W) int array, values in {0..3}
        gt:   (H, W) int array, values in {0..3}

    Returns:
        dict of per-class metrics
    """
    results = {}
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)

        tp = (pred_c & gt_c).sum()
        fp = (pred_c & ~gt_c).sum()
        fn = (~pred_c & gt_c).sum()

        dice = 2 * tp / (2 * tp + fp + fn + 1e-8) if (2 * tp + fp + fn) > 0 else float("nan")
        iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else float("nan")

        if gt_c.any() or pred_c.any():
            hd = hausdorff_distance(pred_c, gt_c)
            hd95 = hausdorff_distance_95(pred_c, gt_c)
            asd = average_surface_distance(pred_c, gt_c)
            vs = volume_similarity(pred_c, gt_c)
        else:
            hd = hd95 = asd = float("nan")
            vs = float("nan")

        results[c] = {
            "dice": dice, "iou": iou, "precision": precision, "recall": recall,
            "hd": hd, "hd95": hd95, "asd": asd, "volume_similarity": vs,
        }

    return results


def load_model(model_info, device):
    """Load a trained model checkpoint."""
    ckpt_path = os.path.join(model_info["dir"], model_info["ckpt"])
    if not os.path.exists(ckpt_path):
        return None

    if model_info["type"] == "multitask":
        from model_multitask import get_multitask_model
        model = get_multitask_model(device)
    else:
        model = get_model(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    dice = ckpt.get("mean_dice", 0)
    print(f"  Loaded from epoch {epoch} (train Dice={dice:.4f})")
    return model


def evaluate_model(model, test_loader, device, model_type="single"):
    """Run full evaluation on test set.

    Returns:
        dict with per-class aggregated metrics + overall metrics
    """
    all_image_metrics = []

    for images, masks in tqdm(test_loader, desc="  Evaluating", leave=False):
        images = images.to(device)

        with torch.no_grad():
            if model_type == "multitask":
                seg_logits, _ = model(images)
            else:
                seg_logits = model(images)
            preds = torch.argmax(seg_logits, dim=1).cpu().numpy()

        masks_np = masks.numpy()

        for i in range(preds.shape[0]):
            img_metrics = evaluate_image(preds[i], masks_np[i])
            all_image_metrics.append(img_metrics)

    aggregated = {}
    for c in range(NUM_CLASSES):
        class_metrics = defaultdict(list)
        for img_m in all_image_metrics:
            for key, val in img_m[c].items():
                if not np.isnan(val):
                    class_metrics[key].append(val)

        aggregated[c] = {}
        for key, vals in class_metrics.items():
            aggregated[c][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "n": len(vals),
            }

    tumor_metrics = {}
    for key in ["dice", "iou", "hd", "hd95", "asd", "volume_similarity"]:
        vals = []
        for c in range(1, NUM_CLASSES):
            if key in aggregated[c]:
                vals.append(aggregated[c][key]["mean"])
        tumor_metrics[key] = float(np.mean(vals)) if vals else float("nan")

    return {
        "per_class": aggregated,
        "tumor_mean": tumor_metrics,
        "n_images": len(all_image_metrics),
    }


def print_eval_report(name, results):
    """Print formatted evaluation report."""
    print(f"\n  {'='*70}")
    print(f"  {name} — Advanced Evaluation ({results['n_images']} images)")
    print(f"  {'='*70}")

    header = (f"  {'Class':12s} | {'Dice':>8s} {'IoU':>8s} {'HD':>8s} "
              f"{'HD95':>8s} {'ASD':>8s} {'VolSim':>8s} | n")
    print(header)
    print(f"  {'-'*12}-+-{'-'*56}-+---")

    for c in range(NUM_CLASSES):
        m = results["per_class"][c]
        vals = []
        for k in ["dice", "iou", "hd", "hd95", "asd", "volume_similarity"]:
            if k in m:
                vals.append(f"{m[k]['mean']:8.4f}")
            else:
                vals.append(f"{'N/A':>8s}")
        n = m.get("dice", {}).get("n", 0)
        print(f"  {CLASS_NAMES[c]:12s} | {' '.join(vals)} | {n}")

    tm = results["tumor_mean"]
    print(f"\n  Tumor Avg  : Dice={tm['dice']:.4f}  IoU={tm['iou']:.4f}  "
          f"HD={tm['hd']:.2f}  HD95={tm['hd95']:.2f}  ASD={tm['asd']:.2f}")


def save_eval_results(name, results, out_dir):
    """Save results to JSON."""
    path = os.path.join(out_dir, f"eval_{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved -> {path}")


def plot_comparison_advanced(all_results, out_dir):
    """Bar charts comparing advanced metrics across architectures."""
    models = list(all_results.keys())
    if len(models) < 2:
        return

    metrics_to_plot = [
        ("Tumor Dice", "dice", False),
        ("Tumor IoU", "iou", False),
        ("Tumor HD95 ↓", "hd95", True),
        ("Tumor ASD ↓", "asd", True),
    ]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 6))

    for ax, (title, key, lower_better) in zip(axes, metrics_to_plot):
        vals = [all_results[m]["tumor_mean"].get(key, 0) for m in models]
        labels = [MODEL_REGISTRY[m]["label"] for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors[:len(models)],
                      alpha=0.85, edgecolor="white", width=0.6)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Advanced Metrics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "advanced_metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison chart -> {path}")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(NUM_CLASSES)
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        dice_vals = []
        for c in range(NUM_CLASSES):
            d = all_results[m]["per_class"][c].get("dice", {}).get("mean", 0)
            dice_vals.append(d)
        bars = ax.bar(x + i*width - 0.4 + width/2, dice_vals, width,
                      label=MODEL_REGISTRY[m]["label"], color=colors[i], alpha=0.85)
        for bar, v in zip(bars, dice_vals):
            if v > 0.001:
                ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_NAMES[c] for c in range(NUM_CLASSES)], fontsize=11)
    ax.set_title("Per-Class Dice Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Dice Coefficient"); ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y"); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(out_dir, "perclass_dice_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class chart -> {path}")


def evaluate_all(model_filter=None, quick=False):
    """Evaluate all (or selected) trained models."""
    out_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  ADVANCED EVALUATION — Brain Tumor Segmentation")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Metrics: Dice, IoU, HD, HD95, ASD, Volume Similarity")

    print("\n--- Loading test data ---")
    all_pairs = discover_pairs(DATASET_ROOT)
    test_pairs = [(i, m) for i, m in all_pairs if "test" in i.lower()]
    if quick:
        import random; random.shuffle(test_pairs); test_pairs = test_pairs[:30]
    print(f"  Test images: {len(test_pairs)}")

    test_ds = BRISCDataset(test_pairs, transform=get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                             num_workers=2, pin_memory=torch.cuda.is_available())

    all_results = {}
    models_to_eval = [model_filter] if model_filter else list(MODEL_REGISTRY.keys())

    for name in models_to_eval:
        info = MODEL_REGISTRY.get(name)
        if info is None:
            print(f"\n  [SKIP] Unknown model: {name}")
            continue

        print(f"\n--- Evaluating: {info['label']} ---")
        ckpt_path = os.path.join(info["dir"], info["ckpt"])
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        model = load_model(info, device)
        if model is None:
            continue

        results = evaluate_model(model, test_loader, device, info["type"])
        print_eval_report(info["label"], results)
        save_eval_results(name, results, out_dir)
        all_results[name] = results

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_results) >= 2:
        print("\n--- Generating comparison plots ---")
        plot_comparison_advanced(all_results, out_dir)

    if all_results:
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        header = f"  {'Model':15s} | {'Dice':>7s} {'IoU':>7s} {'HD95':>7s} {'ASD':>7s}"
        print(header)
        print(f"  {'-'*15}-+-{'-'*33}")
        for name, r in all_results.items():
            tm = r["tumor_mean"]
            print(f"  {MODEL_REGISTRY[name]['label']:15s} | {tm['dice']:7.4f} "
                  f"{tm['iou']:7.4f} {tm['hd95']:7.2f} {tm['asd']:7.2f}")
        print(f"\n  All outputs in: {out_dir}/")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced evaluation")
    parser.add_argument("--model", type=str, default=None,
                        choices=["focal_dice", "weighted_ce", "multitask"],
                        help="Evaluate single model")
    parser.add_argument("--quick", action="store_true",
                        help="Use small subset for testing")
    args = parser.parse_args()
    evaluate_all(model_filter=args.model, quick=args.quick)


