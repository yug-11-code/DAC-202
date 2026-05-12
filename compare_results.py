"""
compare_results.py - Compare All Training Experiments
Brain Tumor Segmentation - BRISC 2025 Dataset

Compares results from all available experiments:
  - Focal+Dice (train.py)
  - Weighted CE (train_ce.py)
  - Multi-Task (train_multitask.py)

Usage:
    python Code/compare_results.py

Generates:
    outputs/comparison/comparison_table.txt
    outputs/comparison/comparison_curves.png
    outputs/comparison/comparison_bar_chart.png
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
OUT_DIR = os.path.join(OUTPUT_DIR, "comparison")

CLASS_NAMES = {0: "background", 1: "glioma", 2: "meningioma", 3: "pituitary"}

EXPERIMENTS = {
    "focal_dice": {
        "dir": os.path.join(OUTPUT_DIR, "baseline_focal_dice"),
        "label": "Focal+Dice",
        "color": "#2196F3",
        "marker": "o",
    },
    "weighted_ce": {
        "dir": os.path.join(OUTPUT_DIR, "baseline_ce"),
        "label": "Weighted CE",
        "color": "#FF5722",
        "marker": "s",
    },
    "multitask": {
        "dir": os.path.join(OUTPUT_DIR, "multitask"),
        "label": "Multi-Task",
        "color": "#4CAF50",
        "marker": "^",
    },
}


def load_results(exp_dir):
    path = os.path.join(exp_dir, "test_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_training_log(exp_dir):
    path = os.path.join(exp_dir, "training_log.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def print_comparison_table(loaded):
    """Print side-by-side comparison of all available experiments."""
    names = list(loaded.keys())
    metrics = [
        ("Pixel Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("Mean Dice (all)", "mean_dice"),
        ("Mean Dice (tumor)", "mean_dice_tumor"),
        ("Mean IoU", "mean_iou"),
        ("ROC-AUC", "roc_auc"),
    ]

    sep = "=" * (30 + 15 * len(names))
    lines = [sep, "  COMPARISON TABLE", sep]

    header = f"  {'Metric':<25s}"
    for n in names:
        header += f" | {EXPERIMENTS[n]['label']:>12s}"
    header += " | Winner"
    lines.append(header)
    lines.append(f"  {'-'*25}" + ("-+-" + "-"*12) * len(names) + "-+---------")

    for label, key in metrics:
        row = f"  {label:<25s}"
        vals = {}
        for n in names:
            v = loaded[n].get(key)
            vals[n] = v
            row += f" | {v:>12.4f}" if v is not None else f" | {'N/A':>12s}"

        valid = {k: v for k, v in vals.items() if v is not None}
        if valid:
            winner = max(valid, key=valid.get)
            row += f" | {EXPERIMENTS[winner]['label']}"
        lines.append(row)

    lines.append("")
    header2 = f"  {'Per-Class Dice':<25s}"
    for n in names:
        header2 += f" | {EXPERIMENTS[n]['label']:>12s}"
    header2 += " | Winner"
    lines.append(header2)
    lines.append(f"  {'-'*25}" + ("-+-" + "-"*12) * len(names) + "-+---------")

    for c in range(4):
        cname = CLASS_NAMES[c]
        row = f"  {cname:<25s}"
        vals = {}
        for n in names:
            d = loaded[n].get("per_class", {}).get(cname, {}).get("dice", 0)
            vals[n] = d
            row += f" | {d:>12.4f}"
        winner = max(vals, key=vals.get)
        row += f" | {EXPERIMENTS[winner]['label']}"
        lines.append(row)

    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    return text


def plot_training_curves(logs, output_path):
    """Overlay training curves from all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_specs = [
        (axes[0, 0], "val_loss", "Validation Loss", "Loss"),
        (axes[0, 1], "mean_dice", "Mean Dice Score", "Dice"),
        (axes[1, 0], "tumor_dice", "Tumor Dice (classes 1-3)", "Tumor Dice"),
        (axes[1, 1], "macro_f1", "Macro F1 Score", "Macro F1"),
    ]

    for ax, col, title, ylabel in plot_specs:
        for name, log in logs.items():
            if log is not None and col in log.columns:
                exp = EXPERIMENTS[name]
                ax.plot(log["epoch"], log[col], color=exp["color"],
                        marker=exp["marker"], markersize=3,
                        label=exp["label"], alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Training Comparison — All Architectures",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves saved -> {output_path}")


def plot_bar_chart(loaded, output_path):
    """Grouped bar chart comparing test metrics."""
    names = list(loaded.keys())
    metric_keys = [
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Mean Dice", "mean_dice"),
        ("Tumor Dice", "mean_dice_tumor"),
        ("Mean IoU", "mean_iou"),
    ]
    for c in range(1, 4):
        metric_keys.append((f"{CLASS_NAMES[c].capitalize()} Dice", f"pcd_{c}"))

    labels = [m[0] for m in metric_keys]
    x = np.arange(len(labels))
    width = 0.8 / len(names)
    colors = [EXPERIMENTS[n]["color"] for n in names]

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, name in enumerate(names):
        vals = []
        for _, key in metric_keys:
            if key.startswith("pcd_"):
                c = int(key.split("_")[1])
                cname = CLASS_NAMES[c]
                vals.append(loaded[name].get("per_class", {}).get(cname, {}).get("dice", 0))
            else:
                vals.append(loaded[name].get(key, 0) or 0)

        bars = ax.bar(x + i*width - 0.4 + width/2, vals, width,
                      label=EXPERIMENTS[name]["label"],
                      color=colors[i], alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            if h > 0.001:
                ax.text(bar.get_x()+bar.get_width()/2., h+0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test Metrics — All Architectures", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.2, axis="y"); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar chart saved -> {output_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n--- Loading results ---")
    loaded = {}
    logs = {}
    for name, info in EXPERIMENTS.items():
        r = load_results(info["dir"])
        if r is not None:
            loaded[name] = r
            logs[name] = load_training_log(info["dir"])
            print(f"  {info['label']:15s}: FOUND")
        else:
            print(f"  {info['label']:15s}: not found (skipping)")

    if len(loaded) < 2:
        print(f"\n  Need at least 2 experiments to compare. Found: {len(loaded)}")
        print("  Run training scripts first:")
        print("    python Code/train.py")
        print("    python Code/train_ce.py")
        print("    python Code/train_multitask.py")
        return

    print()
    table = print_comparison_table(loaded)
    with open(os.path.join(OUT_DIR, "comparison_table.txt"), "w") as f:
        f.write(table)

    print("\n--- Generating plots ---")
    plot_training_curves(logs, os.path.join(OUT_DIR, "comparison_curves.png"))
    plot_bar_chart(loaded, os.path.join(OUT_DIR, "comparison_bar_chart.png"))

    print(f"\n{'='*65}")
    print(f"  All comparison outputs in: {OUT_DIR}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
