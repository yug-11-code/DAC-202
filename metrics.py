"""
metrics.py - Segmentation Metrics for Brain Tumor Project
Computes all metrics from a running confusion matrix (memory-efficient).
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


NUM_CLASSES = 4
CLASS_NAMES = {0: "background", 1: "glioma", 2: "meningioma", 3: "pituitary"}


class SegmentationMetrics:
    """Accumulates a confusion matrix across batches, derives all metrics."""

    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.roc_probs = []
        self.roc_labels = []
        self.max_roc_pixels = 100_000

    def reset(self):
        self.cm[:] = 0
        self.roc_probs.clear()
        self.roc_labels.clear()

    def update(self, preds, targets):
        """Update confusion matrix. preds/targets: (B,H,W) int64 tensors."""
        p = preds.cpu().view(-1).numpy()
        t = targets.cpu().view(-1).numpy()
        valid = (t >= 0) & (t < self.num_classes)
        p, t = p[valid], t[valid]
        for true_c in range(self.num_classes):
            for pred_c in range(self.num_classes):
                self.cm[true_c, pred_c] += int(((t == true_c) & (p == pred_c)).sum())

    def update_roc(self, probs, targets, sample_rate=0.01):
        """Collect subsampled pixel probabilities for ROC-AUC."""
        if len(self.roc_probs) * 1000 > self.max_roc_pixels:
            return
        p = probs.cpu().numpy()
        t = targets.cpu().numpy()
        B, C, H, W = p.shape
        p = p.transpose(0, 2, 3, 1).reshape(-1, C)
        t = t.reshape(-1)
        n = len(t)
        k = max(1, int(n * sample_rate))
        idx = np.random.choice(n, size=k, replace=False)
        self.roc_probs.append(p[idx])
        self.roc_labels.append(t[idx])

    def get_metrics(self):
        """Compute all metrics from accumulated confusion matrix."""
        cm = self.cm.astype(np.float64)
        total = cm.sum()
        if total == 0:
            return self._empty_metrics()

        accuracy = np.diag(cm).sum() / total

        per_class = {}
        f1_list, support_list = [], []
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            tn = total - tp - fp - fn
            support = cm[c, :].sum()

            precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp + 1e-8) if (tn + fp) > 0 else 0.0
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8) if (2 * tp + fp + fn) > 0 else 0.0
            iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0

            per_class[c] = {
                "precision": precision, "recall": recall, "f1": f1,
                "specificity": specificity, "dice": dice, "iou": iou,
                "support": int(support),
            }
            f1_list.append(f1)
            support_list.append(support)

        macro_f1 = np.mean(f1_list)
        total_support = sum(support_list)
        weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / (total_support + 1e-8)

        mean_dice = np.mean([per_class[c]["dice"] for c in range(self.num_classes)])
        mean_dice_tumor = np.mean([per_class[c]["dice"] for c in range(1, self.num_classes)])
        mean_iou = np.mean([per_class[c]["iou"] for c in range(self.num_classes)])

        roc_auc = self._compute_roc_auc()

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "mean_dice": mean_dice,
            "mean_dice_tumor": mean_dice_tumor,
            "mean_iou": mean_iou,
            "roc_auc": roc_auc,
            "per_class": per_class,
            "confusion_matrix": self.cm.copy(),
        }

    def _compute_roc_auc(self):
        if not self.roc_probs:
            return None
        try:
            probs = np.concatenate(self.roc_probs, axis=0)
            labels = np.concatenate(self.roc_labels, axis=0)
            present = np.unique(labels)
            if len(present) < 2:
                return None
            return roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                                 labels=list(range(self.num_classes)))
        except Exception:
            return None

    def _empty_metrics(self):
        return {
            "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0,
            "mean_dice": 0.0, "mean_dice_tumor": 0.0, "mean_iou": 0.0,
            "roc_auc": None,
            "per_class": {c: {"precision": 0, "recall": 0, "f1": 0,
                              "specificity": 0, "dice": 0, "iou": 0, "support": 0}
                          for c in range(self.num_classes)},
            "confusion_matrix": self.cm.copy(),
        }


def print_metrics(metrics, class_names=CLASS_NAMES):
    """Print a formatted metrics report."""
    print(f"  Pixel Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
    print(f"  Mean Dice      : {metrics['mean_dice']:.4f}")
    print(f"  Mean Dice (tumor): {metrics['mean_dice_tumor']:.4f}")
    print(f"  Mean IoU       : {metrics['mean_iou']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC (macro): {metrics['roc_auc']:.4f}")
    print(f"\n  {'Class':12s} | {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Spec':>6s} {'Dice':>6s} {'IoU':>6s} | {'Support':>10s}")
    print(f"  {'-'*12}-+-{'-'*42}-+-{'-'*10}")
    for c in range(len(class_names)):
        m = metrics["per_class"][c]
        print(f"  {class_names[c]:12s} | {m['precision']:6.4f} {m['recall']:6.4f} {m['f1']:6.4f} "
              f"{m['specificity']:6.4f} {m['dice']:6.4f} {m['iou']:6.4f} | {m['support']:>10,}")
