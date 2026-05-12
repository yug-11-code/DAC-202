"""
train_multitask.py - Dual-Head Training (Segmentation + Classification)
Brain Tumor Segmentation - BRISC 2025 Dataset

Trains DualHeadUNet with combined loss:
    L = seg_weight * FocalLoss(seg) + cls_weight * CE(cls)

Usage:
    python Code/train_multitask.py              # full 50 epochs
    python Code/train_multitask.py --quick      # 3 epochs, 100 images
"""

import os, sys, csv, json, time, random, argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (
    discover_pairs, BRISCDataset, get_train_transforms, get_val_transforms,
    DATASET_ROOT, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES, CLASS_COLORS_RGB, NUM_CLASSES, IMG_SIZE
)
from model_multitask import get_multitask_model, derive_cls_label
from loss import compute_rmif_weights, get_loss_fn
from metrics import SegmentationMetrics, print_metrics

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

CONFIG = {
    "seed": 42, "epochs": 50, "batch_size": 16,
    "lr": 1e-3, "encoder_lr_factor": 0.01, "unfreeze_epoch": 5,
    "weight_decay": 1e-4,
    "cosine_T0": 10, "cosine_T_mult": 2,
    "early_stop_patience": 15, "grad_clip_norm": 1.0,
    "val_split": 0.1, "num_workers": 2,
    "seg_loss_weight": 1.0, "cls_loss_weight": 0.3,
    "dice_loss_weight": 0.5,
    "output_dir": os.path.join(OUTPUT_DIR, "multitask"),
    "viz_every_n": 5,
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_score, self.should_stop = 0, None, False
    def step(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score; self.counter = 0; return True
        self.counter += 1
        if self.counter >= self.patience: self.should_stop = True
        return False


class CSVLogger:
    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        with open(path, "w", newline="") as f: csv.DictWriter(f, fields).writeheader()
    def log(self, row):
        with open(self.path, "a", newline="") as f: csv.DictWriter(f, self.fields).writerow(row)


def dice_loss(logits, targets, num_classes=NUM_CLASSES, smooth=1.0):
    probs = torch.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    total = 0.0
    for c in range(1, num_classes):
        p, t = probs[:, c], targets_oh[:, c]
        total += (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
    return 1.0 - total / (num_classes - 1)


def prepare_data(cfg, quick=False):
    all_pairs = discover_pairs(DATASET_ROOT)
    train_pairs = [(i, m) for i, m in all_pairs if "train" in i.lower()]
    test_pairs = [(i, m) for i, m in all_pairs if "test" in i.lower()]
    if quick:
        random.shuffle(train_pairs); train_pairs = train_pairs[:100]; test_pairs = test_pairs[:30]
    random.shuffle(train_pairs)
    n_val = max(1, int(len(train_pairs) * cfg["val_split"]))
    val_pairs, train_pairs = train_pairs[:n_val], train_pairs[n_val:]
    print(f"  {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
    pin = torch.cuda.is_available()
    kw = dict(num_workers=cfg["num_workers"], pin_memory=pin)
    return (
        DataLoader(BRISCDataset(train_pairs, get_train_transforms()), cfg["batch_size"], shuffle=True, **kw),
        DataLoader(BRISCDataset(val_pairs, get_val_transforms()), cfg["batch_size"], shuffle=False, **kw),
        DataLoader(BRISCDataset(test_pairs, get_val_transforms()), cfg["batch_size"], shuffle=False, **kw),
    )


def count_class_pixels(loader):
    counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for _, masks in tqdm(loader, desc="  Counting pixels", leave=False):
        for c in range(NUM_CLASSES): counts[c] += (masks == c).sum().item()
    return counts.tolist()


def train_one_epoch(model, loader, seg_fn, optimizer, device, cfg, epoch):
    model.train()
    seg_losses, cls_losses, total_losses = [], [], []
    cls_correct, cls_total = 0, 0
    seg_metrics = SegmentationMetrics()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for images, masks in tqdm(loader, desc=f"  Train E{epoch+1}", leave=False):
        images, masks = images.to(device), masks.to(device)
        cls_targets = derive_cls_label(masks)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            seg_logits, cls_logits = model(images)
            s_loss = seg_fn(seg_logits, masks) + cfg["dice_loss_weight"] * dice_loss(seg_logits, masks)
            c_loss = F.cross_entropy(cls_logits, cls_targets)
            loss = cfg["seg_loss_weight"] * s_loss + cfg["cls_loss_weight"] * c_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
        scaler.step(optimizer); scaler.update()

        if torch.isnan(loss): return None, None, None, None

        seg_losses.append(s_loss.item()); cls_losses.append(c_loss.item())
        total_losses.append(loss.item())
        preds = torch.argmax(seg_logits.detach(), dim=1)
        seg_metrics.update(preds, masks)
        cls_correct += (cls_logits.argmax(1) == cls_targets).sum().item()
        cls_total += cls_targets.size(0)

    cls_acc = cls_correct / max(cls_total, 1)
    return np.mean(total_losses), np.mean(seg_losses), np.mean(cls_losses), cls_acc


@torch.no_grad()
def validate(model, loader, seg_fn, device, cfg, collect_roc=False):
    model.eval()
    seg_losses, cls_losses, total_losses = [], [], []
    cls_correct, cls_total = 0, 0
    metrics = SegmentationMetrics()

    for images, masks in tqdm(loader, desc="  Val", leave=False):
        images, masks = images.to(device), masks.to(device)
        cls_targets = derive_cls_label(masks)
        seg_logits, cls_logits = model(images)
        s_loss = seg_fn(seg_logits, masks) + cfg["dice_loss_weight"] * dice_loss(seg_logits, masks)
        c_loss = F.cross_entropy(cls_logits, cls_targets)
        loss = cfg["seg_loss_weight"] * s_loss + cfg["cls_loss_weight"] * c_loss
        seg_losses.append(s_loss.item()); cls_losses.append(c_loss.item())
        total_losses.append(loss.item())
        preds = torch.argmax(seg_logits, dim=1)
        metrics.update(preds, masks)
        if collect_roc:
            metrics.update_roc(torch.softmax(seg_logits, dim=1), masks)
        cls_correct += (cls_logits.argmax(1) == cls_targets).sum().item()
        cls_total += cls_targets.size(0)

    cls_acc = cls_correct / max(cls_total, 1)
    return np.mean(total_losses), np.mean(seg_losses), np.mean(cls_losses), cls_acc, metrics.get_metrics()


def _colorize(m):
    c = np.zeros((*m.shape, 3), dtype=np.uint8)
    for k, rgb in CLASS_COLORS_RGB.items(): c[m == k] = rgb
    return c

def visualize_preds(model, loader, device, path, n=4):
    model.eval(); images, masks = next(iter(loader))
    images = images[:n].to(device); masks = masks[:n]
    with torch.no_grad():
        seg, cls = model(images)
        preds = torch.argmax(seg, dim=1).cpu()
        cls_preds = torch.argmax(cls, dim=1).cpu()
    mean = torch.tensor(IMAGENET_MEAN).view(1,3,1,1)
    std = torch.tensor(IMAGENET_STD).view(1,3,1,1)
    imgs = torch.clamp(images.cpu()*std+mean, 0, 1)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    if n == 1: axes = [axes]
    for ax, t in zip(axes[0], ["Original", "GT Mask", "Pred Mask", "Cls"]):
        ax.set_title(t, fontweight="bold")
    for i in range(n):
        axes[i][0].imshow(imgs[i,0].numpy(), cmap="gray"); axes[i][0].axis("off")
        axes[i][1].imshow(_colorize(masks[i].numpy())); axes[i][1].axis("off")
        axes[i][2].imshow(_colorize(preds[i].numpy())); axes[i][2].axis("off")
        gt_cls = derive_cls_label(masks[i:i+1]).item()
        axes[i][3].text(0.5, 0.5, f"GT: {CLASS_NAMES[gt_cls]}\nPred: {CLASS_NAMES[cls_preds[i].item()]}",
                       ha="center", va="center", fontsize=14, transform=axes[i][3].transAxes)
        axes[i][3].axis("off")
    patches = [mpatches.Patch(color=np.array(CLASS_COLORS_RGB[c])/255., label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight", dpi=100); plt.close()
    model.train()

def plot_curves(h, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ep = range(1, len(h["train_loss"])+1)
    axes[0,0].plot(ep, h["train_loss"], "b-", label="Train"); axes[0,0].plot(ep, h["val_loss"], "r-", label="Val")
    axes[0,0].set_title("Total Loss"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(ep, h["val_dice"], "g-", label="Mean Dice"); axes[0,1].plot(ep, h["val_tumor_dice"], "m-", label="Tumor Dice")
    axes[0,1].set_title("Dice Score"); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
    axes[1,0].plot(ep, h["train_cls_acc"], "b-", label="Train"); axes[1,0].plot(ep, h["val_cls_acc"], "r-", label="Val")
    axes[1,0].set_title("Classification Accuracy"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(ep, h["lr"], "k-"); axes[1,1].set_title("Learning Rate"); axes[1,1].set_yscale("log"); axes[1,1].grid(True, alpha=0.3)
    for ax in axes.flat: ax.set_xlabel("Epoch")
    plt.suptitle("Multi-Task Training Curves", fontsize=14); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm_n[i,j]:.2f}\n({cm[i,j]:,})", ha="center", va="center", fontsize=8,
                    color="white" if cm_n[i,j]>0.5 else "black")
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([CLASS_NAMES[i] for i in range(len(CLASS_NAMES))], rotation=45, ha="right")
    ax.set_yticklabels([CLASS_NAMES[i] for i in range(len(CLASS_NAMES))])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix - Multi-Task")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def train_multitask(quick=False):
    cfg = CONFIG.copy()
    if quick:
        cfg["epochs"] = 3; cfg["unfreeze_epoch"] = 2
        cfg["early_stop_patience"] = 99
        cfg["output_dir"] = os.path.join(OUTPUT_DIR, "multitask_quick")

    set_seed(cfg["seed"]); os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  MULTI-TASK TRAINING (Segmentation + Classification)")
    print("=" * 65)
    print(f"  Device : {device}")
    if device.type == "cuda": print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs : {cfg['epochs']}")
    print(f"  Loss   : {cfg['seg_loss_weight']}*Focal+Dice + {cfg['cls_loss_weight']}*CE_cls")

    print("\n--- Data ---")
    train_loader, val_loader, test_loader = prepare_data(cfg, quick)

    print("\n--- Class Weights ---")
    cc = count_class_pixels(train_loader)
    for c in range(NUM_CLASSES): print(f"  {CLASS_NAMES[c]:12s}: {cc[c]:>14,} px")
    rmif = compute_rmif_weights(cc, NUM_CLASSES, device)

    print("\n--- Model ---")
    model = get_multitask_model(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_p:,}  Trainable: {train_p:,} (encoder frozen)")

    seg_fn = get_loss_fn(rmif_weights=rmif, gamma=2.0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["cosine_T0"], T_mult=cfg["cosine_T_mult"])
    early_stop = EarlyStopping(patience=cfg["early_stop_patience"])

    log_fields = ["epoch", "train_loss", "train_seg_loss", "train_cls_loss", "train_cls_acc",
                  "val_loss", "val_seg_loss", "val_cls_loss", "val_cls_acc",
                  "mean_dice", "tumor_dice", "accuracy", "macro_f1", "lr"]
    csv_log = CSVLogger(os.path.join(cfg["output_dir"], "training_log.csv"), log_fields)

    print("\n" + "=" * 65 + "\n  TRAINING\n" + "=" * 65)
    history = defaultdict(list)
    best_dice, best_path = 0.0, os.path.join(cfg["output_dir"], "best_model_multitask.pth")

    for epoch in range(cfg["epochs"]):
        t0 = time.time()

        if epoch == cfg["unfreeze_epoch"]:
            print(f"\n  >>> Epoch {epoch+1}: Unfreezing encoder <<<")
            model.unfreeze_encoder()
            optimizer = torch.optim.Adam([
                {"params": model.encoder.parameters(), "lr": cfg["lr"]*cfg["encoder_lr_factor"]},
                {"params": model.decoder.parameters(), "lr": cfg["lr"]},
                {"params": model.segmentation_head.parameters(), "lr": cfg["lr"]},
                {"params": model.cls_head.parameters(), "lr": cfg["lr"]},
            ], weight_decay=cfg["weight_decay"])
            remaining = cfg["epochs"] - epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=min(cfg["cosine_T0"], remaining), T_mult=cfg["cosine_T_mult"])
            print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        result = train_one_epoch(model, train_loader, seg_fn, optimizer, device, cfg, epoch)
        if result[0] is None: print("[ABORT]"); break
        tr_loss, tr_seg, tr_cls, tr_cls_acc = result

        vl_loss, vl_seg, vl_cls, vl_cls_acc, vm = validate(model, val_loader, seg_fn, device, cfg)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        for k, v in [("train_loss", tr_loss), ("val_loss", vl_loss),
                      ("val_dice", vm["mean_dice"]), ("val_tumor_dice", vm["mean_dice_tumor"]),
                      ("train_cls_acc", tr_cls_acc), ("val_cls_acc", vl_cls_acc), ("lr", lr)]:
            history[k].append(v)

        csv_log.log({"epoch": epoch+1, "train_loss": f"{tr_loss:.6f}",
                     "train_seg_loss": f"{tr_seg:.6f}", "train_cls_loss": f"{tr_cls:.6f}",
                     "train_cls_acc": f"{tr_cls_acc:.4f}",
                     "val_loss": f"{vl_loss:.6f}", "val_seg_loss": f"{vl_seg:.6f}",
                     "val_cls_loss": f"{vl_cls:.6f}", "val_cls_acc": f"{vl_cls_acc:.4f}",
                     "mean_dice": f"{vm['mean_dice']:.4f}", "tumor_dice": f"{vm['mean_dice_tumor']:.4f}",
                     "accuracy": f"{vm['accuracy']:.4f}", "macro_f1": f"{vm['macro_f1']:.4f}",
                     "lr": f"{lr:.2e}"})

        print(f"\n  E{epoch+1}/{cfg['epochs']} ({elapsed:.0f}s) LR={lr:.2e}")
        print(f"  Loss: {tr_loss:.4f} (seg:{tr_seg:.4f} cls:{tr_cls:.4f}) | Val: {vl_loss:.4f}")
        print(f"  Dice: {vm['mean_dice']:.4f} (tumor:{vm['mean_dice_tumor']:.4f}) | Cls Acc: {vl_cls_acc:.4f}")

        tumor_dice = vm["mean_dice_tumor"]
        early_stop.step(tumor_dice)
        if vm["mean_dice"] > best_dice:
            best_dice = vm["mean_dice"]
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "mean_dice": best_dice, "config": cfg}, best_path)
            print(f"  ** Best saved (Dice={best_dice:.4f}) **")

        if (epoch+1) % cfg["viz_every_n"] == 0 or epoch == 0:
            vp = os.path.join(cfg["output_dir"], f"preds_epoch_{epoch+1:03d}.png")
            visualize_preds(model, val_loader, device, vp); print(f"  Saved -> {vp}")

        if early_stop.should_stop: print(f"\n  Early stop at epoch {epoch+1}"); break
        print("-" * 65)

    plot_curves(dict(history), os.path.join(cfg["output_dir"], "training_curves.png"))

    print("\n" + "=" * 65 + "\n  TEST EVALUATION\n" + "=" * 65)
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"  Loaded best from epoch {ck['epoch']} (Dice={ck['mean_dice']:.4f})")

    _, _, _, test_cls_acc, tm = validate(model, test_loader, seg_fn, device, cfg, collect_roc=True)
    print(f"\n  Classification Accuracy: {test_cls_acc:.4f}")
    print_metrics(tm)
    plot_confusion_matrix(tm["confusion_matrix"], os.path.join(cfg["output_dir"], "confusion_matrix.png"))
    visualize_preds(model, test_loader, device, os.path.join(cfg["output_dir"], "test_predictions.png"), n=6)

    results = {"accuracy": float(tm["accuracy"]), "macro_f1": float(tm["macro_f1"]),
               "mean_dice": float(tm["mean_dice"]), "mean_dice_tumor": float(tm["mean_dice_tumor"]),
               "mean_iou": float(tm["mean_iou"]), "cls_accuracy": float(test_cls_acc),
               "roc_auc": float(tm["roc_auc"]) if tm["roc_auc"] else None,
               "per_class": {CLASS_NAMES[c]: {k: float(v) for k, v in m.items()}
                             for c, m in tm["per_class"].items()}}
    with open(os.path.join(cfg["output_dir"], "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Best Dice: {best_dice:.4f} | Cls Acc: {test_cls_acc:.4f}")
    print(f"  Outputs: {cfg['output_dir']}/")
    return history, tm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task training")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    train_multitask(quick=args.quick)

