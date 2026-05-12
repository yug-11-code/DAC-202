"""
Step 1: Data Exploration and Preprocessing Analysis
Brain Tumor Segmentation - BRISC Dataset
"""

import os
import sys
import json
import ast
import random
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATASET_ROOT = r"C:\Users\Arman Srivastava\Desktop\Pillai Project\archive\brisc2025"
OUTPUT_DIR   = r"outputs"
RANDOM_SEED  = 42
SAMPLE_STATS = 100

CLASS_NAMES  = {0: "background", 1: "glioma", 2: "meningioma", 3: "pituitary"}
CLASS_COLORS = {
    0: (0,   0,   0),
    1: (0,   0, 255),
    2: (0, 255,   0),
    3: (255, 0,   0),
}
CLASS_COLORS_RGB = {k: (v[2], v[1], v[0]) for k, v in CLASS_COLORS.items()}

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def sep(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("-" * 60)


def section1_structure(dataset_root):
    sep("SECTION 1 * Dataset Structure Exploration")

    root = Path(dataset_root)
    if not root.exists():
        print(f"[ERROR] Dataset root not found: {root}")
        sys.exit(1)

    seg_task_dir = root / "segmentation_task"
    if seg_task_dir.exists():
        print(f"-> Found 'segmentation_task' directory. Focusing scan there.")
        scan_root = seg_task_dir
    else:
        scan_root = root

    print(f"\nScanning structure starting at: {scan_root.name}")
    for dirpath, dirnames, filenames in os.walk(scan_root):
        rel = Path(dirpath).relative_to(scan_root)
        depth = len(rel.parts)
        if depth > 2:
            continue
        indent = "  " * depth
        print(f"{indent}{Path(dirpath).name}/")
        if depth < 2 and filenames:
            valid_files = [f for f in filenames if not f.startswith('.')]
            for f in valid_files[:3]:
                print(f"{indent}  {f}")
            if len(valid_files) > 3:
                print(f"{indent}  ... ({len(valid_files)} files)")

    img_exts = {".jpg", ".jpeg", ".png"}
    mask_exts = {".png", ".jpg"}

    image_paths = []
    mask_paths = []

    for dirpath, dirnames, filenames in os.walk(scan_root):
        p = Path(dirpath)
        if p.name.lower() == 'images':
            for f in p.glob("*"):
                if f.is_file() and f.suffix.lower() in img_exts:
                    image_paths.append(f)
        elif p.name.lower() == 'masks':
            for f in p.glob("*"):
                if f.is_file() and f.suffix.lower() in mask_exts:
                    mask_paths.append(f)

    if not image_paths:
        print("\n[INFO] No 'images' directories found. Falling back to full scan...")
        all_files = list(scan_root.rglob("*"))
        for f in all_files:
            if not f.is_file(): continue
            if f.suffix.lower() in img_exts:
                if "mask" in f.name.lower():
                    mask_paths.append(f)
                else:
                    image_paths.append(f)

    print(f"\nTotal raw image files located : {len(image_paths)}")
    print(f"Total raw mask files located  : {len(mask_paths)}")

    matched = []
    unmatched_imgs = []

    mask_lookup = {}
    for mp in mask_paths:
        tier = mp.parent.parent.name
        key = f"{tier}_{mp.stem}"
        mask_lookup[key] = mp

    for ip in image_paths:
        tier = ip.parent.parent.name
        key = f"{tier}_{ip.stem}"
        if key in mask_lookup:
            matched.append((ip, mask_lookup[key]))
        else:
            unmatched_imgs.append(ip)

    img_lookup_keys = {f"{ip.parent.parent.name}_{ip.stem}" for ip in image_paths}
    unmatched_masks = [mp for mp in mask_paths if f"{mp.parent.parent.name}_{mp.stem}" not in img_lookup_keys]

    print(f"\nMatched image-mask pairs     : {len(matched)}")
    print(f"Images without mask          : {len(unmatched_imgs)}")
    print(f"Masks without image          : {len(unmatched_masks)}")

    if len(matched) > 0:
        print("\nExamples of successful pairs:")
        for i, (p, m) in enumerate(matched[:2]):
            print(f"  Pair {i+1}:")
            print(f"    Img -> ...{os.path.sep.join(p.parts[-3:])}")
            print(f"    Msk -> ...{os.path.sep.join(m.parts[-3:])}")

    return matched, unmatched_imgs, unmatched_masks


def section2_image_properties(matched):
    sep("SECTION 2 * Image Property Analysis")

    sizes, channels_img, dtypes_img, corrupted = defaultdict(int), defaultdict(int), defaultdict(int), []
    pixel_values = []

    sample_indices = random.sample(range(len(matched)), min(SAMPLE_STATS, len(matched)))

    for idx, (img_path, _) in enumerate(matched):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("cv2 returned None")

            h, w = img.shape[:2]
            sizes[f"{w}x{h}"] += 1
            nc = 1 if img.ndim == 2 else img.shape[2]
            channels_img[nc] += 1
            dtypes_img[str(img.dtype)] += 1

            if idx in sample_indices:
                arr = img.astype(np.float32)
                pixel_values.append(arr.ravel())

        except Exception as e:
            corrupted.append((img_path, str(e)))

    print(f"\nUnique image sizes (WxH) : {dict(sizes)}")
    print(f"Channel counts          : {dict(channels_img)}")
    print(f"Data types              : {dict(dtypes_img)}")

    if pixel_values:
        flat = np.concatenate(pixel_values)
        print(f"\nPixel stats (sample of {len(pixel_values)} images):")
        print(f"  min={flat.min():.1f}  max={flat.max():.1f}  "
              f"mean={flat.mean():.2f}  std={flat.std():.2f}")

    print(f"\nCorrupted / unreadable : {len(corrupted)}")
    for p, e in corrupted[:5]:
        print(f"  {p} -> {e}")

    return sizes, corrupted


def section3_mask_analysis(matched):
    sep("SECTION 3 * Mask Analysis")

    C = 4
    pixel_counts  = defaultdict(int)
    images_with   = defaultdict(int)
    pure_bg_count = 0
    all_unique_vals = set()
    unexpected_flag = []

    print(f"Processing {len(matched)} masks ...")
    for img_path, mask_path in matched:
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                mask = np.array(Image.open(str(mask_path)).convert("L"))
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            uvals = set(np.unique(mask).tolist())
            all_unique_vals |= uvals

            unexpected = uvals - {0, 1, 2, 3}
            if unexpected:
                unexpected_flag.append((mask_path, unexpected))

            classes_in_img = set()
            for c in range(C):
                cnt = int((mask == c).sum())
                pixel_counts[c] += cnt
                if cnt > 0:
                    classes_in_img.add(c)

            if classes_in_img == {0}:
                pure_bg_count += 1
            for c in classes_in_img:
                images_with[c] += 1

        except Exception as e:
            print(f"  [WARN] Could not read mask {mask_path}: {e}")

    N_total = sum(pixel_counts.values())
    print(f"\nAll unique mask values found : {sorted(all_unique_vals)}")
    if all_unique_vals == {0, 1, 2, 3}:
        print("  * Exactly {0,1,2,3} - as expected")
    else:
        print("  [!] Unexpected values detected!")

    weights = {}
    print(f"\n{'Class':<12} {'Name':<12} {'Pixels':>14} {'%':>7} {'RMIF Weight':>12} {'#Images':>8}")
    print("-" * 70)
    for c in range(C):
        nc = pixel_counts[c]
        pct = 100.0 * nc / N_total if N_total else 0
        w   = (np.sqrt(N_total) / (np.sqrt(C) * np.sqrt(nc))) if nc > 0 else 0.0
        weights[c] = round(float(w), 6)
        print(f"{c:<12} {CLASS_NAMES[c]:<12} {nc:>14,} {pct:>6.2f}% {w:>12.4f} {images_with[c]:>8}")

    print(f"\nPure-background images (no tumor): {pure_bg_count}")
    if unexpected_flag:
        print(f"\n[WARNING] {len(unexpected_flag)} masks have unexpected values:")
        for p, v in unexpected_flag[:5]:
            print(f"  {p} -> {v}")

    weights_named = {CLASS_NAMES[c]: weights[c] for c in range(C)}
    weights_named["_raw_by_index"] = weights
    out_path = os.path.join(OUTPUT_DIR, "class_weights.json")
    with open(out_path, "w") as f:
        json.dump(weights_named, f, indent=2)
    print(f"\nRMIF class weights saved -> {out_path}")

    return pixel_counts, weights, images_with, pure_bg_count, unexpected_flag


def _colorize_mask(mask):
    """Return RGB colorized mask (H,W,3)."""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, rgb in CLASS_COLORS_RGB.items():
        colored[mask == c] = rgb
    return colored


def section4_visual_sanity(matched):
    sep("SECTION 4 * Visual Sanity Check")

    samples = random.sample(matched, min(6, len(matched)))
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    col_titles = ["Original Image", "Mask Overlay", "Raw Mask"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=13, fontweight="bold")

    for row, (img_path, mask_path) in enumerate(samples):
        try:
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                mask = np.array(Image.open(str(mask_path)).convert("L"))
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            colored = _colorize_mask(mask)
            overlay = cv2.addWeighted(img_rgb, 0.6, colored, 0.4, 0)

            axes[row][0].imshow(img_rgb);    axes[row][0].axis("off")
            axes[row][1].imshow(overlay);    axes[row][1].axis("off")
            axes[row][2].imshow(colored);    axes[row][2].axis("off")
            axes[row][0].set_ylabel(img_path.name, fontsize=7, rotation=0,
                                    labelpad=120, va="center")
        except Exception as e:
            print(f"  [WARN] Could not process {img_path.name}: {e}")

    patches = [mpatches.Patch(color=np.array(CLASS_COLORS_RGB[c])/255,
                               label=CLASS_NAMES[c]) for c in range(4)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Visual Sanity Check - 6 Random Samples", fontsize=15, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sanity_check.png")
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved -> {out}")


def _make_3ch(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch2 = clahe.apply(gray)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ch3 = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))
    return gray, ch2, ch3


def section5_three_channel_preview(matched):
    sep("SECTION 5 * 3-Channel Input Preview")

    samples = random.sample(matched, min(3, len(matched)))
    fig, axes = plt.subplots(len(samples), 4, figsize=(18, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    col_titles = ["Ch1: Original (Gray)", "Ch2: CLAHE", "Ch3: Sobel Edges", "GT Mask"]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=12, fontweight="bold")

    for row, (img_path, mask_path) in enumerate(samples):
        try:
            img_bgr = cv2.imread(str(img_path))
            ch1, ch2, ch3 = _make_3ch(img_bgr)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                mask = np.array(Image.open(str(mask_path)).convert("L"))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            colored = _colorize_mask(mask)

            for col, arr in enumerate([ch1, ch2, ch3]):
                axes[row][col].imshow(arr, cmap="gray"); axes[row][col].axis("off")
            axes[row][3].imshow(colored); axes[row][3].axis("off")
            axes[row][0].set_ylabel(img_path.name, fontsize=7, rotation=0,
                                    labelpad=120, va="center")
        except Exception as e:
            print(f"  [WARN] {img_path.name}: {e}")

    plt.suptitle("3-Channel Model Input Preview", fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "three_channel_preview.png")
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved -> {out}")


def _assign_label(img_path, mask_path, images_with):
    """Rarest tumor class present in the image (fallback: 0 = background)."""
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.array(Image.open(str(mask_path)).convert("L"))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        present = [c for c in [1, 2, 3] if (mask == c).any()]
        if not present:
            return 0, []
        rarest = min(present, key=lambda c: images_with.get(c, 0))
        return rarest, present
    except Exception:
        return 0, []


def section6_split(matched, images_with):
    sep("SECTION 6 * Train / Val / Test Split  (80 / 10 / 10)")

    records = []
    for img_path, mask_path in matched:
        label, classes_present = _assign_label(img_path, mask_path, images_with)
        records.append({
            "filepath_image": str(img_path),
            "filepath_mask":  str(mask_path),
            "strat_label":    label,
            "tumor_classes_present": str(classes_present),
        })

    df = pd.DataFrame(records)
    labels = df["strat_label"].tolist()

    try:
        train_idx, temp_idx = train_test_split(
            range(len(df)), test_size=0.20, stratify=labels, random_state=RANDOM_SEED)
        temp_labels = [labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, stratify=temp_labels, random_state=RANDOM_SEED)
    except ValueError:
        print("  [WARN] Stratified split failed (too few samples per class). Using random split.")
        train_idx, temp_idx = train_test_split(
            range(len(df)), test_size=0.20, random_state=RANDOM_SEED)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED)

    df["split"] = "train"
    df.loc[list(val_idx),  "split"] = "val"
    df.loc[list(test_idx), "split"] = "test"
    df = df.drop(columns=["strat_label"])

    print(f"\nSplit sizes:")
    print(f"  train : {(df.split=='train').sum()}")
    print(f"  val   : {(df.split=='val').sum()}")
    print(f"  test  : {(df.split=='test').sum()}")

    print("\nClass distribution per split:")
    for split in ["train", "val", "test"]:
        subset = df[df.split == split]
        present_flat = []
        for row in subset["tumor_classes_present"]:
            try:
                present_flat.extend(ast.literal_eval(row))
            except Exception:
                pass
        counts = {CLASS_NAMES[c]: present_flat.count(c) for c in range(1, 4)}
        print(f"  {split:6s}: {counts}")

    csv_path = os.path.join(OUTPUT_DIR, "dataset_splits.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSplit CSV saved -> {csv_path}")
    return df


def section7_summary(matched, unmatched_imgs, unmatched_masks,
                     sizes, corrupted, pixel_counts, weights,
                     images_with, pure_bg_count, unexpected_flag, df_splits):
    sep("SECTION 7 * Summary Report")

    N_total = sum(pixel_counts.values())
    C = 4

    print(f"""
+---------------------------------------------+
|            DATASET SUMMARY REPORT           |
+---------------------------------------------+
| Total matched image-mask pairs : {len(matched):<10}|
| Images without mask            : {len(unmatched_imgs):<10}|
| Masks  without image           : {len(unmatched_masks):<10}|
| Corrupted files                : {len(corrupted):<10}|
+---------------------------------------------+
| Image sizes (W*H -> count)                   |""")
    for sz, cnt in sizes.items():
        print(f"|   {sz:<18} -> {cnt:<22}|")

    print(f"""+---------------------------------------------+
| Class Pixel Distribution                    |
|  {'Class':<12} {'Pixels':>12} {'%':>6} {'RMIF_w':>8}   |""")
    for c in range(C):
        nc  = pixel_counts[c]
        pct = 100.0 * nc / N_total if N_total else 0
        w   = weights[c]
        print(f"|  {CLASS_NAMES[c]:<12} {nc:>12,} {pct:>5.1f}% {w:>8.4f}   |")

    print(f"""+---------------------------------------------+
| Images containing each class                |""")
    for c in range(1, C):
        print(f"|  {CLASS_NAMES[c]:<12} : {images_with[c]:<28}|")
    print(f"|  pure background : {pure_bg_count:<27}|")

    train_n = (df_splits.split == "train").sum()
    val_n   = (df_splits.split == "val").sum()
    test_n  = (df_splits.split == "test").sum()
    print(f"""+---------------------------------------------+
| Dataset Splits                              |
|  train : {train_n:<38}|
|  val   : {val_n:<38}|
|  test  : {test_n:<38}|
+---------------------------------------------+
| Warnings                                    |""")

    warns = []
    if corrupted:       warns.append(f"{len(corrupted)} corrupted file(s)")
    if unmatched_imgs:  warns.append(f"{len(unmatched_imgs)} images missing masks")
    if unmatched_masks: warns.append(f"{len(unmatched_masks)} masks missing images")
    if unexpected_flag: warns.append(f"{len(unexpected_flag)} masks with unexpected pixel values")

    if warns:
        for w in warns:
            print(f"|  [!] {w:<40}|")
    else:
        print(f"|  [*] No warnings - dataset looks clean!     |")

    print("+---------------------------------------------+")

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [CLASS_NAMES[c] for c in range(C)]
    ys = [pixel_counts[c] for c in range(C)]
    colors = [np.array(CLASS_COLORS_RGB[c]) / 255.0 for c in range(C)]
    colors[0] = np.array([0.4, 0.4, 0.4])
    bars = ax.bar(xs, ys, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total Pixel Count")
    ax.set_title("Class Pixel Distribution - BRISC Dataset")
    ax.set_yscale("log")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nDistribution chart saved -> {out}")


def main():
    print("=" * 60)
    print("  Brain Tumor Segmentation - Step 1: Data Exploration")
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print("=" * 60)

    matched, unmatched_imgs, unmatched_masks = section1_structure(DATASET_ROOT)
    if not matched:
        print("[ERROR] No matched image-mask pairs found. Check DATASET_ROOT and naming convention.")
        sys.exit(1)

    sizes, corrupted = section2_image_properties(matched)

    pixel_counts, weights, images_with, pure_bg_count, unexpected_flag = \
        section3_mask_analysis(matched)

    section4_visual_sanity(matched)

    section5_three_channel_preview(matched)

    df_splits = section6_split(matched, images_with)

    section7_summary(matched, unmatched_imgs, unmatched_masks,
                     sizes, corrupted, pixel_counts, weights,
                     images_with, pure_bg_count, unexpected_flag, df_splits)

    print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("  class_weights.json  <- use these in your Focal + Dice loss")
    print("  dataset_splits.csv  <- train/val/test file paths")
    print("  sanity_check.png    <- visual overlay check")
    print("  three_channel_preview.png")
    print("  class_distribution.png")
    print("\nStep 1 complete!")


if __name__ == "__main__":
    main()
