"""
dataset.py - Preprocessing & DataLoader Pipeline
Brain Tumor Segmentation - BRISC 2025 Dataset

Aligned with step1_exploration.py findings:
  - Dataset: BRISC 2025 (segmentation_task)
  - Structure: segmentation_task/{train,test}/{images,masks}/
  - Images: .jpg in images/ folders
  - Masks:  .png in masks/ folders (same stem as image)
  - 4 classes: 0=background, 1=glioma, 2=meningioma, 3=pituitary
  - 3-channel input: [Original Gray, CLAHE, Sobel Edge Magnitude]
  - Split: 80/10/10 stratified by rarest tumor class (matches step1)
  - Normalization: ImageNet mean/std for pretrained EfficientNet-B4 encoder
  - RMIF class weights loaded from outputs/class_weights.json (from step1)
"""

import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


DATASET_ROOT = os.environ.get("DATASET_ROOT", r"C:\Users\Arman Srivastava\Desktop\Pillai Project\archive\brisc2025")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "outputs")

IMG_SIZE     = 256
BATCH_SIZE   = 16
NUM_WORKERS  = 2
RANDOM_SEED  = 42
NUM_CLASSES  = 4

CLASS_NAMES  = {0: "background", 1: "glioma", 2: "meningioma", 3: "pituitary"}
CLASS_COLORS_RGB = {
    0: (0,   0,   0),
    1: (255, 0,   0),
    2: (0,   255, 0),
    3: (0,   0,   255),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def discover_pairs(root_dir):
    """
    Scan the BRISC dataset for (image, mask) pairs.

    Looks for 'segmentation_task' subdirectory, then finds all
    'images/' and 'masks/' folders recursively. Pairs by matching
    the tier (train/test) and filename stem.

    Returns:
        list of (image_path_str, mask_path_str) tuples
    """
    root = Path(root_dir)
    if not root.exists():
        print(f"[ERROR] Dataset root not found: {root}")
        sys.exit(1)

    seg_task_dir = root / "segmentation_task"
    if seg_task_dir.exists():
        scan_root = seg_task_dir
        print(f"-> Found 'segmentation_task' directory. Scanning there.")
    else:
        scan_root = root

    img_exts  = {".jpg", ".jpeg", ".png"}
    mask_exts = {".png", ".jpg"}

    image_paths = []
    mask_paths  = []

    for dirpath, dirnames, filenames in os.walk(scan_root):
        p = Path(dirpath)
        if p.name.lower() == "images":
            for f in p.glob("*"):
                if f.is_file() and f.suffix.lower() in img_exts:
                    image_paths.append(f)
        elif p.name.lower() == "masks":
            for f in p.glob("*"):
                if f.is_file() and f.suffix.lower() in mask_exts:
                    mask_paths.append(f)

    if not image_paths:
        print("[INFO] No 'images' directories found. Falling back to full scan...")
        for f in scan_root.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() in img_exts:
                if "mask" in f.name.lower():
                    mask_paths.append(f)
                else:
                    image_paths.append(f)

    mask_lookup = {}
    for mp in mask_paths:
        tier = mp.parent.parent.name
        key  = f"{tier}_{mp.stem}"
        mask_lookup[key] = mp

    matched = []
    for ip in image_paths:
        tier = ip.parent.parent.name
        key  = f"{tier}_{ip.stem}"
        if key in mask_lookup:
            matched.append((str(ip), str(mask_lookup[key])))

    print(f"Discovered {len(matched)} image-mask pairs")
    print(f"  (from {len(image_paths)} images, {len(mask_paths)} masks)")
    return matched


def build_3channel(image_bgr):
    """
    Build 3-channel input from a BGR image:
        Ch1: Original grayscale
        Ch2: CLAHE-enhanced (clipLimit=2.0, tileGridSize=8x8)
        Ch3: Sobel gradient magnitude
    Returns: numpy array of shape (H, W, 3), dtype uint8
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch_clahe = clahe.apply(gray)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ch_edges = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))

    return np.stack([gray, ch_clahe, ch_edges], axis=-1)


def read_mask(mask_path):
    """
    Read a BRISC 2025 segmentation mask and convert to multi-class labels.

    BRISC 2025 masks are BINARY:
      - 0   = background
      - 255 = tumor region
      - 1-7 and 248-254 = anti-aliasing artifacts at boundaries

    The tumor CLASS is encoded in the filename:
      - '_gl_' -> class 1 (glioma)
      - '_me_' -> class 2 (meningioma)
      - '_pi_' -> class 3 (pituitary)

    Strategy: threshold at 127 to get binary mask, then assign the
    class label derived from the filename.

    Falls back to PIL if cv2 returns None.
    If 3-channel, takes the first channel only.
    Returns: numpy array (H, W) dtype uint8 with values in {0, 1, 2, 3}.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        mask = np.array(Image.open(mask_path).convert("L"))
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = mask.astype(np.uint8)

    tumor_class = _class_from_filename(mask_path)

    binary = (mask > 127).astype(np.uint8)
    result = binary * tumor_class

    return result


def _class_from_filename(path):
    """
    Extract tumor class label from BRISC 2025 filename convention.

    Filename pattern: brisc2025_{split}_{id}_{type}_{plane}_{seq}.png
    where type is: gl (glioma=1), me (meningioma=2), pi (pituitary=3)

    Returns: int class label (1, 2, or 3). Defaults to 1 if unrecognised.
    """
    basename = os.path.basename(path).lower()
    if '_gl_' in basename:
        return 1
    elif '_me_' in basename:
        return 2
    elif '_pi_' in basename:
        return 3
    else:
        return 1


def get_train_transforms(img_size=IMG_SIZE):
    """Augmentation pipeline for training set."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            p=0.3,
        ),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=IMG_SIZE):
    """Transforms for validation and test sets (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class BRISCDataset(Dataset):
    """
    PyTorch Dataset for BRISC brain tumor segmentation.

    Each sample returns:
        image: (3, H, W) float32 tensor - normalized 3-channel input
               [gray, clahe, sobel_edges]
        mask:  (H, W) int64 tensor - class labels {0,1,2,3}
    """

    def __init__(self, pairs, transform=None):
        """
        Args:
            pairs: list of (image_path_str, mask_path_str)
            transform: albumentations Compose pipeline
        """
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Could not read image: {img_path}")

        image_3ch = build_3channel(img_bgr)

        mask = read_mask(mask_path)

        if self.transform:
            transformed = self.transform(image=image_3ch, mask=mask)
            image_tensor = transformed["image"]
            mask_tensor  = transformed["mask"]
        else:
            image_3ch = cv2.resize(image_3ch, (IMG_SIZE, IMG_SIZE))
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)
            image_tensor = torch.from_numpy(
                image_3ch.transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            mask_tensor = torch.from_numpy(mask)

        mask_tensor = mask_tensor.long()

        return image_tensor, mask_tensor


def _assign_stratification_label(mask_path, images_with):
    """
    Determine stratification label for splitting.
    Uses the rarest tumor class present in the mask.
    Falls back to 0 (background) if no tumor is found.
    """
    try:
        mask = read_mask(mask_path)
        present = [c for c in [1, 2, 3] if (mask == c).any()]
        if not present:
            return 0
        rarest = min(present, key=lambda c: images_with.get(c, 0))
        return rarest
    except Exception:
        return 0


def _compute_images_with(pairs):
    """
    Count how many images contain each class.
    Needed for stratification by rarest class.
    """
    from collections import defaultdict
    images_with = defaultdict(int)
    for _, mask_path in pairs:
        try:
            mask = read_mask(mask_path)
            for c in [0, 1, 2, 3]:
                if (mask == c).any():
                    images_with[c] += 1
        except Exception:
            pass
    return dict(images_with)


def load_class_weights(weights_path=None):
    """
    Load RMIF class weights from the JSON file produced by step1_exploration.
    Returns a list [w0, w1, w2, w3] suitable for loss functions.
    """
    if weights_path is None:
        weights_path = os.path.join(OUTPUT_DIR, "class_weights.json")

    if not os.path.exists(weights_path):
        print(f"[WARN] Class weights file not found: {weights_path}")
        print("  Run step1_exploration.py first, or weights default to [1,1,1,1]")
        return [1.0, 1.0, 1.0, 1.0]

    with open(weights_path, "r") as f:
        data = json.load(f)

    raw = data.get("_raw_by_index", {})
    weights = [raw.get(str(c), raw.get(c, 1.0)) for c in range(NUM_CLASSES)]
    print(f"Loaded RMIF class weights: {[f'{w:.4f}' for w in weights]}")
    return weights


def create_dataloaders(root_dir=None,
                       batch_size=BATCH_SIZE,
                       num_workers=NUM_WORKERS,
                       seed=RANDOM_SEED):
    """
    Discover pairs, compute stratified 80/10/10 split, build DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, all_pairs
    """
    if root_dir is None:
        root_dir = os.environ.get("DATASET_ROOT", DATASET_ROOT)

    pairs = discover_pairs(root_dir)
    if not pairs:
        print("[ERROR] No image-mask pairs found.")
        sys.exit(1)

    print("\nComputing class frequencies for stratified split...")
    images_with = _compute_images_with(pairs)
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:12s}: {images_with.get(c, 0)} images")

    strat_labels = [
        _assign_stratification_label(mask_path, images_with)
        for _, mask_path in pairs
    ]

    indices = list(range(len(pairs)))
    try:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.20, stratify=strat_labels,
            random_state=seed
        )
        temp_labels = [strat_labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, stratify=temp_labels,
            random_state=seed
        )
    except ValueError:
        print("  [WARN] Stratified split failed. Falling back to random split.")
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.20, random_state=seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=seed
        )

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs   = [pairs[i] for i in val_idx]
    test_pairs  = [pairs[i] for i in test_idx]

    print(f"\nSplit sizes:")
    print(f"  train : {len(train_pairs)}")
    print(f"  val   : {len(val_pairs)}")
    print(f"  test  : {len(test_pairs)}")
    print(f"  total : {len(pairs)}")

    train_dataset = BRISCDataset(train_pairs, transform=get_train_transforms())
    val_dataset   = BRISCDataset(val_pairs,   transform=get_val_transforms())
    test_dataset  = BRISCDataset(test_pairs,  transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, pairs


def colorize_mask(mask_np):
    """Return RGB colorized mask (H, W, 3) from class-label mask (H, W)."""
    colored = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for c, rgb in CLASS_COLORS_RGB.items():
        colored[mask_np == c] = rgb
    return colored


def run_sanity_checks(train_loader, val_loader, test_loader, total_pairs):
    """Print dataset statistics and visualize one sample."""
    print("\n" + "=" * 60)
    print("  SANITY CHECKS")
    print("-" * 60)

    train_n = len(train_loader.dataset)
    val_n   = len(val_loader.dataset)
    test_n  = len(test_loader.dataset)
    print(f"\nTotal samples : {len(total_pairs)}")
    print(f"  train       : {train_n}")
    print(f"  val         : {val_n}")
    print(f"  test        : {test_n}")

    images, masks = next(iter(train_loader))

    print(f"\nImage tensor:")
    print(f"  shape  : {images.shape}")
    print(f"  dtype  : {images.dtype}")
    print(f"  min    : {images.min().item():.4f}")
    print(f"  max    : {images.max().item():.4f}")

    print(f"\nMask tensor:")
    print(f"  shape  : {masks.shape}")
    print(f"  dtype  : {masks.dtype}")
    unique_vals = torch.unique(masks).tolist()
    print(f"  unique : {unique_vals}")

    if images.min() < -3.0 or images.max() > 3.5:
        print("  [!] Image values look unusual - check normalization")
    else:
        print("  [OK] Image values in expected normalized range")

    expected_classes = {0, 1, 2, 3}
    if set(int(v) for v in unique_vals) <= expected_classes:
        print(f"  [OK] Mask contains valid class labels (subset of {expected_classes})")
    else:
        print(f"  [!] Mask has unexpected values outside {expected_classes}")

    weights = load_class_weights()

    print("\nGenerating sanity check visualization...")
    img = images[0]
    msk = masks[0]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_denorm = img * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)

    ch_names = ["Ch1: Original", "Ch2: CLAHE", "Ch3: Sobel Edges", "GT Mask (colorized)"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(3):
        axes[i].imshow(img_denorm[i].numpy(), cmap="gray")
        axes[i].set_title(ch_names[i], fontsize=12, fontweight="bold")
        axes[i].axis("off")

    mask_colored = colorize_mask(msk.numpy())
    axes[3].imshow(mask_colored)
    axes[3].set_title(ch_names[3], fontsize=12, fontweight="bold")
    axes[3].axis("off")

    patches = [mpatches.Patch(color=np.array(CLASS_COLORS_RGB[c]) / 255.0,
                              label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Sample from Train DataLoader (denormalized)", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "dataloader_sanity_check.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Visualization saved -> {out_path}")

    print("\n" + "=" * 60)
    print("  Sanity checks complete!")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("  Brain Tumor Segmentation - Dataset & DataLoader Pipeline")
    print("  Dataset: BRISC 2025 (segmentation_task)")
    print(f"  Root     : {DATASET_ROOT}")
    print(f"  Size     : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch    : {BATCH_SIZE}")
    print(f"  Classes  : {NUM_CLASSES} ({', '.join(CLASS_NAMES.values())})")
    print("=" * 60)

    train_loader, val_loader, test_loader, all_pairs = create_dataloaders()
    run_sanity_checks(train_loader, val_loader, test_loader, all_pairs)
