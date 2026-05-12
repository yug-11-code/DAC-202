"""
loss.py - Multiclass Focal Loss with RMIF Class Weighting
Brain Tumor Segmentation - BRISC 2025 Dataset

Loss function adapted from the LandSeg paper (Chauhan et al.):
    Focal Loss with Root Mean Inverse Frequency (RMIF) alpha weighting.
    Originally used for 9-class land cover segmentation with extreme
    class imbalance. Applied here to 4-class brain tumor segmentation
    (C = 4 instead of C = 9).

This is a SINGLE unified loss -- the RMIF weights serve as the per-class
alpha inside the Focal Loss formula. There is no separate CE term,
no lambda hyperparameter, no additive Dice component.

Formula (per pixel):
    FL(p_t) = -w_c * (1 - p_t)^gamma * log(p_t)

where:
    p_t   = softmax probability of the ground-truth class
    w_c   = RMIF weight for class c  (this IS alpha)
    gamma = 2.0 (fixed)

Approximate BRISC 2025 class pixel counts (5000 train images, 256x256):
    No Tumor   (class 0): ~250,000,000 pixels  <- dominant (~85-90%)
    Glioma     (class 1): ~15,000,000  pixels   (~4-6%)
    Meningioma (class 2): ~10,000,000  pixels   (~2-4%)
    Pituitary  (class 3): ~12,000,000  pixels   (~3-5%)

Usage:
    from loss import compute_rmif_weights, get_loss_fn

    weights = compute_rmif_weights(class_pixel_counts, num_classes=4, device=device)
    loss_fn = get_loss_fn(rmif_weights=weights, gamma=2.0)
    loss    = loss_fn(logits, targets)

Input shapes:
    logits  : (B, 4, 256, 256) float32 -- raw logits from model (NO softmax)
    targets : (B, 256, 256)    int64   -- class labels {0, 1, 2, 3}

Output:
    scalar loss tensor (mean reduction)
"""

import torch
import torch.nn.functional as F

eps = 1e-8


def compute_rmif_weights(class_pixel_counts, num_classes=4, device="cpu"):
    """
    Compute Root Mean Inverse Frequency (RMIF) class weights.

    From the LandSeg paper (Chauhan et al., Section 3.3):
        w_c = sqrt(N_total) / (C * sqrt(N_c))

    Engineering additions beyond the paper:
        (a) Normalize so weights sum to num_classes (for training stability)
        (b) Clip each weight to max 10.0 (guard against extreme values)
        Order: normalize first, then clip.

    Args:
        class_pixel_counts: list or 1-D tensor of length num_classes
            Pixel counts [N_0, N_1, N_2, N_3] from the training set.
        num_classes: int, number of segmentation classes (default 4)
        device: torch device to place weights on

    Returns:
        1-D float32 tensor of shape (num_classes,) on device
    """
    counts = torch.tensor(class_pixel_counts, dtype=torch.float32)

    N_total = counts.sum()

    weights = torch.sqrt(N_total) / (num_classes * torch.sqrt(counts))

    weights = weights * (num_classes / weights.sum())

    weights = torch.clamp(weights, max=10.0)

    weights = weights.to(dtype=torch.float32, device=device)

    return weights


def get_loss_fn(rmif_weights, gamma=2.0):
    """
    Build a Multiclass Focal Loss function with RMIF alpha weighting.

    The returned callable implements:
        FL(p_t) = -w_c * (1 - p_t)^gamma * log(p_t)

    using F.log_softmax for numerical stability.

    Args:
        rmif_weights: float32 tensor of shape (num_classes,)
            RMIF class weights from compute_rmif_weights()
        gamma: float, focal loss focusing parameter (default 2.0, fixed)

    Returns:
        loss_fn(logits, targets) -> scalar tensor
    """
    _weights = rmif_weights.clone()
    _gamma = gamma

    def loss_fn(logits, targets):
        """
        Compute Multiclass Focal Loss with RMIF alpha.

        Args:
            logits:  (B, C, H, W) float32 -- raw logits from model
            targets: (B, H, W)    int64   -- class labels {0, ..., C-1}

        Returns:
            scalar loss tensor (mean reduction)
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets_unsqueezed = targets.unsqueeze(1)
        log_p_t = log_probs.gather(dim=1, index=targets_unsqueezed).squeeze(1)
        p_t = probs.gather(dim=1, index=targets_unsqueezed).squeeze(1)

        alpha_t = _weights[targets]

        focal_weight = (1.0 - p_t) ** _gamma

        loss = -alpha_t * focal_weight * log_p_t

        return loss.mean()

    return loss_fn


if __name__ == "__main__":
    print("=" * 60)
    print("  loss.py - Multiclass Focal Loss + RMIF Sanity Check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\n--- Check 1: RMIF weight properties ---")
    class_counts = [250_000_000, 15_000_000, 10_000_000, 12_000_000]
    weights = compute_rmif_weights(class_counts, num_classes=4, device=device)

    print(f"RMIF weights : {weights}")
    print(f"Weights sum  : {weights.sum().item():.4f}  (expected ~= 4.0)")
    print(f"Weights max  : {weights.max().item():.4f}  (expected <= 10.0)")

    assert weights.shape == (4,),                        "Wrong weight shape"
    assert weights.dtype == torch.float32,               "Weights must be float32"
    assert abs(weights.sum().item() - 4.0) < 0.01,      "Weights must sum to 4.0"
    assert weights.max().item() <= 10.0,                 "Weights must be clipped to 10.0"
    assert (weights > 0).all(),                          "All weights must be positive"
    assert weights[0] == weights.min(),                  "Background must have lowest weight"
    assert (weights[1:] > weights[0]).all(),             "Tumor weights > background weight"
    print("  All weight assertions passed!")

    print("\n--- Check 2: Loss function output properties ---")
    loss_fn = get_loss_fn(rmif_weights=weights, gamma=2.0)

    logits  = torch.randn(4, 4, 256, 256).to(device)
    targets = torch.randint(0, 4, (4, 256, 256)).to(device)
    loss    = loss_fn(logits, targets)

    print(f"Loss value   : {loss.item():.4f}")
    assert loss.ndim == 0,        "Loss must be a scalar"
    assert loss.item() > 0,       "Loss must be positive"
    assert not torch.isnan(loss), "Loss must not be NaN"
    assert not torch.isinf(loss), "Loss must not be Inf"
    print("  All loss property assertions passed!")

    print("\n--- Check 3: Loss direction (wrong > correct) ---")
    bad_logits = torch.zeros(2, 4, 256, 256).to(device)
    bad_logits[:, 0, :, :] = 10.0
    tumor_targets = torch.ones(2, 256, 256, dtype=torch.long).to(device)
    bad_loss = loss_fn(bad_logits, tumor_targets)

    good_logits = torch.zeros(2, 4, 256, 256).to(device)
    good_logits[:, 1, :, :] = 10.0
    good_loss = loss_fn(good_logits, tumor_targets)

    print(f"Bad  loss (wrong class predicted)  : {bad_loss.item():.4f}")
    print(f"Good loss (correct class predicted): {good_loss.item():.4f}")
    assert bad_loss.item() > good_loss.item(), \
        "Loss must be higher when predictions are wrong"
    print("  Loss direction assertion passed!")

    print("\n--- Check 4: Focal effect (confident < uncertain) ---")
    confident_correct = torch.zeros(2, 4, 256, 256).to(device)
    confident_correct[:, 1, :, :] = 100.0
    confident_loss = loss_fn(confident_correct, tumor_targets)

    uncertain_correct = torch.zeros(2, 4, 256, 256).to(device)
    uncertain_correct[:, 1, :, :] = 0.1
    uncertain_loss = loss_fn(uncertain_correct, tumor_targets)

    print(f"Confident correct loss : {confident_loss.item():.6f}  (expected ~= 0)")
    print(f"Uncertain correct loss : {uncertain_loss.item():.4f}   (expected > 0)")
    assert confident_loss.item() < uncertain_loss.item(), \
        "Focal effect: confident correct predictions must have lower loss"
    print("  Focal effect assertion passed!")

    print("\n" + "=" * 60)
    print("  loss.py -- All checks passed!")
    print("=" * 60)
