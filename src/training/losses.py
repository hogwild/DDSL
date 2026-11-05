from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al. 2017)
    FL = - alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter (typical 1~3, often 2.0)
        alpha: class weights tensor of shape [C] (like CE weight), or None
        reduction: 'mean' | 'sum' | 'none'
        ignore_index: label id to ignore (optional)
        eps: numerical stability
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 reduction: str = "mean", ignore_index: int | None = None, eps: float = 1e-12):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # shape [C] or None
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C]
        target: [B]
        """
        # mask ignore_index if provided
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            logits = logits[valid]
            target = target[valid]
            if target.numel() == 0:
                return logits.sum() * 0.0  # zero loss with correct device/dtype

        log_probs = F.log_softmax(logits, dim=-1)              # [B, C]
        probs = log_probs.exp().clamp(min=self.eps, max=1.0)  # [B, C]

        # pick p_t and log(p_t)
        idx = torch.arange(target.size(0), device=target.device)
        pt = probs[idx, target]          # [B]
        logpt = log_probs[idx, target]   # [B]

        focal_factor = (1.0 - pt).pow(self.gamma)  # [B]

        # alpha weighting (class weights)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target]    # [B]
            loss = -alpha_t * focal_factor * logpt
        else:
            loss = -focal_factor * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


def build_loss(
    loss_type: str = "cross_entropy",
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    device: torch.device | str | None = None,
) -> nn.Module:
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)

    loss_type = loss_type.lower()

    if loss_type in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_type in {"focal", "focal_loss"}:
        return FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            reduction="mean",
            ignore_index=None,
        )

    raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_class_weights_from_ids(
    ids,
    num_classes: int,
) -> torch.Tensor:
    ids = np.asarray(ids, dtype=np.int64)

    freq = np.zeros(num_classes, dtype=np.float64)
    unique, counts = np.unique(ids, return_counts=True)

    for cid, cnt in zip(unique, counts):
        if 0 <= cid < num_classes:
            freq[cid] = cnt

    freq = np.maximum(freq, 1.0)

    weights = 1.0 / np.sqrt(freq)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)