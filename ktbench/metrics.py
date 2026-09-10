"""Mask-safe losses and metrics for dynamically padded KT batches."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != labels.shape or logits.shape != mask.shape:
        raise ValueError("logits, labels, and metric mask must have identical shapes")
    if not mask.any():
        raise ValueError("metric mask contains no supervised target")
    losses = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction="none")
    return losses.masked_select(mask).mean()


def masked_binary_counts(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5
) -> dict[str, int]:
    """Return additive classification counts without PAD/EOS positions."""

    truth = labels.masked_select(mask).bool()
    prediction = torch.sigmoid(logits.masked_select(mask)) >= threshold
    return {
        "targets": int(truth.numel()),
        "correct": int((prediction == truth).sum().item()),
        "true_positive": int((prediction & truth).sum().item()),
        "false_positive": int((prediction & ~truth).sum().item()),
        "true_negative": int((~prediction & ~truth).sum().item()),
        "false_negative": int((~prediction & truth).sum().item()),
    }
