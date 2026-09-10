"""Mask-safe losses and metrics for dynamically padded KT batches."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


def binary_metrics(
    probabilities: torch.Tensor, labels: torch.Tensor, *, loss: float, threshold: float = 0.5
) -> dict[str, float | int]:
    """Compute the benchmark's complete metric contract on genuine targets."""

    probability_values = probabilities.detach().cpu().double().numpy()
    label_values = labels.detach().cpu().long().numpy()
    if probability_values.ndim != 1 or label_values.shape != probability_values.shape:
        raise ValueError("probabilities and labels must be aligned one-dimensional arrays")
    if len(label_values) == 0 or not set(np.unique(label_values)).issubset({0, 1}):
        raise ValueError("metrics require non-empty binary targets")
    predicted = (probability_values >= threshold).astype(np.int64)
    auc = float("nan")
    if len(np.unique(label_values)) == 2:
        auc = float(roc_auc_score(label_values, probability_values))
    return {
        "targets": int(len(label_values)),
        "loss": float(loss),
        "roc_auc": auc,
        "accuracy": float(accuracy_score(label_values, predicted)),
        "precision": float(precision_score(label_values, predicted, zero_division=0)),
        "recall": float(recall_score(label_values, predicted, zero_division=0)),
        "f1": float(f1_score(label_values, predicted, zero_division=0)),
        "mse": float(mean_squared_error(label_values, probability_values)),
        "threshold": float(threshold),
    }
