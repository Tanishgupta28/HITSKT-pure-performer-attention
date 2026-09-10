"""Shared scientific training controls used by every benchmark model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EarlyStoppingConfig:
    metric: str = "validation_roc_auc"
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "max"
    strict_improvement: bool = True

    def __post_init__(self) -> None:
        if (
            self.metric != "validation_roc_auc"
            or self.patience != 5
            or self.min_delta != 0.0
            or self.mode != "max"
            or not self.strict_improvement
        ):
            raise ValueError("the benchmark early-stopping protocol is fixed")

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


COMMON_EARLY_STOPPING = EarlyStoppingConfig()


@dataclass(frozen=True)
class EarlyStoppingDecision:
    improved: bool
    should_stop: bool
    best_epoch: int | None
    best_validation_auc: float | None
    consecutive_epochs_without_improvement: int


class ValidationAUCEarlyStopping:
    """Stop after exactly five consecutive non-improving validation epochs."""

    def __init__(self, config: EarlyStoppingConfig = COMMON_EARLY_STOPPING) -> None:
        self.config = config
        self.best_validation_auc: float | None = None
        self.best_epoch: int | None = None
        self.consecutive_epochs_without_improvement = 0

    def step(self, epoch: int, validation_auc: float) -> EarlyStoppingDecision:
        if epoch < 1:
            raise ValueError("epochs are one-based")
        finite = math.isfinite(validation_auc)
        improved = finite and (
            self.best_validation_auc is None
            or validation_auc > self.best_validation_auc + self.config.min_delta
        )
        if improved:
            self.best_validation_auc = float(validation_auc)
            self.best_epoch = int(epoch)
            self.consecutive_epochs_without_improvement = 0
        else:
            self.consecutive_epochs_without_improvement += 1
        return EarlyStoppingDecision(
            improved=improved,
            should_stop=(
                self.consecutive_epochs_without_improvement >= self.config.patience
            ),
            best_epoch=self.best_epoch,
            best_validation_auc=self.best_validation_auc,
            consecutive_epochs_without_improvement=self.consecutive_epochs_without_improvement,
        )
