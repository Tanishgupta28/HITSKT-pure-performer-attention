"""Shared scientific training controls used by every benchmark model."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


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

def capture_rng_state() -> dict[str, object]:
    """Capture every project RNG in a weights-only-loadable representation."""

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, object]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, dict):
        raise ValueError("invalid NumPy RNG checkpoint state")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["state"].cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA RNG checkpoint device count mismatch")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


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

    @property
    def should_stop(self) -> bool:
        """Also honor a stopping state restored before final test evaluation."""

        return self.consecutive_epochs_without_improvement >= self.config.patience

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
            should_stop=self.should_stop,
            best_epoch=self.best_epoch,
            best_validation_auc=self.best_validation_auc,
            consecutive_epochs_without_improvement=self.consecutive_epochs_without_improvement,
        )
