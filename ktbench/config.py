"""Project-wide reproducibility configuration."""

from __future__ import annotations

import os
import random


PROJECT_SEED = 42

# Must be set before the first CUDA BLAS operation in the process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", str(PROJECT_SEED))

import numpy as np
import torch

def seed_everything(seed: int = PROJECT_SEED, *, deterministic: bool = True) -> None:
    """Seed every RNG used by the benchmark from one centralized value."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
