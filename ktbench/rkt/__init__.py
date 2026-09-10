"""RKT paper-faithful performance-only Phi-relation variant."""

from .data import (
    RKTBatch,
    RKTExample,
    RKTTargetDataset,
    build_memory_strength_initialization,
    collate_rkt,
)
from .phi import (
    CrossFittedPhiRepository,
    SparsePhiMatrix,
    assign_student_folds,
    build_cross_fitted_phi_cache,
)

__all__ = [
    "CrossFittedPhiRepository",
    "RKTBatch",
    "RKTExample",
    "RKTTargetDataset",
    "SparsePhiMatrix",
    "assign_student_folds",
    "build_cross_fitted_phi_cache",
    "build_memory_strength_initialization",
    "collate_rkt",
]
