#!/usr/bin/env python3
"""Prepare deterministic RKT fold, Phi, and S_u initialization artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.session_store import SessionStore
from ktbench.rkt.data import build_memory_strength_initialization
from ktbench.rkt.phi import assign_student_folds, build_cross_fitted_phi_cache


RAW_TIMESTAMP_PROVENANCE = {
    "assist2017": {
        "native_unit": "Unix seconds",
        "evidence": "raw startTime 1096470301 maps to processed 1096470301000",
        "preprocessing_conversion": "processed_timestamp_ms = raw_startTime * 1000",
    },
    "junyi": {
        "native_unit": "Unix seconds",
        "evidence": "raw startTime 1536849900 maps to processed 1536849900000",
        "preprocessing_conversion": "processed_timestamp_ms = raw_startTime * 1000",
    },
    "ednet_kt1": {
        "native_unit": "Unix milliseconds shifted for privacy",
        "evidence": "official KT1/u1.csv begins 1565096190868 and is preserved",
        "preprocessing_conversion": "processed_timestamp_ms = raw_timestamp",
    },
}


def _smoke_students(store: SessionStore, per_fold: int) -> np.ndarray:
    students = store.session_student[store.student_offsets[:-1]].astype(np.int32)
    mapping = assign_student_folds(students, seed=PROJECT_SEED)
    generator = np.random.default_rng(PROJECT_SEED)
    selected = []
    for fold in range(5):
        candidates = mapping.loc[mapping["fold"] == fold, "student_id"].to_numpy()
        order = generator.permutation(len(candidates))
        selected.extend(candidates[order[:per_fold]].tolist())
    return np.asarray(sorted(selected), dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(RAW_TIMESTAMP_PROVENANCE))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--smoke-students-per-fold", type=int)
    args = parser.parse_args()
    seed_everything()
    store = SessionStore(args.store_root)
    selected = None
    if args.smoke_students_per_fold is not None:
        if args.smoke_students_per_fold <= 0:
            raise ValueError("smoke student count must be positive")
        selected = _smoke_students(store, args.smoke_students_per_fold)
        args.output_root.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.output_root / "selected_students.csv", selected, fmt="%d", header="student_id", comments="")
    phi = build_cross_fitted_phi_cache(
        store,
        args.output_root,
        seed=PROJECT_SEED,
        selected_students=selected,
    )
    memory = build_memory_strength_initialization(store, args.output_root)
    provenance = {
        "dataset": args.dataset,
        "seed": PROJECT_SEED,
        "raw_timestamp": RAW_TIMESTAMP_PROVENANCE[args.dataset],
        "model_conversion": "delta_hours = (target_timestamp_ms - history_timestamp_ms) / 3,600,000",
        "phi": phi,
        "memory_strength": memory,
        "selected_smoke_students": None if selected is None else len(selected),
    }
    (args.output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
