"""Leakage-safe, cached directed Phi relations for the RKT reproduction."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from numba import njit

from ktbench.config import PROJECT_SEED
from ktbench.data.session_store import SessionStore


FOLD_COUNT = 5
HISTORY_LENGTH = 49


def assign_student_folds(
    student_ids: Iterable[int], *, seed: int = PROJECT_SEED, folds: int = FOLD_COUNT
) -> pd.DataFrame:
    """Seeded permutation plus round-robin assignment, balanced within one."""

    values = np.asarray(sorted(set(int(value) for value in student_ids)), dtype=np.int32)
    if len(values) < folds:
        raise ValueError("student population is smaller than fold count")
    generator = np.random.default_rng(seed)
    shuffled = values[generator.permutation(len(values))]
    assignments = np.empty(len(values), dtype=np.uint8)
    fold_for_student = {int(student): index % folds for index, student in enumerate(shuffled)}
    for index, student in enumerate(values):
        assignments[index] = fold_for_student[int(student)]
    return pd.DataFrame({"student_id": values, "fold": assignments})


@njit(cache=True)
def _student_pair_outcomes(
    questions: np.ndarray,
    correctness: np.ndarray,
    vocabulary_size: int,
    history_length: int,
) -> np.ndarray:
    """Encode one contribution per latest distinct prior question and target."""

    length = len(questions)
    next_same = np.full(length, -1, dtype=np.int64)
    last = np.full(vocabulary_size + 1, -1, dtype=np.int64)
    for index in range(length - 1, -1, -1):
        question = int(questions[index])
        next_same[index] = last[question]
        last[question] = index
    output = np.empty(max(1, length * history_length), dtype=np.int64)
    cursor = 0
    base = vocabulary_size + 1
    for target in range(length):
        start = max(0, target - history_length)
        target_question = int(questions[target])
        target_correct = int(correctness[target])
        for prior in range(start, target):
            # A prior occurrence is used only if the same question does not
            # occur again between it and this target.
            if next_same[prior] != -1 and next_same[prior] < target:
                continue
            prior_question = int(questions[prior])
            prior_correct = int(correctness[prior])
            pair = target_question * base + prior_question
            outcome = prior_correct * 2 + target_correct
            output[cursor] = pair * 4 + outcome
            cursor += 1
    return output[:cursor]


class _SparseAccumulator:
    def __init__(self, root: Path, chunk_values: int) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunk_values = chunk_values
        self.buffers: list[np.ndarray] = []
        self.buffered = 0
        self.parts: list[Path] = []

    def add(self, values: np.ndarray) -> None:
        if len(values) == 0:
            return
        self.buffers.append(values)
        self.buffered += len(values)
        if self.buffered >= self.chunk_values:
            self.flush()

    def flush(self) -> None:
        if not self.buffers:
            return
        values = np.concatenate(self.buffers)
        codes, counts = np.unique(values, return_counts=True)
        path = self.root / f"part-{len(self.parts):05d}.npz"
        np.savez(path, codes=codes, counts=counts.astype(np.int64))
        self.parts.append(path)
        self.buffers.clear()
        self.buffered = 0

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        self.flush()
        if not self.parts:
            return np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.int64)
        all_codes = []
        all_counts = []
        for part in self.parts:
            values = np.load(part)
            all_codes.append(values["codes"])
            all_counts.append(values["counts"])
        codes = np.concatenate(all_codes)
        weights = np.concatenate(all_counts)
        order = np.argsort(codes, kind="stable")
        codes = codes[order]
        weights = weights[order]
        unique, starts = np.unique(codes, return_index=True)
        totals = np.add.reduceat(weights, starts)
        pair_keys = unique // 4
        outcomes = unique % 4
        pairs, pair_starts = np.unique(pair_keys, return_index=True)
        counts = np.zeros((len(pairs), 4), dtype=np.int64)
        pair_rows = np.searchsorted(pairs, pair_keys)
        counts[pair_rows, outcomes] = totals
        return pairs, counts


def _merge_fold_counts(
    fold_data: list[tuple[np.ndarray, np.ndarray]], excluded_fold: int | None
) -> tuple[np.ndarray, np.ndarray]:
    selected = [data for fold, data in enumerate(fold_data) if fold != excluded_fold]
    if not selected:
        return np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.int64)
    keys = np.unique(np.concatenate([item[0] for item in selected]))
    counts = np.zeros((len(keys), 4), dtype=np.int64)
    for item_keys, item_counts in selected:
        counts[np.searchsorted(keys, item_keys)] += item_counts
    keep = counts.any(axis=1)
    return keys[keep], counts[keep]


def _student_training_bounds(store: SessionStore, student_index: int) -> tuple[int, int]:
    session_start = int(store.student_offsets[student_index])
    session_stop = int(store.student_offsets[student_index + 1])
    sessions = np.arange(session_start, session_stop)
    training = sessions[store.session_split[sessions] == 0]
    if len(training) == 0:
        return 0, 0
    return int(store.session_offsets[training[0]]), int(store.session_offsets[training[-1] + 1])


def build_cross_fitted_phi_cache(
    store: SessionStore,
    output_root: str | Path,
    *,
    seed: int = PROJECT_SEED,
    selected_students: Iterable[int] | None = None,
    chunk_values: int = 5_000_000,
) -> dict[str, object]:
    """Build five fold-own counts and cache complements plus all-train Phi."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_SUCCESS").unlink(missing_ok=True)
    scratch = output_root / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    student_ids = store.session_student[store.student_offsets[:-1]].astype(np.int32)
    fold_mapping = assign_student_folds(student_ids, seed=seed)
    fold_mapping.to_csv(output_root / "student_folds.csv", index=False)
    fold_by_student = np.full(int(student_ids.max()) + 1, 255, dtype=np.uint8)
    fold_by_student[fold_mapping["student_id"].to_numpy()] = fold_mapping["fold"].to_numpy()
    selected = None if selected_students is None else set(int(value) for value in selected_students)
    accumulators = [_SparseAccumulator(scratch / f"fold-{fold}", chunk_values) for fold in range(FOLD_COUNT)]
    included_students = 0
    training_targets = 0
    vocabulary_size = int(store.metadata["num_questions"])
    for student_index, student in enumerate(student_ids):
        student_value = int(student)
        if selected is not None and student_value not in selected:
            continue
        start, stop = _student_training_bounds(store, student_index)
        if stop <= start:
            continue
        codes = _student_pair_outcomes(
            np.asarray(store.question[start:stop], dtype=np.int64),
            np.asarray(store.correct[start:stop], dtype=np.uint8),
            vocabulary_size,
            HISTORY_LENGTH,
        )
        accumulators[int(fold_by_student[student_value])].add(codes)
        included_students += 1
        training_targets += stop - start

    fold_data = [accumulator.finish() for accumulator in accumulators]
    for fold, (keys, counts) in enumerate(fold_data):
        np.savez_compressed(output_root / f"counts_fold_{fold}.npz", keys=keys, counts=counts)
    all_keys, all_counts = _merge_fold_counts(fold_data, None)
    np.savez_compressed(output_root / "phi_all_train.npz", keys=all_keys, counts=all_counts)
    for fold in range(FOLD_COUNT):
        keys, counts = _merge_fold_counts(fold_data, fold)
        np.savez_compressed(output_root / f"phi_excluding_fold_{fold}.npz", keys=keys, counts=counts)
    shutil.rmtree(scratch)

    fold_counts = fold_mapping.groupby("fold").size().reindex(range(FOLD_COUNT), fill_value=0)
    split_targets = {}
    session_lengths = np.diff(store.session_offsets)
    for name, split in (("train", 0), ("validation", 1), ("test", 2)):
        split_targets[name] = int(session_lengths[store.session_split == split].sum())
    metadata: dict[str, object] = {
        "format_version": 1,
        "label": "RKT paper-faithful performance-only Phi-relation variant with 5-fold student-level cross-fitting.",
        "seed": seed,
        "fold_count": FOLD_COUNT,
        "fold_assignment": "seeded NumPy permutation of sorted student IDs, then round-robin index modulo 5",
        "students_per_fold": {str(index): int(value) for index, value in fold_counts.items()},
        "folds_balanced_within_one": int(fold_counts.max() - fold_counts.min()) <= 1,
        "cache_students": included_students,
        "training_targets_cross_fitted": training_targets,
        "all_dataset_targets": split_targets,
        "history_length": HISTORY_LENGTH,
        "phi_source": "training-partition histories only",
        "training_rule": "target student's entire fold excluded",
        "validation_test_rule": "all training folds; no validation/test interactions",
        "undefined_relation": 0.0,
        "threshold": None,
        "text_relation": None,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_root / "_SUCCESS").write_text("complete\n")
    return metadata


@dataclass(frozen=True)
class SparsePhiMatrix:
    keys: np.ndarray
    counts: np.ndarray
    vocabulary_size: int

    @classmethod
    def load(cls, path: str | Path, vocabulary_size: int) -> "SparsePhiMatrix":
        values = np.load(path)
        return cls(values["keys"], values["counts"], vocabulary_size)

    def lookup(self, target_questions: np.ndarray, prior_questions: np.ndarray) -> np.ndarray:
        pairs = target_questions.astype(np.int64) * (self.vocabulary_size + 1) + prior_questions
        positions = np.searchsorted(self.keys, pairs)
        present = positions < len(self.keys)
        present[present] &= self.keys[positions[present]] == pairs[present]
        result = np.zeros(pairs.shape, dtype=np.float32)
        if not present.any():
            return result
        values = self.counts[positions[present]].astype(np.float64)
        n00, n01, n10, n11 = (values[:, index] for index in range(4))
        denominator = np.sqrt((n10 + n11) * (n00 + n01) * (n01 + n11) * (n00 + n10))
        defined = denominator > 0
        phi = np.zeros(len(values), dtype=np.float64)
        phi[defined] = (n11[defined] * n00[defined] - n01[defined] * n10[defined]) / denominator[defined]
        result[present] = phi.astype(np.float32)
        return result


class CrossFittedPhiRepository:
    def __init__(self, root: str | Path, vocabulary_size: int) -> None:
        self.root = Path(root)
        if not (self.root / "_SUCCESS").exists():
            raise ValueError("incomplete Phi cache")
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        self.folds = pd.read_csv(self.root / "student_folds.csv")
        maximum_student = int(self.folds["student_id"].max())
        self.fold_by_student = np.full(maximum_student + 1, 255, dtype=np.uint8)
        self.fold_by_student[self.folds["student_id"]] = self.folds["fold"]
        self.all_train = SparsePhiMatrix.load(
            self.root / "phi_all_train.npz", vocabulary_size
        )
        self.excluding = [
            SparsePhiMatrix.load(self.root / f"phi_excluding_fold_{fold}.npz", vocabulary_size)
            for fold in range(FOLD_COUNT)
        ]

    def lookup(
        self,
        student_id: int,
        split: int,
        target_questions: np.ndarray,
        prior_questions: np.ndarray,
    ) -> np.ndarray:
        matrix = self.all_train
        if split == 0:
            fold = int(self.fold_by_student[student_id])
            if fold >= FOLD_COUNT:
                raise ValueError(f"student {student_id} has no persisted fold")
            matrix = self.excluding[fold]
        return matrix.lookup(target_questions, prior_questions)
