"""Rolling-target data and train-only initialization for RKT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ktbench.data.session_store import SessionStore
from ktbench.rkt.phi import CrossFittedPhiRepository, HISTORY_LENGTH, _student_training_bounds


MILLISECONDS_PER_HOUR = 3_600_000.0


@dataclass(frozen=True)
class RKTExample:
    student_id: int
    history_question: Sequence[int]
    history_correct: Sequence[int]
    history_timestamp_ms: Sequence[int]
    target_question: int
    target_correct: int
    target_timestamp_ms: int
    target_event_id: int
    split: int


class RKTTargetDataset(Dataset[RKTExample]):
    """Every interaction in selected split sessions is exactly one target."""

    def __init__(
        self,
        store: SessionStore,
        split: str,
        *,
        selected_students: Iterable[int] | None = None,
    ) -> None:
        split_ids = {"train": 0, "validation": 1, "test": 2}
        if split not in split_ids:
            raise ValueError(f"unknown split: {split}")
        self.store = store
        self.split = split_ids[split]
        selected = None if selected_students is None else set(int(value) for value in selected_students)
        sessions = np.flatnonzero(store.session_split == self.split)
        if selected is not None:
            sessions = sessions[
                np.fromiter(
                    (int(store.session_student[index]) in selected for index in sessions),
                    dtype=np.bool_,
                    count=len(sessions),
                )
            ]
        self.sessions = sessions.astype(np.int64)
        lengths = store.session_offsets[self.sessions + 1] - store.session_offsets[self.sessions]
        self.cumulative_targets = np.cumsum(lengths, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.cumulative_targets[-1]) if len(self.cumulative_targets) else 0

    def _events_for_indices(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        segments = np.searchsorted(self.cumulative_targets, indices, side="right")
        previous = np.zeros(len(indices), dtype=np.int64)
        nonzero = segments > 0
        previous[nonzero] = self.cumulative_targets[segments[nonzero] - 1]
        sessions = self.sessions[segments]
        events = self.store.session_offsets[sessions] + indices - previous
        return sessions.astype(np.int64), events.astype(np.int64)

    def history_lengths(self, indices: np.ndarray) -> np.ndarray:
        sessions, events = self._events_for_indices(np.asarray(indices, dtype=np.int64))
        first_sessions = sessions - self.store.session_number[sessions].astype(np.int64) + 1
        first_events = self.store.session_offsets[first_sessions]
        return np.minimum(events - first_events, HISTORY_LENGTH).astype(np.int32)

    def __getitem__(self, index: int) -> RKTExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sessions, events = self._events_for_indices(np.asarray([index], dtype=np.int64))
        session = int(sessions[0])
        event = int(events[0])
        student = int(self.store.session_student[session])
        student_first_session = session - int(self.store.session_number[session]) + 1
        student_first_event = int(self.store.session_offsets[student_first_session])
        history_start = max(student_first_event, event - HISTORY_LENGTH)
        return RKTExample(
            student_id=student,
            history_question=self.store.question[history_start:event],
            history_correct=self.store.correct[history_start:event],
            history_timestamp_ms=self.store.timestamp_ms[history_start:event],
            target_question=int(self.store.question[event]),
            target_correct=int(self.store.correct[event]),
            target_timestamp_ms=int(self.store.timestamp_ms[event]),
            target_event_id=event,
            split=self.split,
        )


@dataclass
class RKTBatch:
    student_id: torch.Tensor
    history_question: torch.Tensor
    history_correct: torch.Tensor
    history_position: torch.Tensor
    history_mask: torch.Tensor
    delta_hours: torch.Tensor
    phi: torch.Tensor
    target_question: torch.Tensor
    target_labels: torch.Tensor
    target_event_id: torch.Tensor
    split: torch.Tensor

    def to(self, device: str | torch.device) -> "RKTBatch":
        return RKTBatch(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()
            }
        )


def collate_rkt(
    examples: Sequence[RKTExample], *, phi_repository: CrossFittedPhiRepository
) -> RKTBatch:
    if not examples:
        raise ValueError("cannot collate empty RKT batch")
    batch_size = len(examples)
    questions = torch.zeros(batch_size, HISTORY_LENGTH, dtype=torch.long)
    correctness = torch.zeros_like(questions)
    positions = torch.arange(1, HISTORY_LENGTH + 1).repeat(batch_size, 1)
    mask = torch.zeros(batch_size, HISTORY_LENGTH, dtype=torch.bool)
    delta = torch.zeros(batch_size, HISTORY_LENGTH, dtype=torch.float32)
    phi = torch.zeros_like(delta)
    prior_question_requests: list[np.ndarray] = []
    for row, example in enumerate(examples):
        length = len(example.history_question)
        if length > HISTORY_LENGTH:
            raise ValueError("RKT history exceeds approved rolling length")
        if length == 0:
            prior_question_requests.append(np.empty(0, dtype=np.int64))
            continue
        start = HISTORY_LENGTH - length
        history_question = np.asarray(example.history_question, dtype=np.int64)
        prior_question_requests.append(history_question)
        history_timestamp = np.asarray(example.history_timestamp_ms, dtype=np.int64)
        questions[row, start:] = torch.tensor(history_question)
        correctness[row, start:] = torch.tensor(example.history_correct, dtype=torch.long)
        mask[row, start:] = True
        differences = example.target_timestamp_ms - history_timestamp
        if np.any(differences < 0):
            raise ValueError("future timestamp found in RKT history")
        delta[row, start:] = torch.tensor(differences / MILLISECONDS_PER_HOUR, dtype=torch.float32)
    relations = phi_repository.lookup_grouped(
        (example.student_id for example in examples),
        (example.split for example in examples),
        (example.target_question for example in examples),
        prior_question_requests,
    )
    for row, relation in enumerate(relations):
        if len(relation):
            phi[row, HISTORY_LENGTH - len(relation) :] = torch.from_numpy(relation)
    return RKTBatch(
        student_id=torch.tensor([item.student_id for item in examples], dtype=torch.long),
        history_question=questions,
        history_correct=correctness,
        history_position=positions,
        history_mask=mask,
        delta_hours=delta,
        phi=phi,
        target_question=torch.tensor([item.target_question for item in examples], dtype=torch.long),
        target_labels=torch.tensor([item.target_correct for item in examples], dtype=torch.long),
        target_event_id=torch.tensor([item.target_event_id for item in examples], dtype=torch.long),
        split=torch.tensor([item.split for item in examples], dtype=torch.uint8),
    )


def build_memory_strength_initialization(
    store: SessionStore, output_root: str | Path
) -> dict[str, object]:
    """Persist per-student median positive train gaps and global fallback."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    student_ids = store.session_student[store.student_offsets[:-1]].astype(np.int32)
    positive_by_student: list[np.ndarray] = []
    medians = np.full(len(student_ids), np.nan, dtype=np.float64)
    for student_index in range(len(student_ids)):
        start, stop = _student_training_bounds(store, student_index)
        timestamp = np.asarray(store.timestamp_ms[start:stop], dtype=np.int64)
        gaps = np.diff(timestamp)
        positive = gaps[gaps > 0].astype(np.float64) / MILLISECONDS_PER_HOUR
        positive_by_student.append(positive)
        if len(positive):
            medians[student_index] = float(np.median(positive))
    all_positive = np.concatenate([item for item in positive_by_student if len(item)])
    if len(all_positive) == 0:
        raise ValueError("training data contains no positive timestamp gap")
    fallback = float(np.median(all_positive))
    used_fallback = np.isnan(medians)
    medians[used_fallback] = fallback
    table = pd.DataFrame(
        {
            "student_id": student_ids,
            "initial_s_hours": medians,
            "used_global_fallback": used_fallback,
        }
    )
    table.to_csv(output_root / "student_memory_initialization.csv", index=False)
    values = np.full(int(student_ids.max()) + 1, fallback, dtype=np.float32)
    values[student_ids] = medians.astype(np.float32)
    np.save(output_root / "initial_s_hours.npy", values)
    report: dict[str, object] = {
        "timestamp_input_unit": "milliseconds",
        "conversion": "delta_hours = (target_timestamp_ms - history_timestamp_ms) / 3,600,000",
        "source": "positive consecutive gaps within training sessions only",
        "global_training_only_fallback_hours": fallback,
        "students": len(student_ids),
        "students_using_fallback": int(used_fallback.sum()),
        "validation_or_test_used": False,
    }
    (output_root / "memory_initialization.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
