"""Leakage-safe rolling interaction targets for non-hierarchical KT baselines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ktbench.config import PROJECT_SEED
from ktbench.data.session_store import SessionStore


@dataclass(frozen=True)
class RollingTarget:
    student_id: int
    history_question: Sequence[int]
    history_skill: Sequence[int]
    history_correct: Sequence[int]
    history_timestamp_ms: Sequence[int]
    target_question: int
    target_skill: int
    target_correct: int
    target_timestamp_ms: int
    target_event_id: int
    split: int


class RollingTargetDataset(Dataset[RollingTarget]):
    """Expose each event in one split exactly once with strictly prior history."""

    SPLITS = {"train": 0, "validation": 1, "test": 2}

    def __init__(
        self,
        store: SessionStore,
        split: str,
        *,
        history_length: int,
        selected_students: Iterable[int] | None = None,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"unknown split: {split}")
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.store = store
        self.split = self.SPLITS[split]
        self.history_length = int(history_length)
        selected = (
            None
            if selected_students is None
            else set(int(value) for value in selected_students)
        )
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
        return np.minimum(events - first_events, self.history_length).astype(np.int32)

    def __getitem__(self, index: int) -> RollingTarget:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sessions, events = self._events_for_indices(np.asarray([index], dtype=np.int64))
        session = int(sessions[0])
        event = int(events[0])
        student = int(self.store.session_student[session])
        first_session = session - int(self.store.session_number[session]) + 1
        first_event = int(self.store.session_offsets[first_session])
        history_start = max(first_event, event - self.history_length)
        return RollingTarget(
            student_id=student,
            history_question=self.store.question[history_start:event],
            history_skill=self.store.skill[history_start:event],
            history_correct=self.store.correct[history_start:event],
            history_timestamp_ms=self.store.timestamp_ms[history_start:event],
            target_question=int(self.store.question[event]),
            target_skill=int(self.store.skill[event]),
            target_correct=int(self.store.correct[event]),
            target_timestamp_ms=int(self.store.timestamp_ms[event]),
            target_event_id=event,
            split=self.split,
        )


@dataclass
class RollingTargetBatch:
    student_id: torch.Tensor
    history_question: torch.Tensor
    history_skill: torch.Tensor
    history_correct: torch.Tensor
    history_mask: torch.Tensor
    target_question: torch.Tensor
    target_skill: torch.Tensor
    target_labels: torch.Tensor
    target_event_id: torch.Tensor
    split: torch.Tensor

    def to(self, device: str | torch.device) -> "RollingTargetBatch":
        return RollingTargetBatch(
            **{name: value.to(device) for name, value in vars(self).items()}
        )

    def pin_memory(self) -> "RollingTargetBatch":
        return RollingTargetBatch(
            **{name: value.pin_memory() for name, value in vars(self).items()}
        )


def collate_rolling_targets(examples: Sequence[RollingTarget]) -> RollingTargetBatch:
    """Dynamically left-pad histories to the longest history in this batch."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    width = max(1, max(len(example.history_question) for example in examples))
    batch_size = len(examples)
    questions = torch.zeros(batch_size, width, dtype=torch.long)
    skills = torch.zeros_like(questions)
    correctness = torch.zeros_like(questions)
    mask = torch.zeros(batch_size, width, dtype=torch.bool)
    for row, example in enumerate(examples):
        length = len(example.history_question)
        if not (
            len(example.history_skill)
            == len(example.history_correct)
            == len(example.history_timestamp_ms)
            == length
        ):
            raise ValueError("unaligned rolling history")
        if length == 0:
            continue
        start = width - length
        timestamps = np.asarray(example.history_timestamp_ms, dtype=np.int64)
        if np.any(timestamps > example.target_timestamp_ms):
            raise ValueError("future interaction found in rolling history")
        questions[row, start:] = torch.tensor(
            np.asarray(example.history_question), dtype=torch.long
        )
        skills[row, start:] = torch.tensor(
            np.asarray(example.history_skill), dtype=torch.long
        )
        correctness[row, start:] = torch.tensor(
            np.asarray(example.history_correct), dtype=torch.long
        )
        mask[row, start:] = True
    return RollingTargetBatch(
        student_id=torch.tensor([item.student_id for item in examples], dtype=torch.long),
        history_question=questions,
        history_skill=skills,
        history_correct=correctness,
        history_mask=mask,
        target_question=torch.tensor(
            [item.target_question for item in examples], dtype=torch.long
        ),
        target_skill=torch.tensor([item.target_skill for item in examples], dtype=torch.long),
        target_labels=torch.tensor([item.target_correct for item in examples], dtype=torch.long),
        target_event_id=torch.tensor(
            [item.target_event_id for item in examples], dtype=torch.long
        ),
        split=torch.tensor([item.split for item in examples], dtype=torch.uint8),
    )


class RollingLengthBatchSampler(Sampler[list[int]]):
    """Bounded-memory deterministic length bucketing for very large target sets."""

    def __init__(
        self,
        dataset: RollingTargetDataset,
        *,
        batch_size: int,
        bucket_size: int = 4096,
        shuffle: bool = True,
        seed: int = PROJECT_SEED,
    ) -> None:
        if batch_size <= 0 or bucket_size < batch_size:
            raise ValueError("invalid batch or bucket size")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        starts = list(range(0, len(self.dataset), self.bucket_size))
        py_rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            py_rng.shuffle(starts)
        numpy_rng = np.random.default_rng(self.seed + self.epoch)
        for start in starts:
            stop = min(start + self.bucket_size, len(self.dataset))
            indices = np.arange(start, stop, dtype=np.int64)
            lengths = self.dataset.history_lengths(indices)
            if self.shuffle:
                tie_keys = numpy_rng.random(len(indices))
                order = np.lexsort((tie_keys, lengths))
            else:
                order = np.argsort(lengths, kind="stable")
            ordered = indices[order]
            for batch_start in range(0, len(ordered), self.batch_size):
                yield ordered[batch_start : batch_start + self.batch_size].tolist()
