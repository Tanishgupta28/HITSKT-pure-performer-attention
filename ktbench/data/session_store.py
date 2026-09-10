"""Memory-mapped, lossless session storage for hierarchical KT batches."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset

from ktbench.batching import (
    ExampleShape,
    LengthBucketTokenBatchSampler,
    Session,
    SessionExample,
    collate_hitskt,
)


SPLIT_NAMES = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class SessionStoreMetadata:
    format_version: int
    interactions: int
    sessions: int
    students: int
    num_questions: int
    num_skills: int
    maximum_session_length: int
    split_sessions: dict[str, int]
    action_truncation: bool = False
    action_chunking: bool = False


def _open_array(path: Path, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def build_session_store(processed_root: str | Path, output_root: str | Path) -> SessionStoreMetadata:
    """Convert validated ordered Parquet events into lossless memory maps."""

    processed_root = Path(processed_root)
    output_root = Path(output_root)
    summary_path = processed_root / "reports" / "summary.json"
    if not (processed_root / "_SUCCESS").exists() or not summary_path.exists():
        raise ValueError(f"processed dataset is incomplete: {processed_root}")
    summary = json.loads(summary_path.read_text())
    interactions = int(summary["retained_interactions"])
    sessions = int(summary["retained_sessions"])
    students = int(summary["retained_students"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_SUCCESS").unlink(missing_ok=True)

    question = _open_array(output_root / "question.npy", "int32", (interactions,))
    skill = _open_array(output_root / "skill.npy", "int32", (interactions,))
    correct = _open_array(output_root / "correct.npy", "uint8", (interactions,))
    session_offsets = _open_array(output_root / "session_offsets.npy", "int64", (sessions + 1,))
    session_student = _open_array(output_root / "session_student.npy", "int32", (sessions,))
    session_number = _open_array(output_root / "session_number.npy", "int32", (sessions,))
    session_split = _open_array(output_root / "session_split.npy", "uint8", (sessions,))

    event_cursor = 0
    session_cursor = 0
    maximum_question = 0
    maximum_skill = 0
    previous_student = None
    previous_session_number = 0
    event_files = sorted((processed_root / "events").glob("*.parquet"))
    if not event_files:
        raise ValueError("processed dataset has no event shards")
    columns = [
        "student_id",
        "question_id",
        "skill_id",
        "correct",
        "session_id",
        "session_position",
        "split",
    ]
    for event_file in event_files:
        parquet = pq.ParquetFile(event_file)
        for record_batch in parquet.iter_batches(batch_size=262_144, columns=columns):
            values = record_batch.to_pydict()
            rows = record_batch.num_rows
            stop = event_cursor + rows
            if stop > interactions:
                raise ValueError("event count exceeds preprocessing summary")
            q_values = np.asarray(values["question_id"], dtype=np.int32)
            s_values = np.asarray(values["skill_id"], dtype=np.int32)
            c_values = np.asarray(values["correct"], dtype=np.uint8)
            question[event_cursor:stop] = q_values
            skill[event_cursor:stop] = s_values
            correct[event_cursor:stop] = c_values
            maximum_question = max(maximum_question, int(q_values.max(initial=0)))
            maximum_skill = max(maximum_skill, int(s_values.max(initial=0)))

            for local_index, position in enumerate(values["session_position"]):
                if position != 1:
                    continue
                if session_cursor >= sessions:
                    raise ValueError("session count exceeds preprocessing summary")
                student = int(values["student_id"][local_index])
                number = int(values["session_id"][local_index])
                if student == previous_student:
                    if number != previous_session_number + 1:
                        raise ValueError("non-contiguous session numbering within student")
                elif number != 1:
                    raise ValueError("first session for student is not numbered one")
                session_offsets[session_cursor] = event_cursor + local_index
                session_student[session_cursor] = student
                session_number[session_cursor] = number
                session_split[session_cursor] = int(values["split"][local_index])
                previous_student = student
                previous_session_number = number
                session_cursor += 1
            event_cursor = stop

    if event_cursor != interactions or session_cursor != sessions:
        raise ValueError(
            f"store count mismatch: events {event_cursor}/{interactions}, "
            f"sessions {session_cursor}/{sessions}"
        )
    session_offsets[sessions] = interactions
    session_lengths = np.diff(session_offsets)
    if np.any(session_lengths <= 0):
        raise ValueError("empty session encountered")
    student_starts = np.flatnonzero(session_number == 1).astype(np.int64)
    if len(student_starts) != students:
        raise ValueError(f"student count mismatch: {len(student_starts)}/{students}")
    student_offsets = _open_array(output_root / "student_offsets.npy", "int64", (students + 1,))
    student_offsets[:-1] = student_starts
    student_offsets[-1] = sessions

    split_counts = {
        name: int(np.count_nonzero(session_split == value)) for name, value in SPLIT_NAMES.items()
    }
    metadata = SessionStoreMetadata(
        format_version=1,
        interactions=interactions,
        sessions=sessions,
        students=students,
        num_questions=maximum_question,
        num_skills=maximum_skill,
        maximum_session_length=int(session_lengths.max()),
        split_sessions=split_counts,
    )
    (output_root / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n")
    (output_root / "_SUCCESS").write_text("complete\n")
    for array in (
        question,
        skill,
        correct,
        session_offsets,
        session_student,
        session_number,
        session_split,
        student_offsets,
    ):
        array.flush()
    return metadata


class SessionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not (self.root / "_SUCCESS").exists():
            raise ValueError(f"incomplete session store: {self.root}")
        self.metadata: dict[str, Any] = json.loads((self.root / "metadata.json").read_text())
        self.question = np.load(self.root / "question.npy", mmap_mode="r")
        self.skill = np.load(self.root / "skill.npy", mmap_mode="r")
        self.correct = np.load(self.root / "correct.npy", mmap_mode="r")
        self.session_offsets = np.load(self.root / "session_offsets.npy", mmap_mode="r")
        self.session_student = np.load(self.root / "session_student.npy", mmap_mode="r")
        self.session_number = np.load(self.root / "session_number.npy", mmap_mode="r")
        self.session_split = np.load(self.root / "session_split.npy", mmap_mode="r")
        self.student_offsets = np.load(self.root / "student_offsets.npy", mmap_mode="r")

    def session(self, index: int) -> Session:
        start, stop = (int(value) for value in self.session_offsets[index : index + 2])
        return Session(self.question[start:stop], self.skill[start:stop], self.correct[start:stop])

    def session_length(self, index: int) -> int:
        return int(self.session_offsets[index + 1] - self.session_offsets[index])


class RollingSessionDataset(Dataset[SessionExample]):
    """Each post-first session is a target with chronological rolling history."""

    def __init__(
        self,
        store: SessionStore,
        split: str,
        *,
        history_sessions: int | None = 15,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split: {split}")
        if history_sessions is not None and history_sessions <= 0:
            raise ValueError("history_sessions must be positive or None")
        split_id = SPLIT_NAMES[split]
        self.store = store
        self.split = split_id
        self.history_sessions = history_sessions
        self.targets = np.flatnonzero(
            (store.session_split == split_id) & (store.session_number > 1)
        ).astype(np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def _history_bounds(self, target: int) -> tuple[int, int]:
        student_start = target - int(self.store.session_number[target]) + 1
        history_start = student_start
        if self.history_sessions is not None:
            history_start = max(history_start, target - self.history_sessions)
        return history_start, target

    def __getitem__(self, index: int) -> SessionExample:
        target_index = int(self.targets[index])
        history_start, history_stop = self._history_bounds(target_index)
        history = [self.store.session(item) for item in range(history_start, history_stop)]
        return SessionExample(history, self.store.session(target_index), self.split)

    def shape(self, index: int) -> ExampleShape:
        target_index = int(self.targets[index])
        history_start, history_stop = self._history_bounds(target_index)
        offsets = self.store.session_offsets
        history_lengths = tuple(
            int(offsets[item + 1] - offsets[item]) for item in range(history_start, history_stop)
        )
        return ExampleShape(history_lengths, self.store.session_length(target_index))


class DatasetShapeSequence(Sequence[ExampleShape]):
    """Lazy shape adapter accepted by the token-budget sampler."""

    def __init__(self, dataset: RollingSessionDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | slice) -> ExampleShape | list[ExampleShape]:
        if isinstance(index, slice):
            return [self.dataset.shape(item) for item in range(*index.indices(len(self)))]
        return self.dataset.shape(index)

    def sort_lengths(self) -> np.ndarray:
        """Vector of max action lengths used for fast stable bucketing."""

        result = np.empty(len(self.dataset), dtype=np.int32)
        offsets = self.dataset.store.session_offsets
        numbers = self.dataset.store.session_number
        for output_index, raw_target in enumerate(self.dataset.targets):
            target = int(raw_target)
            student_start = target - int(numbers[target]) + 1
            history_start = student_start
            if self.dataset.history_sessions is not None:
                history_start = max(history_start, target - self.dataset.history_sessions)
            starts = offsets[history_start : target + 1]
            history_lengths = starts[1:] - starts[:-1]
            target_length = int(offsets[target + 1] - offsets[target])
            result[output_index] = max(int(history_lengths.max(initial=0)), target_length) + 1
        return result


def make_session_dataloader(
    dataset: RollingSessionDataset,
    *,
    token_budget: int = 32_768,
    maximum_batch_size: int = 64,
    bucket_size: int = 512,
    shuffle: bool = True,
    seed: int = 0,
    workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader[SessionExample], LengthBucketTokenBatchSampler]:
    """Create the production dynamic DataLoader and expose its epoch sampler."""

    store = dataset.store
    sampler = LengthBucketTokenBatchSampler(
        DatasetShapeSequence(dataset),
        token_budget=token_budget,
        max_batch_size=maximum_batch_size,
        bucket_size=bucket_size,
        shuffle=shuffle,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(
            collate_hitskt,
            num_questions=int(store.metadata["num_questions"]),
            num_skills=int(store.metadata["num_skills"]),
        ),
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return loader, sampler
