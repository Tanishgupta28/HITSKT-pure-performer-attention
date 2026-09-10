"""Variable-length hierarchical examples and token-budgeted batching."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from ktbench.config import PROJECT_SEED


PAD_ID = 0
CORRECT_PAD_ID = 2
CORRECT_EOS_ID = 3
CORRECT_BOS_ID = 4


@dataclass(frozen=True)
class Session:
    question_ids: Sequence[int]
    skill_ids: Sequence[int]
    correct: Sequence[int]

    def __post_init__(self) -> None:
        length = len(self.question_ids)
        if length == 0:
            raise ValueError("a real session cannot be empty")
        if len(self.skill_ids) != length or len(self.correct) != length:
            raise ValueError("session fields have different lengths")
        if any(value not in (0, 1) for value in self.correct):
            raise ValueError("session correctness must be binary")

    def __len__(self) -> int:
        return len(self.question_ids)


@dataclass(frozen=True)
class SessionExample:
    history: Sequence[Session]
    target: Session
    split: int

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("HiTSKT examples require at least one earlier session")
        if self.split not in (0, 1, 2):
            raise ValueError("split must be 0, 1, or 2")


@dataclass(frozen=True)
class ExampleShape:
    history_lengths: tuple[int, ...]
    target_length: int

    @property
    def sort_length(self) -> int:
        return max((*self.history_lengths, self.target_length)) + 1


def example_shape(example: SessionExample) -> ExampleShape:
    return ExampleShape(tuple(len(session) for session in example.history), len(example.target))


def padded_token_cost(shapes: Sequence[ExampleShape]) -> int:
    """Exact token slots produced by the packed/dynamic collator.

    One EOS token is included for every real session. Historical sessions are
    flattened, so only those sessions—not absent hierarchy slots—pay action
    padding. The session encoder adds one EOS vector per example.
    """

    if not shapes:
        return 0
    history_sessions = sum(len(shape.history_lengths) for shape in shapes)
    max_history_actions = max(
        length + 1 for shape in shapes for length in shape.history_lengths
    )
    max_target_actions = max(shape.target_length + 1 for shape in shapes)
    max_history_count = max(len(shape.history_lengths) + 1 for shape in shapes)
    return (
        history_sessions * max_history_actions
        + len(shapes) * max_target_actions
        + len(shapes) * max_history_count
    )


class LengthBucketTokenBatchSampler(Sampler[list[int]]):
    """Bucket similar lengths, then greedily respect a padded-token budget."""

    def __init__(
        self,
        shapes: Sequence[ExampleShape],
        *,
        token_budget: int,
        max_batch_size: int,
        bucket_size: int = 512,
        shuffle: bool = True,
        seed: int = PROJECT_SEED,
    ) -> None:
        if not shapes:
            raise ValueError("shapes cannot be empty")
        if token_budget <= 0 or max_batch_size <= 0 or bucket_size <= 0:
            raise ValueError("batching limits must be positive")
        self.shapes = shapes
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _ordered_indices(self) -> list[int]:
        if hasattr(self.shapes, "sort_lengths"):
            lengths = np.asarray(self.shapes.sort_lengths())
            indices = np.argsort(lengths, kind="stable").tolist()
        else:
            indices = sorted(
                range(len(self.shapes)), key=lambda index: self.shapes[index].sort_length
            )
        buckets = [indices[start : start + self.bucket_size] for start in range(0, len(indices), self.bucket_size)]
        if self.shuffle:
            generator = random.Random(self.seed + self.epoch)
            for bucket in buckets:
                generator.shuffle(bucket)
            generator.shuffle(buckets)
        return [index for bucket in buckets for index in bucket]

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        history_sessions = 0
        max_history_actions = 0
        max_target_actions = 0
        max_history_count = 0
        for index in self._ordered_indices():
            shape = self.shapes[index]
            proposed_size = len(batch) + 1
            proposed_history_sessions = history_sessions + len(shape.history_lengths)
            proposed_max_history_actions = max(
                max_history_actions, max(length + 1 for length in shape.history_lengths)
            )
            proposed_max_target_actions = max(max_target_actions, shape.target_length + 1)
            proposed_max_history_count = max(
                max_history_count, len(shape.history_lengths) + 1
            )
            proposed_cost = (
                proposed_history_sessions * proposed_max_history_actions
                + proposed_size * proposed_max_target_actions
                + proposed_size * proposed_max_history_count
            )
            if batch and (
                proposed_size > self.max_batch_size or proposed_cost > self.token_budget
            ):
                yield batch
                batch = [index]
                history_sessions = len(shape.history_lengths)
                max_history_actions = max(length + 1 for length in shape.history_lengths)
                max_target_actions = shape.target_length + 1
                max_history_count = len(shape.history_lengths) + 1
            else:
                batch.append(index)
                history_sessions = proposed_history_sessions
                max_history_actions = proposed_max_history_actions
                max_target_actions = proposed_max_target_actions
                max_history_count = proposed_max_history_count
        if batch:
            yield batch

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))


@dataclass
class HiTSKTBatch:
    history_question: torch.Tensor
    history_skill: torch.Tensor
    history_correct: torch.Tensor
    history_mask: torch.Tensor
    history_owner: torch.Tensor
    history_order: torch.Tensor
    history_count: torch.Tensor
    target_question: torch.Tensor
    target_skill: torch.Tensor
    target_correct_input: torch.Tensor
    target_labels: torch.Tensor
    target_attention_mask: torch.Tensor
    target_metric_mask: torch.Tensor
    split: torch.Tensor
    padded_tokens: int

    def to(self, device: torch.device | str) -> "HiTSKTBatch":
        values = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in vars(self).items()
        }
        return HiTSKTBatch(**values)


def collate_hitskt(
    examples: Sequence[SessionExample], *, num_questions: int, num_skills: int
) -> HiTSKTBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    question_eos = num_questions + 1
    skill_eos = num_skills + 1
    history_entries = [
        (owner, order, session)
        for owner, example in enumerate(examples)
        for order, session in enumerate(example.history)
    ]
    max_history = max(len(session) + 1 for _, _, session in history_entries)
    max_target = max(len(example.target) + 1 for example in examples)
    history_count = torch.tensor([len(example.history) for example in examples], dtype=torch.long)

    history_question = torch.full((len(history_entries), max_history), PAD_ID, dtype=torch.long)
    history_skill = torch.full_like(history_question, PAD_ID)
    history_correct = torch.full_like(history_question, CORRECT_PAD_ID)
    history_mask = torch.zeros_like(history_question, dtype=torch.bool)
    history_owner = torch.empty(len(history_entries), dtype=torch.long)
    history_order = torch.empty(len(history_entries), dtype=torch.long)
    for row, (owner, order, session) in enumerate(history_entries):
        length = len(session)
        history_question[row, :length] = torch.tensor(session.question_ids)
        history_skill[row, :length] = torch.tensor(session.skill_ids)
        history_correct[row, :length] = torch.tensor(session.correct)
        history_question[row, length] = question_eos
        history_skill[row, length] = skill_eos
        history_correct[row, length] = CORRECT_EOS_ID
        history_mask[row, : length + 1] = True
        history_owner[row] = owner
        history_order[row] = order

    batch_size = len(examples)
    target_question = torch.full((batch_size, max_target), PAD_ID, dtype=torch.long)
    target_skill = torch.full_like(target_question, PAD_ID)
    target_correct_input = torch.full_like(target_question, CORRECT_PAD_ID)
    target_labels = torch.full_like(target_question, CORRECT_PAD_ID)
    target_attention_mask = torch.zeros_like(target_question, dtype=torch.bool)
    target_metric_mask = torch.zeros_like(target_question, dtype=torch.bool)
    for row, example in enumerate(examples):
        target = example.target
        length = len(target)
        labels = torch.tensor(target.correct, dtype=torch.long)
        target_question[row, :length] = torch.tensor(target.question_ids)
        target_skill[row, :length] = torch.tensor(target.skill_ids)
        target_labels[row, :length] = labels
        target_correct_input[row, 0] = CORRECT_BOS_ID
        if length > 1:
            target_correct_input[row, 1:length] = labels[:-1]
        target_correct_input[row, length] = labels[-1]
        target_question[row, length] = question_eos
        target_skill[row, length] = skill_eos
        target_attention_mask[row, : length + 1] = True
        target_metric_mask[row, :length] = True

    shapes = [example_shape(example) for example in examples]
    return HiTSKTBatch(
        history_question=history_question,
        history_skill=history_skill,
        history_correct=history_correct,
        history_mask=history_mask,
        history_owner=history_owner,
        history_order=history_order,
        history_count=history_count,
        target_question=target_question,
        target_skill=target_skill,
        target_correct_input=target_correct_input,
        target_labels=target_labels,
        target_attention_mask=target_attention_mask,
        target_metric_mask=target_metric_mask,
        split=torch.tensor([example.split for example in examples], dtype=torch.uint8),
        padded_tokens=padded_token_cost(shapes),
    )
