"""Experimental transport-only optimizations; not enabled in production runners.

The existing sampler still defines every minibatch and its exact order/shape.
Packing groups minibatches for transport only, never for an optimizer update.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ktbench.data.rolling_targets import RollingTargetBatch, RollingTargetDataset


def vectorized_batch(dataset: RollingTargetDataset, indices: Sequence[int]) -> RollingTargetBatch:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("a nonempty one-dimensional batch is required")
    if np.any(indices < 0) or np.any(indices >= len(dataset)):
        raise IndexError("target index outside dataset")
    store = dataset.store
    sessions, events = dataset._events_for_indices(indices)
    first_sessions = sessions - store.session_number[sessions].astype(np.int64) + 1
    first_events = store.session_offsets[first_sessions]
    lengths = np.minimum(events - first_events, dataset.history_length)
    width = max(1, int(lengths.max()))
    distances = np.arange(width, 0, -1, dtype=np.int64)[None, :]
    mask = distances <= lengths[:, None]
    history = np.maximum(events[:, None] - distances, 0)
    if np.any(mask & (store.timestamp_ms[history] > store.timestamp_ms[events, None])):
        raise ValueError("future interaction found in rolling history")

    def tensor(values: np.ndarray, dtype: np.dtype = np.int64) -> torch.Tensor:
        return torch.from_numpy(np.array(values, dtype=dtype, copy=True))

    return RollingTargetBatch(
        student_id=tensor(store.session_student[sessions]),
        history_question=tensor(np.where(mask, store.question[history], 0)),
        history_skill=tensor(np.where(mask, store.skill[history], 0)),
        history_correct=tensor(np.where(mask, store.correct[history], 0)),
        history_mask=tensor(mask, np.bool_),
        target_question=tensor(store.question[events]),
        target_skill=tensor(store.skill[events]),
        target_labels=tensor(store.correct[events]),
        target_event_id=tensor(events),
        split=tensor(np.full(len(indices), dataset.split), np.uint8),
    )


@dataclass
class PackedBatches:
    storage: torch.Tensor
    layouts: list[dict[str, tuple[int, int, torch.dtype, tuple[int, ...]]]]

    def pin_memory(self) -> "PackedBatches":
        return PackedBatches(self.storage.pin_memory(), self.layouts)

    def to_batches(self, device: str | torch.device) -> list[RollingTargetBatch]:
        storage = self.storage.to(device, non_blocking=True)
        return [
            RollingTargetBatch(**{
                name: storage[start:stop].view(dtype).reshape(shape)
                for name, (start, stop, dtype, shape) in layout.items()
            })
            for layout in self.layouts
        ]


def pack_batches(batches: Sequence[RollingTargetBatch]) -> PackedBatches:
    """Byte-preserving aligned packing; no dtype conversion or arithmetic."""
    if not batches:
        raise ValueError("cannot pack an empty block")
    pieces, layouts = [], []
    offset = 0
    for batch in batches:
        layout = {}
        for name, value in vars(batch).items():
            if value.device.type != "cpu":
                raise ValueError("packing requires host batches")
            padding = (-offset) % 8
            if padding:
                pieces.append(torch.zeros(padding, dtype=torch.uint8))
                offset += padding
            raw = value.contiguous().reshape(-1).view(torch.uint8)
            stop = offset + raw.numel()
            layout[name] = (offset, stop, value.dtype, tuple(value.shape))
            pieces.append(raw)
            offset = stop
        layouts.append(layout)
    return PackedBatches(torch.cat(pieces), layouts)


class BatchPlanDataset(Dataset):
    """Use a preselected original sampler plan without changing batch boundaries."""

    def __init__(self, dataset, plan, block_size=1, packed=False):
        if block_size < 1 or (not packed and block_size != 1):
            raise ValueError("nonpacked plans require block size one; packed blocks must be positive")
        self.dataset, self.plan = dataset, plan
        self.block_size, self.packed = block_size, packed

    def __len__(self):
        return (len(self.plan) + self.block_size - 1) // self.block_size

    def __getitem__(self, index):
        start = index * self.block_size
        batches = [vectorized_batch(self.dataset, indices)
                   for indices in self.plan[start:start + self.block_size]]
        return pack_batches(batches) if self.packed else batches[0]
