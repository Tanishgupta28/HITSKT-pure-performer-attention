import numpy as np
import pytest
import torch

from ktbench.data.fast_rolling import vectorized_batch, pack_batches
from ktbench.data.rolling_targets import RollingTargetDataset, collate_rolling_targets
from test_rolling_baselines import _store


@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("history", [49, 99, 199])
def test_vectorized_and_packed_batches_exact(tmp_path, split, history):
    dataset = RollingTargetDataset(_store(tmp_path), split, history_length=history)
    batches = []
    plans = [[0], [0, len(dataset) - 1, 0]] + [
        list(range(start, min(start + 7, len(dataset))))[::-1]
        for start in range(0, len(dataset), 7)
    ]
    for indices in plans:
        expected = collate_rolling_targets([dataset[i] for i in indices])
        actual = vectorized_batch(dataset, indices)
        for key, value in vars(expected).items():
            assert value.dtype == getattr(actual, key).dtype
            assert torch.equal(value, getattr(actual, key))
        batches.append(actual)
    restored = pack_batches(batches).to_batches("cpu")
    for expected, actual in zip(batches, restored, strict=True):
        for key, value in vars(expected).items():
            assert value.dtype == getattr(actual, key).dtype
            assert torch.equal(value, getattr(actual, key))


def test_vectorized_rejects_future_history(tmp_path):
    dataset = RollingTargetDataset(_store(tmp_path), "train", history_length=99)
    timestamps = np.array(dataset.store.timestamp_ms)
    timestamps[0] = timestamps[1] + 1
    dataset.store.timestamp_ms = timestamps
    with pytest.raises(ValueError, match="future"):
        vectorized_batch(dataset, [1])
