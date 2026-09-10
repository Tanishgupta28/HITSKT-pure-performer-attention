import random

import numpy as np
import torch

from ktbench.batching import ExampleShape, LengthBucketTokenBatchSampler
from ktbench.config import PROJECT_SEED, seed_everything


def test_single_project_seed_controls_all_rngs_and_batching_default() -> None:
    assert PROJECT_SEED == 42
    seed_everything()
    values = (random.random(), np.random.random(), torch.rand(1).item())
    seed_everything()
    assert values == (random.random(), np.random.random(), torch.rand(1).item())

    shapes = [ExampleShape((index + 1,), index + 2) for index in range(20)]
    first = LengthBucketTokenBatchSampler(
        shapes, token_budget=100, max_batch_size=4, bucket_size=5
    )
    second = LengthBucketTokenBatchSampler(
        shapes, token_budget=100, max_batch_size=4, bucket_size=5
    )
    assert first.seed == second.seed == PROJECT_SEED
    assert list(first) == list(second)
