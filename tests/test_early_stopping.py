import math
import random

import numpy as np
import pytest
import torch

from ktbench.training import (
    COMMON_EARLY_STOPPING,
    EarlyStoppingConfig,
    ValidationAUCEarlyStopping,
    capture_rng_state,
    restore_rng_state,
)


def test_common_early_stopping_contract_is_fixed() -> None:
    assert COMMON_EARLY_STOPPING.metric == "validation_roc_auc"
    assert COMMON_EARLY_STOPPING.patience == 5
    assert COMMON_EARLY_STOPPING.min_delta == 0
    assert COMMON_EARLY_STOPPING.strict_improvement is True
    with pytest.raises(ValueError):
        EarlyStoppingConfig(patience=6)


def test_strict_improvement_reset_and_exact_five_epoch_stop() -> None:
    control = ValidationAUCEarlyStopping()
    first = control.step(1, 0.7)
    assert first.improved and not first.should_stop and first.best_epoch == 1
    equal = control.step(2, 0.7)
    assert not equal.improved and equal.consecutive_epochs_without_improvement == 1
    improved = control.step(3, 0.71)
    assert improved.improved and improved.consecutive_epochs_without_improvement == 0
    for epoch in range(4, 8):
        assert not control.step(epoch, 0.70).should_stop
    stopped = control.step(8, 0.70)
    assert stopped.should_stop
    assert stopped.best_epoch == 3
    assert stopped.best_validation_auc == 0.71


def test_nonfinite_auc_never_becomes_best_checkpoint() -> None:
    control = ValidationAUCEarlyStopping()
    decision = control.step(1, math.nan)
    assert not decision.improved
    assert decision.best_epoch is None


def test_all_project_rng_states_round_trip_through_weights_only_checkpoint(
    tmp_path,
) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    state = capture_rng_state()
    checkpoint = tmp_path / "rng.pt"
    torch.save({"rng_state": state}, checkpoint)
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
        torch.rand(4, device="cuda").cpu() if torch.cuda.is_available() else None,
    )
    loaded = torch.load(checkpoint, weights_only=True)["rng_state"]
    restore_rng_state(loaded)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
        torch.rand(4, device="cuda").cpu() if torch.cuda.is_available() else None,
    )
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
    if torch.cuda.is_available():
        assert torch.equal(expected[3], actual[3])
