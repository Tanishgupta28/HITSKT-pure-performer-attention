"""The authorized DKT/EdNet exception preserves state and other run protocols."""

import hashlib
import json

import pytest
import torch

from ktbench.training import COMMON_EARLY_STOPPING, DKTEdNetEarlyStoppingConfig
from scripts.benchmark_exact_transport import exact
from scripts.train_baseline import _early_stopping_config, train_baseline
from test_rolling_baselines import _store


@pytest.mark.parametrize("model,dataset", [
    ("dkt", "assist2017"), ("dkt", "junyi"),
    ("dkvmn", "ednet_kt1"), ("sakt", "ednet_kt1"),
])
def test_patience_exception_rejects_other_experiments(model, dataset):
    with pytest.raises(ValueError, match="only for DKT/full EdNet"):
        _early_stopping_config(model, dataset, 10)
    assert _early_stopping_config(model, dataset, 5) is COMMON_EARLY_STOPPING


def test_exception_preserves_every_other_selection_setting():
    expected = COMMON_EARLY_STOPPING.as_dict()
    expected["patience"] = 10
    assert _early_stopping_config("dkt", "ednet_kt1", 10).as_dict() == expected
    with pytest.raises(ValueError):
        DKTEdNetEarlyStoppingConfig(min_delta=0.01)


def test_extending_restored_patience_keeps_count_and_stops_at_ten(tmp_path, monkeypatch):
    """Use synthetic metrics to exercise stopping only; no benchmark results."""
    import scripts.train_baseline as runner

    store = _store(tmp_path / "data")
    output = tmp_path / "run"
    calls = []

    def epoch(model, dataset, *args, **kwargs):
        calls.append((dataset.split, kwargs["epoch"]))
        if dataset.split == 2:
            raise RuntimeError("interrupted before synthetic test")
        return {"targets": len(dataset), "loss": 0.7, "roc_auc": 0.6,
                "accuracy": 0.5, "precision": 0.5, "recall": 0.5,
                "f1": 0.5, "mse": 0.25, "threshold": 0.5}

    monkeypatch.setattr(runner, "_run_epoch", epoch)
    with pytest.raises(RuntimeError, match="synthetic test"):
        train_baseline("dkt", "ednet_kt1", store, output, epoch_ceiling_override=20)
    before = torch.load(output / "last_model.pt", weights_only=True)
    assert before["epoch"] == 6
    assert before["early_stopping_state"]["consecutive_epochs_without_improvement"] == 5
    best_hash = hashlib.sha256((output / "best_model.pt").read_bytes()).hexdigest()
    calls.clear()
    with pytest.raises(RuntimeError, match="synthetic test"):
        train_baseline("dkt", "ednet_kt1", store, output, epoch_ceiling_override=20,
                       resume=True, early_stopping_patience_override=10)
    assert [epoch for split, epoch in calls if split == 0] == list(range(7, 12))
    config = json.loads((output / "config.json").read_text())
    assert config["epochs_completed"] == 11
    assert config["early_stopping_patience"] == 10
    assert config["early_stopping_patience_previous"] == 5
    assert config["early_stopping_patience_changed_after_epoch"] == 6
    checkpoint = torch.load(output / "last_model.pt", weights_only=True)
    assert checkpoint["early_stopping_state"]["consecutive_epochs_without_improvement"] == 10
    assert hashlib.sha256((output / "best_model.pt").read_bytes()).hexdigest() == best_hash
    resume_event = next(e for e in map(json.loads, (output / "training.jsonl").read_text().splitlines())
                        if e["event"] == "training_resumed")
    assert resume_event["consecutive_epochs_without_improvement"] == 5
    assert resume_event["early_stopping_patience"] == 10
    calls.clear()

    def test_only(model, dataset, *args, **kwargs):
        assert dataset.split == 2
        calls.append((dataset.split, kwargs["epoch"]))
        return {"targets": len(dataset), "loss": 0.7, "roc_auc": 0.6,
                "accuracy": 0.5, "precision": 0.5, "recall": 0.5,
                "f1": 0.5, "mse": 0.25, "threshold": 0.5}

    monkeypatch.setattr(runner, "_run_epoch", test_only)
    result = train_baseline("dkt", "ednet_kt1", store, output,
                            epoch_ceiling_override=20, resume=True)
    assert calls == [(2, 1)]
    assert result["patience"] == 10 and result["epochs_completed"] == 11
    assert result["best_checkpoint_reloaded"] is True


def test_dkt_patience_change_resume_preserves_exact_training_state(tmp_path, monkeypatch):
    """Real updates verify model, Adam and all RNG against uninterrupted training."""
    import scripts.train_baseline as runner

    store = _store(tmp_path / "data")
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    expected = train_baseline("dkt", "ednet_kt1", store, full, epoch_ceiling_override=2,
                              transport="packed64", early_stopping_patience_override=10)
    original_write = runner._write_json

    class BoundaryStop(Exception):
        pass

    def stop_after_epoch(path, value):
        original_write(path, value)
        if path == resumed / "config.json" and value.get("epochs_completed") == 1:
            raise BoundaryStop()

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_write_json", stop_after_epoch)
        with pytest.raises(BoundaryStop):
            train_baseline("dkt", "ednet_kt1", store, resumed, epoch_ceiling_override=2,
                           transport="packed64")
    actual = train_baseline("dkt", "ednet_kt1", store, resumed, epoch_ceiling_override=2,
                            transport="packed64", resume=True,
                            early_stopping_patience_override=10)
    assert actual["test"] == expected["test"]
    assert actual["best_epoch"] == expected["best_epoch"]
    assert exact(torch.load(full / "last_model.pt", weights_only=True),
                 torch.load(resumed / "last_model.pt", weights_only=True))
