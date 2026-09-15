"""Interrupted final evaluation must never extend an already stopped run.

These protocol tests use explicitly mocked epoch metrics, not benchmark results.
Real forward/backward and exact stochastic resume tests remain separate.
"""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from ktbench.rkt.data import build_memory_strength_initialization
from ktbench.rkt.phi import build_cross_fitted_phi_cache
from test_rkt_variant import _make_store


@pytest.mark.parametrize("model_name", ["dkt", "dkvmn", "sakt", "rkt", "hitskt"])
def test_resume_after_patience_stop_only_runs_final_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
    store = _make_store(tmp_path / "data")
    output = tmp_path / "experiment"
    family = model_name if model_name in ("rkt", "hitskt") else "baseline"
    runner = importlib.import_module(f"scripts.train_{family}")
    if model_name == "rkt":
        prepared = tmp_path / "prepared"
        build_cross_fitted_phi_cache(store, prepared, chunk_values=20)
        build_memory_strength_initialization(store, prepared)
        arguments = ("fixture", store, prepared, output)
    elif model_name == "hitskt":
        arguments = ("assist2017", store, output)
    else:
        arguments = (model_name, "fixture", store, output)

    calls = []

    def mock_epoch(model, dataset, *args, **kwargs):
        calls.append((dataset.split, kwargs["epoch"]))
        if dataset.split == 2:
            raise RuntimeError("simulated interruption before final test")
        return {
            "targets": len(dataset), "loss": 0.7, "roc_auc": 0.6,
            "accuracy": 0.5, "precision": 0.5, "recall": 0.5,
            "f1": 0.5, "mse": 0.25, "threshold": 0.5,
        }

    monkeypatch.setattr(runner, "_run_epoch", mock_epoch)
    train = getattr(runner, f"train_{family}")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(*arguments, epoch_ceiling_override=8)
    config = json.loads((output / "config.json").read_text())
    assert config["epochs_completed"] == 6
    assert config["best_epoch"] == 1
    checkpoint_hash = hashlib.sha256((output / "last_model.pt").read_bytes()).hexdigest()
    calls.clear()

    def final_test_only(model, dataset, *args, **kwargs):
        assert dataset.split == 2, "resume must not run another train/validation epoch"
        calls.append((dataset.split, kwargs["epoch"]))
        return {
            "targets": len(dataset), "loss": 0.7, "roc_auc": 0.6,
            "accuracy": 0.5, "precision": 0.5, "recall": 0.5,
            "f1": 0.5, "mse": 0.25, "threshold": 0.5,
        }

    monkeypatch.setattr(runner, "_run_epoch", final_test_only)
    result = train(*arguments, epoch_ceiling_override=8, resume=True)
    assert calls == [(2, 1)]
    assert result["epochs_completed"] == 6
    assert result["best_epoch"] == 1
    assert result["patience"] == 5
    assert result["best_checkpoint_reloaded"] is True
    assert (output / "_SUCCESS").exists()
    assert hashlib.sha256((output / "last_model.pt").read_bytes()).hexdigest() == checkpoint_hash
    events = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    epoch_events = [event for event in events if event["event"] == "epoch_complete"]
    assert [event["epoch"] for event in epoch_events] == list(range(1, 7))
    assert epoch_events[-1]["early_stop"] is True
    assert epoch_events[-1]["consecutive_epochs_without_improvement"] == 5
