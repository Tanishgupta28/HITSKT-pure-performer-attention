import json

import pytest
import torch

from ktbench.config import seed_everything
from ktbench.baseline_registry import make_baseline_model, BASELINE_TRAINING_CONFIGS
from ktbench.data.rolling_targets import RollingTargetDataset
from ktbench.training import capture_rng_state, restore_rng_state
from scripts.benchmark_exact_transport import exact, cpu_tree
from scripts.train_baseline import _run_epoch, train_baseline
from test_rolling_baselines import _store


@pytest.mark.parametrize("name", ["dkt", "sakt"])
@pytest.mark.parametrize("workers", [0, 2])
def test_production_transport_exact(tmp_path, name, workers):
    store = _store(tmp_path)
    profile = BASELINE_TRAINING_CONFIGS[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything()
    model = make_baseline_model(name, num_questions=11, num_skills=5).to(device)
    initial = cpu_tree(model.state_dict())
    rng = capture_rng_state()
    outputs = []
    for transport in ("reference", "packed64"):
        model.load_state_dict(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=profile["learning_rate"])
        restore_rng_state(rng)
        metrics = []
        for split in ("train", "validation", "test"):
            dataset = RollingTargetDataset(store, split, history_length=profile["history_length"])
            metrics.append(_run_epoch(model, dataset, batch_size=3, device=device,
                workers=workers, epoch=16, optimizer=optimizer if split == "train" else None,
                gradient_clip=profile["gradient_clip"], transport=transport))
        outputs.append((cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict()),
                        capture_rng_state(), metrics))
    assert exact(*outputs)


def test_reference_to_packed_resume_exact(tmp_path, monkeypatch):
    store = _store(tmp_path)
    full = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    expected = train_baseline("sakt", "fixture", store, full, epoch_ceiling_override=2)
    # Simulate a stop immediately after a fully written epoch checkpoint/config.
    import scripts.train_baseline as runner
    original_write = runner._write_json

    class BoundaryStop(Exception):
        pass

    def stop_after_checkpoint(path, value):
        original_write(path, value)
        if path == resumed / "config.json" and value.get("epochs_completed") == 1:
            raise BoundaryStop()

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_write_json", stop_after_checkpoint)
        with pytest.raises(BoundaryStop):
            train_baseline("sakt", "fixture", store, resumed, epoch_ceiling_override=2)
    actual = train_baseline("sakt", "fixture", store, resumed,
                            epoch_ceiling_override=2, resume=True, transport="packed64")
    assert actual["test"] == expected["test"]
    assert actual["best_epoch"] == expected["best_epoch"]
    assert exact(torch.load(full / "last_model.pt", weights_only=True),
                 torch.load(resumed / "last_model.pt", weights_only=True))
    config = json.loads((resumed / "config.json").read_text())
    assert config["transport"] == "packed64"
    assert config["previous_transport"] == "reference"
