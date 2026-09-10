import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.rolling_targets import (
    RollingLengthBatchSampler,
    RollingTargetDataset,
    collate_rolling_targets,
)
from ktbench.data.session_store import SessionStore, build_session_store
from ktbench.metrics import binary_metrics
from ktbench.models import (
    DKT,
    DKVMN,
    SAKT,
    DKTConfig,
    DKVMNConfig,
    SAKTConfig,
    load_baseline_checkpoint,
)
from scripts.train_baseline import train_baseline


def _store(root: Path, *, long_session: bool = True) -> SessionStore:
    rows = []
    timestamp = 1_600_000_000_000
    for student in range(1, 7):
        for session in range(1, 6):
            split = 0 if session <= 3 else session - 3
            length = 230 if long_session and student == 1 and session == 1 else 3
            for position in range(1, length + 1):
                timestamp += 1_000
                rows.append(
                    {
                        "student_id": student,
                        "question_id": (position + session) % 11 + 1,
                        "skill_id": (position + student) % 5 + 1,
                        "correct": (position + student + session) % 2,
                        "timestamp": timestamp,
                        "session_id": session,
                        "session_position": position,
                        "split": split,
                    }
                )
            timestamp += 40_000_000
    processed = root / "processed"
    (processed / "events").mkdir(parents=True)
    (processed / "reports").mkdir()
    frame = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(frame), processed / "events" / "part.parquet")
    (processed / "reports" / "summary.json").write_text(
        json.dumps(
            {
                "retained_interactions": len(frame),
                "retained_sessions": 30,
                "retained_students": 6,
            }
        )
    )
    (processed / "_SUCCESS").write_text("complete\n")
    build_session_store(processed, root / "store")
    return SessionStore(root / "store")


def test_rolling_target_once_lengths_chronology_and_split_history(tmp_path: Path) -> None:
    store = _store(tmp_path)
    datasets = {
        name: RollingTargetDataset(store, name, history_length=199)
        for name in ("train", "validation", "test")
    }
    target_ids = [item.target_event_id for data in datasets.values() for item in data]
    assert len(target_ids) == len(set(target_ids)) == store.metadata["interactions"]

    capped = datasets["train"][205]
    assert len(capped.history_question) == 199
    assert np.all(np.asarray(capped.history_timestamp_ms) <= capped.target_timestamp_ms)
    first_validation = datasets["validation"][0]
    assert len(first_validation.history_question) > 0
    assert first_validation.split == 1

    sakt = RollingTargetDataset(store, "train", history_length=99)
    assert len(sakt[205].history_question) == 99


def test_dynamic_padding_mask_and_seeded_bucketing(tmp_path: Path) -> None:
    dataset = RollingTargetDataset(_store(tmp_path), "train", history_length=199)
    batch = collate_rolling_targets([dataset[0], dataset[4], dataset[205]])
    assert batch.history_question.shape == (3, 199)
    assert batch.history_mask.sum(dim=1).tolist() == [0, 4, 199]
    assert torch.all(batch.history_question[~batch.history_mask] == 0)

    first = RollingLengthBatchSampler(dataset, batch_size=7, bucket_size=21)
    second = RollingLengthBatchSampler(dataset, batch_size=7, bucket_size=21)
    first_indices = [index for group in first for index in group]
    second_indices = [index for group in second for index in group]
    assert first.seed == second.seed == PROJECT_SEED
    assert first_indices == second_indices
    assert sorted(first_indices) == list(range(len(dataset)))


@pytest.mark.parametrize(
    ("model_class", "config"),
    [
        (DKT, DKTConfig(num_skills=5)),
        (DKVMN, DKVMNConfig(num_skills=5)),
        (SAKT, SAKTConfig(num_questions=11, num_skills=5)),
    ],
)
def test_baseline_forward_backward_pad_and_target_masking(
    tmp_path: Path, model_class: type[torch.nn.Module], config: object
) -> None:
    seed_everything()
    history_length = getattr(config, "history_length")
    dataset = RollingTargetDataset(_store(tmp_path), "train", history_length=history_length)
    batch = collate_rolling_targets([dataset[0], dataset[4], dataset[120]])
    model = model_class(config).train()
    logits = model(batch)
    assert logits.shape == batch.target_labels.shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.target_labels.float()
    )
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    model.eval()
    changed_padding = collate_rolling_targets([dataset[0], dataset[4], dataset[120]])
    changed_padding.history_question[~changed_padding.history_mask] = 7
    changed_padding.history_skill[~changed_padding.history_mask] = 3
    changed_padding.history_correct[~changed_padding.history_mask] = 1
    changed_label = collate_rolling_targets([dataset[0], dataset[4], dataset[120]])
    changed_label.target_labels[:] = 1 - changed_label.target_labels
    with torch.no_grad():
        expected = model(batch)
        assert torch.allclose(expected, model(changed_padding))
        assert torch.equal(expected, model(changed_label))

    metrics = binary_metrics(
        torch.sigmoid(expected), batch.target_labels, loss=float(loss.detach())
    )
    assert metrics["targets"] == len(batch.target_labels)
    checkpoint = tmp_path / f"{model_class.__name__}.pt"
    torch.save(model.checkpoint(), checkpoint)
    restored = load_baseline_checkpoint(checkpoint).eval()
    with torch.no_grad():
        assert torch.equal(expected, restored(batch))


def test_dkvmn_balanced_write_scan_matches_reference_outputs_and_gradients(
    tmp_path: Path,
) -> None:
    seed_everything()
    dataset = RollingTargetDataset(_store(tmp_path), "train", history_length=199)
    batch = collate_rolling_targets([dataset[0], dataset[4], dataset[120], dataset[205]])
    balanced = DKVMN(DKVMNConfig(num_skills=5))
    sequential = DKVMN(DKVMNConfig(num_skills=5))
    sequential.load_state_dict(balanced.state_dict())

    balanced_memory = balanced._final_memory_balanced(batch)
    sequential_memory = sequential._final_memory_sequential(batch)
    assert torch.allclose(balanced_memory, sequential_memory, rtol=2e-5, atol=2e-6)

    balanced_memory.square().mean().backward()
    sequential_memory.square().mean().backward()
    for balanced_parameter, sequential_parameter in zip(
        balanced.parameters(), sequential.parameters(), strict=True
    ):
        if balanced_parameter.grad is None or sequential_parameter.grad is None:
            assert balanced_parameter.grad is sequential_parameter.grad is None
        else:
            assert torch.allclose(
                balanced_parameter.grad,
                sequential_parameter.grad,
                rtol=5e-5,
                atol=2e-6,
            )


def test_production_runner_writes_and_reloads_best_checkpoint(tmp_path: Path) -> None:
    store = _store(tmp_path / "data", long_session=False)
    output = tmp_path / "experiment"
    result = train_baseline(
        "dkt",
        "fixture",
        store,
        output,
        epoch_ceiling_override=1,
    )
    config = json.loads((output / "config.json").read_text())
    assert result["best_checkpoint_reloaded"] is True
    assert result["best_epoch"] == config["best_epoch"] == 1
    assert result["epochs_completed"] == config["epochs_completed"] == 1
    assert config["early_stopping_patience"] == 5
    assert config["early_stopping_min_delta"] == 0
    assert (output / "best_model.pt").exists()
    assert (output / "final_results.json").exists()
    assert (output / "_SUCCESS").exists()


def test_production_runner_resumes_model_optimizer_and_patience_state(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "data", long_session=False)
    output = tmp_path / "experiment"
    train_baseline("dkt", "fixture", store, output, epoch_ceiling_override=1)

    (output / "_SUCCESS").unlink()
    (output / "final_results.json").unlink()
    epoch_lines = [
        line
        for line in (output / "training.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "epoch_complete"
    ]
    (output / "training.jsonl").write_text("\n".join(epoch_lines) + "\n")
    with (output / "metrics.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
        fieldnames = source.seek(0) or next(csv.reader(source))
    with (output / "metrics.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if row["split"] != "test")

    result = train_baseline(
        "dkt",
        "fixture",
        store,
        output,
        epoch_ceiling_override=2,
        resume=True,
    )
    events = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert [event["epoch"] for event in events if event["event"] == "epoch_complete"] == [1, 2]
    resume_event = next(event for event in events if event["event"] == "training_resumed")
    assert resume_event["resumed_from_epoch"] == 1
    assert result["epochs_completed"] == 2
    assert result["resume_count"] == 1
    assert (output / "_SUCCESS").exists()
