import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ktbench.data.session_store import (
    RollingSessionDataset,
    SessionStore,
    build_session_store,
    make_session_dataloader,
)
from scripts.train_hitskt import train_hitskt


def _processed_fixture(root: Path) -> Path:
    rows = []
    cursor = 0
    # Five sessions retained for each student, with deliberately varied lengths.
    for student, lengths in [(1, [1, 2, 4, 1, 3]), (2, [2, 1, 3, 2, 5])]:
        for session_id, length in enumerate(lengths, start=1):
            split = 0 if session_id <= 3 else session_id - 3
            for position in range(1, length + 1):
                rows.append(
                    {
                        "student_id": student,
                        "question_id": cursor % 9 + 1,
                        "skill_id": cursor % 4 + 1,
                        "correct": cursor % 2,
                        "timestamp": 1_600_000_000_000 + cursor * 1_000,
                        "session_id": session_id,
                        "session_position": position,
                        "split": split,
                    }
                )
                cursor += 1
    (root / "events").mkdir(parents=True)
    (root / "reports").mkdir()
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), root / "events" / "part-00000.parquet")
    (root / "reports" / "summary.json").write_text(
        json.dumps(
            {
                "retained_interactions": len(rows),
                "retained_sessions": 10,
                "retained_students": 2,
            }
        )
    )
    (root / "_SUCCESS").write_text("complete\n")
    return root


def test_store_is_lossless_and_rolling_splits_cannot_see_future(tmp_path: Path) -> None:
    processed = _processed_fixture(tmp_path / "processed")
    metadata = build_session_store(processed, tmp_path / "store")
    store = SessionStore(tmp_path / "store")

    assert metadata.interactions == 24
    assert metadata.sessions == 10
    assert metadata.maximum_session_length == 5
    assert not metadata.action_truncation
    assert not metadata.action_chunking
    assert sum(store.session_length(index) for index in range(10)) == 24

    validation = RollingSessionDataset(store, "validation", history_sessions=None)
    test = RollingSessionDataset(store, "test", history_sessions=None)
    assert len(validation) == len(test) == 2
    assert [len(session) for session in validation[0].history] == [1, 2, 4]
    assert len(validation[0].target) == 1
    assert [len(session) for session in test[0].history] == [1, 2, 4, 1]
    assert len(test[0].target) == 3


def test_history_window_never_truncates_actions_within_session(tmp_path: Path) -> None:
    processed = _processed_fixture(tmp_path / "processed")
    build_session_store(processed, tmp_path / "store")
    dataset = RollingSessionDataset(SessionStore(tmp_path / "store"), "test", history_sessions=2)

    assert [len(session) for session in dataset[0].history] == [4, 1]
    assert len(dataset[0].target) == 3
    assert dataset.shape(0).history_lengths == (4, 1)

    loader, sampler = make_session_dataloader(
        dataset,
        token_budget=100,
        maximum_batch_size=2,
        bucket_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert batch.target_metric_mask.sum().item() == 8
    assert batch.history_mask.sum().item() == 14
    sampler.set_epoch(1)


def test_hitskt_production_runner_uses_common_stop_and_reloads_best(tmp_path: Path) -> None:
    processed = _processed_fixture(tmp_path / "processed")
    build_session_store(processed, tmp_path / "store")
    output = tmp_path / "experiment"
    result = train_hitskt(
        "assist2017",
        SessionStore(tmp_path / "store"),
        output,
        epoch_ceiling_override=1,
    )

    config = json.loads((output / "config.json").read_text())
    assert config["attention"] == "pure causal ELU+1 Performer linear attention"
    assert config["action_truncation"] is False
    assert config["action_chunking"] is False
    assert config["early_stopping_patience"] == 5
    assert config["early_stopping_min_delta"] == 0
    assert result["best_epoch"] == result["epochs_completed"] == 1
    assert result["best_checkpoint_reloaded"] is True
    assert (output / "_SUCCESS").exists()
