from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pytest

from ktbench.data.flat import (
    FlatDataValidationError,
    SESSION_GAP_SECONDS,
    preprocess_flat,
)


def test_assist_uses_authoritative_event_order_and_rebuilds_splits(tmp_path: Path) -> None:
    rows = []
    # Archive/source row order is deliberately different from action order.
    for action, timestamp in [(2, 0), (1, 0), (3, SESSION_GAP_SECONDS), (4, 2 * SESSION_GAP_SECONDS), (5, 3 * SESSION_GAP_SECONDS), (6, 4 * SESSION_GAP_SECONDS)]:
        rows.append(
            {
                "studentId": 10,
                "action_num": action,
                "skill": 7,
                "problemId": 100 + action,
                "correct": action % 2,
                "startTime": timestamp,
            }
        )
    rows.append(
        {
            "studentId": 20,
            "action_num": 1,
            "skill": 8,
            "problemId": 200,
            "correct": 1,
            "startTime": 0,
        }
    )
    source = tmp_path / "assist.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / "processed"
    summary = preprocess_flat(
        "assist2017", source, output, rows_per_shard=2, verify_hash=False
    )

    assert summary["source_rows"] == 7
    assert summary["retained_students"] == 1
    assert summary["retained_interactions"] == 6
    events = ds.dataset(output / "events", format="parquet").to_table().to_pandas()
    assert events["source_event_id"].tolist()[:2] == [1, 2]
    assert events["session_id"].tolist() == [1, 1, 2, 3, 4, 5]
    assert events["split"].tolist() == [0, 0, 0, 0, 1, 2]
    assert events["session_position"].tolist() == [1, 2, 1, 1, 1, 1]


def test_question_with_multiple_source_skills_fails_closed(tmp_path: Path) -> None:
    rows = []
    for session in range(5):
        rows.append(
            {
                "studentId": 1,
                "skill": 1 if session % 2 == 0 else 2,
                "problemId": 10,
                "correct": 1,
                "startTime": session * SESSION_GAP_SECONDS,
            }
        )
    source = tmp_path / "junyi.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    with pytest.raises(FlatDataValidationError, match="multiple skill values"):
        preprocess_flat("junyi", source, tmp_path / "processed", verify_hash=False)


def test_ambiguous_target_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "junyi.csv"
    pd.DataFrame(
        {
            "studentId": [1],
            "skill": [1],
            "problemId": [1],
            "correct": [-1],
            "startTime": [0],
        }
    ).to_csv(source, index=False)
    with pytest.raises(FlatDataValidationError, match="ambiguous target"):
        preprocess_flat("junyi", source, tmp_path / "processed", verify_hash=False)

