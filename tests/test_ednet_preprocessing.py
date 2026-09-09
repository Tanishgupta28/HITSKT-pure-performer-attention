import csv
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pyarrow.dataset as ds
import pytest

from ktbench.data.ednet import (
    EdNetValidationError,
    SESSION_GAP_MS,
    UNTAGGED_KEY,
    assign_sessions_and_splits,
    build_metadata_mappings,
    canonicalize_tags,
    preprocess_ednet,
)


QUESTION_COLUMNS = [
    "question_id",
    "bundle_id",
    "explanation_id",
    "correct_answer",
    "part",
    "tags",
    "deployed_at",
]


def test_canonicalize_complete_sorted_tag_set_and_untagged() -> None:
    result = canonicalize_tags("179;2;181")
    assert result.canonical_key == "2;179;181"
    assert result.tags == (2, 179, 181)
    assert not result.is_untagged

    missing = canonicalize_tags("-1")
    assert missing.canonical_key == UNTAGGED_KEY
    assert missing.tags == ()
    assert missing.is_untagged

    duplicated = canonicalize_tags("152;147;163;179;178;149;178")
    assert duplicated.canonical_key == "147;149;152;163;178;179"
    assert duplicated.tags == (147, 149, 152, 163, 178, 179)


@pytest.mark.parametrize(
    "value", [None, "", " 1", "1 ", "1;;2", "1;-1", "0"]
)
def test_canonicalize_rejects_unapproved_tag_metadata(value: object) -> None:
    with pytest.raises(EdNetValidationError):
        canonicalize_tags(value)


def test_mapping_is_numeric_deterministic_and_reserves_untagged() -> None:
    questions = pd.DataFrame(
        [
            ["q10", "b1", "e1", "a", "1", "10;2", "1"],
            ["q2", "b2", "e2", "b", "1", "-1", "1"],
            ["q1", "b3", "e3", "c", "1", "3", "1"],
            ["q20", "b4", "e4", "d", "1", "2", "1"],
        ],
        columns=QUESTION_COLUMNS,
    )
    mappings = build_metadata_mappings(questions)

    assert mappings.original_tags["original_tag_id"].tolist() == [2, 3, 10]
    assert mappings.composites["canonical_skill_key"].tolist() == [
        UNTAGGED_KEY,
        "2",
        "2;10",
        "3",
    ]
    assert mappings.composites["skill_id"].tolist() == [1, 2, 3, 4]
    assert mappings.questions["question_key"].tolist() == ["q1", "q2", "q10", "q20"]
    assert mappings.questions.loc[
        mappings.questions["question_key"] == "q2", "skill_id"
    ].item() == 1


def test_session_threshold_and_chronological_60_20_20_split() -> None:
    timestamps = [
        0,
        1,
        SESSION_GAP_MS + 1,
        2 * SESSION_GAP_MS + 1,
        3 * SESSION_GAP_MS + 1,
        4 * SESSION_GAP_MS + 1,
    ]
    sessions, splits, count = assign_sessions_and_splits(timestamps)
    assert count == 5
    assert sessions.tolist() == [1, 1, 2, 3, 4, 5]
    assert splits.tolist() == [0, 0, 0, 0, 1, 2]


def _csv_text(rows: list[list[object]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue()


def test_end_to_end_retains_rows_once_and_preserves_timestamp_ties(
    tmp_path: Path,
) -> None:
    contents_zip = tmp_path / "contents.zip"
    questions = [
        QUESTION_COLUMNS,
        ["q1", "b1", "e1", "a", "1", "-1", "1"],
        ["q2", "b2", "e2", "b", "1", "3;2", "1"],
        ["q3", "b3", "e3", "c", "1", "4", "1"],
    ]
    with ZipFile(contents_zip, "w", ZIP_DEFLATED) as archive:
        archive.writestr("contents/questions.csv", _csv_text(questions))

    header = ["timestamp", "solving_id", "question_id", "user_answer", "elapsed_time"]
    retained_rows = [
        header,
        [0, 1, "q1", "a", 10],
        [0, 1, "q2", "a", 11],
        [2, 1, "q2", "", 11],
        [SESSION_GAP_MS, 2, "q2", "b", 12],
        [2 * SESSION_GAP_MS, 3, "q3", "c", 13],
        [3 * SESSION_GAP_MS, 4, "q3", "d", 14],
        [4 * SESSION_GAP_MS, 5, "q1", "b", 15],
    ]
    excluded_rows = [header, [0, 1, "q1", "a", 10]]
    kt1_zip = tmp_path / "kt1.zip"
    with ZipFile(kt1_zip, "w", ZIP_DEFLATED) as archive:
        # Deliberately reverse numeric user order in the archive.
        archive.writestr("KT1/u20.csv", _csv_text(retained_rows))
        archive.writestr("KT1/u3.csv", _csv_text(excluded_rows))

    output = tmp_path / "processed"
    summary = preprocess_ednet(
        kt1_zip, contents_zip, output, rows_per_shard=2
    )

    assert summary["genuine_original_tags"] == 3
    assert summary["untagged_questions"] == 1
    assert summary["composite_skills"] == 3
    assert summary["raw_students"] == 2
    assert summary["raw_interactions"] == 8
    assert summary["unanswered_interactions"] == 1
    assert summary["unanswered_affected_students"] == 1
    assert summary["supervised_interactions"] == 7
    assert summary["unanswered_percentage"] == 12.5
    assert summary["raw_untagged_interactions"] == 3
    assert summary["raw_multitag_interactions"] == 3
    assert summary["retained_students"] == 1
    assert summary["retained_interactions"] == 6
    assert summary["retained_untagged_interactions"] == 2
    assert summary["event_shards"] == 3

    events = ds.dataset(output / "events", format="parquet").to_table().to_pandas()
    assert len(events) == 6
    assert events["source_row"].tolist()[:2] == [1, 2]
    assert events["timestamp"].tolist()[:2] == [0, 0]
    assert events["skill_id"].tolist()[0] == 1
    assert events["split"].tolist() == [0, 0, 0, 0, 1, 2]
    unanswered = (
        ds.dataset(output / "audit" / "unanswered_events", format="parquet")
        .to_table()
        .to_pandas()
    )
    assert len(unanswered) == 1
    assert unanswered.loc[0, "original_student_id"] == 20
    assert unanswered.loc[0, "source_row"] == 3
    assert unanswered.loc[0, "user_answer"] == ""
    assert (output / "_SUCCESS").exists()
    assert not (output / "_FAILED").exists()
