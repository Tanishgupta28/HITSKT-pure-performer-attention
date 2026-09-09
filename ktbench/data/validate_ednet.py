"""Independent validation for generated EdNet-KT1 benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ktbench.data.ednet import EVENT_SCHEMA, UNANSWERED_SCHEMA


class ProcessedDataValidationError(AssertionError):
    """Raised when processed artifacts violate their manifest contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProcessedDataValidationError(message)


def _parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("part-*.parquet"))
    _require(bool(files), f"no Parquet shards found in {directory}")
    return files


def validate_processed_ednet(root: Path) -> dict[str, int]:
    """Stream all supervised shards and cross-check audit/mapping totals."""

    _require((root / "_SUCCESS").is_file(), "missing _SUCCESS marker")
    _require(not (root / "_FAILED").exists(), "_FAILED marker is present")
    summary = json.loads((root / "reports" / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    _require(manifest["complete"] is True, "manifest is not complete")

    row_count = 0
    student_count = 0
    session_count = 0
    split_rows = np.zeros(3, dtype=np.int64)
    split_sessions = np.zeros(3, dtype=np.int64)
    max_question_id = 0
    last: dict[str, int] | None = None
    event_files = _parquet_files(root / "events")
    _require(len(event_files) == manifest["event_shards"], "event shard mismatch")

    for path in event_files:
        parquet = pq.ParquetFile(path)
        _require(parquet.schema_arrow == EVENT_SCHEMA, f"schema mismatch in {path}")
        for batch in parquet.iter_batches(batch_size=262_144):
            _require(
                sum(column.null_count for column in batch.columns) == 0,
                f"null supervised value in {path}",
            )
            columns = {
                name: batch.column(index).to_numpy(zero_copy_only=False)
                for index, name in enumerate(EVENT_SCHEMA.names)
            }
            size = len(columns["student_id"])
            if size == 0:
                continue
            row_count += size
            student = columns["student_id"].astype(np.int64, copy=False)
            timestamp = columns["timestamp"].astype(np.int64, copy=False)
            source_row = columns["source_row"].astype(np.int64, copy=False)
            session = columns["session_id"].astype(np.int64, copy=False)
            position = columns["session_position"].astype(np.int64, copy=False)
            split = columns["split"].astype(np.int64, copy=False)
            correct = columns["correct"].astype(np.int64, copy=False)

            _require(np.isin(correct, [0, 1]).all(), f"non-binary label in {path}")
            _require(np.isin(split, [0, 1, 2]).all(), f"invalid split in {path}")
            _require(
                (columns["question_id"] >= 1).all(),
                f"non-positive question ID in {path}",
            )
            _require(
                ((columns["skill_id"] >= 1) & (columns["skill_id"] <= summary["composite_skills"])).all(),
                f"out-of-range skill ID in {path}",
            )

            if last is None:
                previous_student = np.concatenate(([-1], student[:-1]))
                previous_timestamp = np.concatenate(([-1], timestamp[:-1]))
                previous_source = np.concatenate(([-1], source_row[:-1]))
                previous_session = np.concatenate(([0], session[:-1]))
                previous_position = np.concatenate(([0], position[:-1]))
                previous_split = np.concatenate(([0], split[:-1]))
            else:
                previous_student = np.concatenate(([last["student"]], student[:-1]))
                previous_timestamp = np.concatenate(([last["timestamp"]], timestamp[:-1]))
                previous_source = np.concatenate(([last["source_row"]], source_row[:-1]))
                previous_session = np.concatenate(([last["session"]], session[:-1]))
                previous_position = np.concatenate(([last["position"]], position[:-1]))
                previous_split = np.concatenate(([last["split"]], split[:-1]))

            new_student = student != previous_student
            same_student = ~new_student
            _require(
                (student >= previous_student).all(),
                f"student order regressed in {path}",
            )
            if np.any(new_student):
                expected = previous_student[new_student] + 1
                # The first student has no predecessor; all later IDs are contiguous.
                if last is None and new_student[0]:
                    expected[0] = student[0]
                _require(
                    (student[new_student] == expected).all(),
                    f"non-contiguous retained student IDs in {path}",
                )
            _require(
                (timestamp[same_student] >= previous_timestamp[same_student]).all(),
                f"timestamp order regressed in {path}",
            )
            tied = same_student & (timestamp == previous_timestamp)
            _require(
                (source_row[tied] > previous_source[tied]).all(),
                f"source-row tie order regressed in {path}",
            )
            _require((session[new_student] == 1).all(), f"student session not reset in {path}")
            session_delta = session[same_student] - previous_session[same_student]
            _require(
                np.isin(session_delta, [0, 1]).all(),
                f"non-contiguous session IDs in {path}",
            )
            new_session = new_student | (session != previous_session)
            _require((position[new_session] == 1).all(), f"session position not reset in {path}")
            continuing = ~new_session
            _require(
                (position[continuing] == previous_position[continuing] + 1).all(),
                f"session position is non-contiguous in {path}",
            )
            _require(
                (split[same_student] >= previous_split[same_student]).all(),
                f"split chronology regressed in {path}",
            )
            _require(
                (split[continuing] == previous_split[continuing]).all(),
                f"split changes inside a session in {path}",
            )

            student_count += int(new_student.sum())
            session_count += int(new_session.sum())
            split_rows += np.bincount(split, minlength=3)
            split_sessions += np.bincount(split[new_session], minlength=3)
            max_question_id = max(max_question_id, int(columns["question_id"].max()))
            last = {
                "student": int(student[-1]),
                "timestamp": int(timestamp[-1]),
                "source_row": int(source_row[-1]),
                "session": int(session[-1]),
                "position": int(position[-1]),
                "split": int(split[-1]),
            }

    audit_files = _parquet_files(root / "audit" / "unanswered_events")
    _require(
        len(audit_files) == manifest["unanswered_audit_shards"],
        "unanswered shard mismatch",
    )
    audit_frames = []
    for path in audit_files:
        parquet = pq.ParquetFile(path)
        _require(parquet.schema_arrow == UNANSWERED_SCHEMA, f"audit schema mismatch in {path}")
        audit_frames.append(parquet.read().to_pandas())
    audit = pd.concat(audit_frames, ignore_index=True)
    _require(not audit.isna().any().any(), "null value in unanswered audit")
    _require((audit["user_answer"] == "").all(), "nonempty answer in unanswered audit")

    users = pd.read_csv(root / "mappings" / "students.csv")
    questions = pd.read_csv(root / "mappings" / "questions.csv")
    tags = pd.read_csv(root / "mappings" / "original_tags.csv")
    composites = pd.read_csv(root / "mappings" / "composite_skills.csv")
    distribution = pd.read_csv(root / "reports" / "composite_skill_distribution.csv")
    impacts = pd.read_csv(root / "audit" / "session_impacts.csv")

    expected_split_rows = np.array(
        [summary["split_interactions"][name] for name in ("train", "validation", "test")]
    )
    expected_split_sessions = np.array(
        [summary["split_sessions"][name] for name in ("train", "validation", "test")]
    )
    _require(row_count == summary["retained_interactions"] == manifest["retained_rows"], "retained row mismatch")
    _require(student_count == summary["retained_students"] == len(users), "retained student mismatch")
    _require(session_count == summary["retained_sessions"], "retained session mismatch")
    _require(np.array_equal(split_rows, expected_split_rows), "split row mismatch")
    _require(np.array_equal(split_sessions, expected_split_sessions), "split session mismatch")
    _require(len(audit) == summary["unanswered_interactions"] == manifest["unanswered_audit_rows"], "unanswered row mismatch")
    _require(audit["original_student_id"].nunique() == summary["unanswered_affected_students"], "unanswered student mismatch")
    _require(max_question_id <= len(questions), "question ID exceeds persisted mapping")
    _require(len(tags) == summary["genuine_original_tags"], "original tag count mismatch")
    _require(len(composites) == summary["composite_skills"], "composite count mismatch")
    _require(int(composites.iloc[0]["skill_id"]) == 1 and composites.iloc[0]["canonical_skill_key"] == "<UNTAGGED>", "untagged mapping mismatch")
    _require(distribution["raw_interaction_count"].sum() == summary["raw_interactions"], "raw distribution mismatch")
    _require(distribution["supervised_interaction_count"].sum() == summary["supervised_interactions"], "supervised distribution mismatch")
    _require(distribution["retained_interaction_count"].sum() == summary["retained_interactions"], "retained distribution mismatch")
    _require(impacts["unanswered_interactions"].sum() == summary["unanswered_interactions"], "session-impact unanswered mismatch")
    _require((impacts["session_count_delta"] != 0).sum() == summary["session_count_changed_students"], "session-count impact mismatch")
    _require(impacts["answered_grouping_changed"].sum() == summary["session_grouping_changed_students"], "session-grouping impact mismatch")

    return {
        "validated_event_rows": row_count,
        "validated_students": student_count,
        "validated_sessions": session_count,
        "validated_unanswered_rows": len(audit),
        "validated_event_shards": len(event_files),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed EdNet-KT1 artifacts")
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the successful validation summary to this JSON path",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_processed_ednet(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
