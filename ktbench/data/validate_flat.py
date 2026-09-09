"""Independent streaming validation for ASSIST2017 and Junyi event builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ktbench.data.flat import FLAT_EVENT_SCHEMA, SESSION_GAP_SECONDS


class FlatArtifactValidationError(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FlatArtifactValidationError(message)


def validate_flat(root: Path) -> dict[str, int]:
    _require((root / "_SUCCESS").is_file(), "missing _SUCCESS")
    _require(not (root / "_FAILED").exists(), "_FAILED is present")
    summary = json.loads((root / "reports/summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    questions = pd.read_csv(root / "mappings/questions.csv")
    skills = pd.read_csv(root / "mappings/composite_skills.csv")
    tags = pd.read_csv(root / "mappings/original_tags.csv")
    students = pd.read_csv(root / "mappings/students.csv")
    question_skills = pd.read_csv(root / "mappings/question_skills.csv")
    distribution = pd.read_csv(root / "reports/composite_skill_distribution.csv")
    files = sorted((root / "events").glob("part-*.parquet"))
    _require(len(files) == manifest["event_shards"], "event shard mismatch")

    rows = 0
    student_count = 0
    session_count = 0
    split_rows = np.zeros(3, dtype=np.int64)
    split_sessions = np.zeros(3, dtype=np.int64)
    last: dict[str, int] | None = None
    for path in files:
        parquet = pq.ParquetFile(path)
        _require(parquet.schema_arrow == FLAT_EVENT_SCHEMA, f"schema mismatch: {path}")
        for batch in parquet.iter_batches(batch_size=262_144):
            _require(
                sum(column.null_count for column in batch.columns) == 0,
                f"null event value: {path}",
            )
            c = {
                name: batch.column(index).to_numpy(zero_copy_only=False)
                for index, name in enumerate(FLAT_EVENT_SCHEMA.names)
            }
            size = len(c["student_id"])
            if not size:
                continue
            rows += size
            student = c["student_id"].astype(np.int64, copy=False)
            timestamp = c["timestamp"].astype(np.int64, copy=False)
            source_row = c["source_row"].astype(np.int64, copy=False)
            source_event = c["source_event_id"].astype(np.int64, copy=False)
            session = c["session_id"].astype(np.int64, copy=False)
            position = c["session_position"].astype(np.int64, copy=False)
            split = c["split"].astype(np.int64, copy=False)
            correct = c["correct"].astype(np.int64, copy=False)

            _require(np.isin(correct, [0, 1]).all(), f"non-binary target: {path}")
            _require(np.isin(split, [0, 1, 2]).all(), f"invalid split: {path}")
            _require(
                ((c["question_id"] >= 1) & (c["question_id"] <= len(questions))).all(),
                f"question ID outside mapping: {path}",
            )
            _require(
                ((c["skill_id"] >= 1) & (c["skill_id"] <= len(skills))).all(),
                f"skill ID outside mapping: {path}",
            )

            if last is None:
                previous_student = np.concatenate(([-1], student[:-1]))
                previous_timestamp = np.concatenate(([-1], timestamp[:-1]))
                previous_source_row = np.concatenate(([-1], source_row[:-1]))
                previous_source_event = np.concatenate(([-1], source_event[:-1]))
                previous_session = np.concatenate(([0], session[:-1]))
                previous_position = np.concatenate(([0], position[:-1]))
                previous_split = np.concatenate(([0], split[:-1]))
            else:
                previous_student = np.concatenate(([last["student"]], student[:-1]))
                previous_timestamp = np.concatenate(([last["timestamp"]], timestamp[:-1]))
                previous_source_row = np.concatenate(([last["source_row"]], source_row[:-1]))
                previous_source_event = np.concatenate(([last["source_event"]], source_event[:-1]))
                previous_session = np.concatenate(([last["session"]], session[:-1]))
                previous_position = np.concatenate(([last["position"]], position[:-1]))
                previous_split = np.concatenate(([last["split"]], split[:-1]))

            new_student = student != previous_student
            same_student = ~new_student
            _require((student >= previous_student).all(), f"student order regressed: {path}")
            _require(
                (timestamp[same_student] >= previous_timestamp[same_student]).all(),
                f"timestamp order regressed: {path}",
            )
            tied = same_student & (timestamp == previous_timestamp)
            if summary["event_order_column"]:
                _require(
                    (source_event[tied] > previous_source_event[tied]).all(),
                    f"authoritative tie order regressed: {path}",
                )
            else:
                _require(
                    (source_row[tied] > previous_source_row[tied]).all(),
                    f"source-row tie order regressed: {path}",
                )
                _require((source_event == -1).all(), f"unexpected event index: {path}")

            _require((session[new_student] == 1).all(), f"session not reset: {path}")
            session_changed = session != previous_session
            expected_change = (
                timestamp - previous_timestamp >= SESSION_GAP_SECONDS * 1000
            )
            _require(
                (session_changed[same_student] == expected_change[same_student]).all(),
                f"session boundary differs from 10-hour rule: {path}",
            )
            new_session = new_student | session_changed
            _require((position[new_session] == 1).all(), f"position not reset: {path}")
            continuing = ~new_session
            _require(
                (position[continuing] == previous_position[continuing] + 1).all(),
                f"position not contiguous: {path}",
            )
            _require(
                (split[same_student] >= previous_split[same_student]).all(),
                f"split order regressed: {path}",
            )
            _require(
                (split[continuing] == previous_split[continuing]).all(),
                f"split changes within session: {path}",
            )
            student_count += int(new_student.sum())
            session_count += int(new_session.sum())
            split_rows += np.bincount(split, minlength=3)
            split_sessions += np.bincount(split[new_session], minlength=3)
            last = {
                "student": int(student[-1]),
                "timestamp": int(timestamp[-1]),
                "source_row": int(source_row[-1]),
                "source_event": int(source_event[-1]),
                "session": int(session[-1]),
                "position": int(position[-1]),
                "split": int(split[-1]),
            }

    split_names = ("train", "validation", "test")
    expected_rows = np.array([summary["split_interactions"][name] for name in split_names])
    expected_sessions = np.array([summary["split_sessions"][name] for name in split_names])
    _require(rows == summary["retained_interactions"] == manifest["retained_rows"], "row count mismatch")
    _require(student_count == summary["retained_students"] == len(students), "student count mismatch")
    _require(session_count == summary["retained_sessions"], "session count mismatch")
    _require(np.array_equal(split_rows, expected_rows), "split interaction mismatch")
    _require(np.array_equal(split_sessions, expected_sessions), "split session mismatch")
    _require(len(questions) == summary["questions"], "question count mismatch")
    _require(len(tags) == summary["original_skills"], "original tag count mismatch")
    _require(len(skills) == summary["composite_skills"], "composite count mismatch")
    _require(distribution["source_interaction_count"].sum() == summary["source_rows"], "source distribution mismatch")
    _require(distribution["retained_interaction_count"].sum() == rows, "retained distribution mismatch")
    if summary["dataset"] == "junyi":
        _require(question_skills["question_id"].nunique() == len(question_skills) == len(questions), "Junyi question composite mapping is not one-to-one")
    return {
        "validated_event_rows": rows,
        "validated_event_shards": len(files),
        "validated_students": student_count,
        "validated_sessions": session_count,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ASSIST2017 or Junyi artifacts")
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_flat(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
