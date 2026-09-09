"""Vectorized preprocessing for accepted ASSIST2017 and full Junyi CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SESSION_GAP_SECONDS = 10 * 60 * 60

FLAT_EVENT_SCHEMA = pa.schema(
    [
        ("student_id", pa.int32()),
        ("question_id", pa.int32()),
        ("skill_id", pa.int32()),
        ("correct", pa.uint8()),
        ("timestamp", pa.int64()),
        ("source_row", pa.int32()),
        ("source_event_id", pa.int64()),
        ("session_id", pa.int32()),
        ("session_position", pa.int32()),
        ("question_attempt_no", pa.int32()),
        ("skill_attempt_no", pa.int32()),
        ("split", pa.uint8()),
    ]
)


class FlatDataValidationError(ValueError):
    """Raised when an accepted flat source violates its declared schema."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    required_columns: tuple[str, ...]
    event_order_column: str | None
    expected_rows: int
    expected_sha256: str


CONFIGS = {
    "assist2017": DatasetConfig(
        name="assist2017",
        required_columns=(
            "studentId",
            "action_num",
            "skill",
            "problemId",
            "correct",
            "startTime",
        ),
        event_order_column="action_num",
        expected_rows=942_807,
        expected_sha256="7577287d4ead073fdb9393b586a260043151f27338246bd076ca740e28be8bbb",
    ),
    "junyi": DatasetConfig(
        name="junyi",
        required_columns=("studentId", "skill", "problemId", "correct", "startTime"),
        event_order_column=None,
        expected_rows=14_660_217,
        expected_sha256="8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_ids(values: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    originals = np.sort(values.unique())
    ids = np.searchsorted(originals, values.to_numpy()) + 1
    mapping = pd.DataFrame({"id": np.arange(1, len(originals) + 1), "source_id": originals})
    return ids.astype(np.int32), mapping


def rebuild_sessions_and_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild session, split, position, and attempt fields after filtering."""

    if frame.empty:
        raise FlatDataValidationError("cannot build sequences from an empty frame")
    gaps = frame.groupby("student_id", sort=False)["startTime"].diff()
    boundary = gaps.isna() | gaps.ge(SESSION_GAP_SECONDS)
    frame["session_id"] = (
        boundary.groupby(frame["student_id"], sort=False).cumsum().astype(np.int32)
    )
    session_count = frame.groupby("student_id", sort=False)["session_id"].transform("max")
    frame = frame.loc[session_count.ge(5)].copy()
    if frame.empty:
        raise FlatDataValidationError("no students remain after the five-session filter")

    # Reindex retained students only, preserving source-student numeric order.
    retained_originals = np.sort(frame["source_student_id"].unique())
    frame["student_id"] = (
        np.searchsorted(retained_originals, frame["source_student_id"].to_numpy()) + 1
    ).astype(np.int32)
    session_count = frame.groupby("student_id", sort=False)["session_id"].transform("max")
    holdout = session_count.floordiv(5)
    train_end = session_count - 2 * holdout
    validation_end = session_count - holdout
    split = np.zeros(len(frame), dtype=np.uint8)
    split[frame["session_id"].to_numpy() > train_end.to_numpy()] = 1
    split[frame["session_id"].to_numpy() > validation_end.to_numpy()] = 2
    frame["split"] = split
    frame["session_position"] = (
        frame.groupby(["student_id", "session_id"], sort=False).cumcount() + 1
    ).astype(np.int32)
    frame["question_attempt_no"] = (
        frame.groupby(["student_id", "question_id"], sort=False).cumcount() + 1
    ).astype(np.int32)
    frame["skill_attempt_no"] = (
        frame.groupby(["student_id", "skill_id"], sort=False).cumcount() + 1
    ).astype(np.int32)
    return frame


def _write_event_shards(frame: pd.DataFrame, directory: Path, rows_per_shard: int) -> int:
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    output = frame[
        [
            "student_id",
            "question_id",
            "skill_id",
            "correct",
            "timestamp",
            "source_row",
            "source_event_id",
            "session_id",
            "session_position",
            "question_attempt_no",
            "skill_attempt_no",
            "split",
        ]
    ]
    shard_count = 0
    for start in range(0, len(output), rows_per_shard):
        shard = output.iloc[start : start + rows_per_shard]
        table = pa.Table.from_pandas(shard, schema=FLAT_EVENT_SCHEMA, preserve_index=False)
        pq.write_table(
            table,
            directory / f"part-{shard_count:05d}.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        shard_count += 1
    return shard_count


def _render_report(summary: dict[str, object]) -> str:
    order = (
        f"authoritative `{summary['event_order_column']}` with source row as fallback"
        if summary["event_order_column"]
        else "original CSV row order for equal timestamps"
    )
    skill_policy = (
        "The supplied skill on each interaction is preserved. No authoritative "
        "complete question-tag mapping is available, so skills are not merged "
        "across rows and interactions are never expanded."
        if summary["dataset"] == "assist2017"
        else "Each question's complete set of tags observed in the accepted "
        "full source is deduplicated and numerically sorted into one composite "
        "skill. Every occurrence receives that one ID; rows are never expanded."
    )
    return f"""# {summary['dataset']} preprocessing report

- Source rows: {summary['source_rows']:,}
- Source SHA-256: `{summary['source_sha256']}`
- Source students: {summary['source_students']:,}
- Questions: {summary['questions']:,}
- Genuine original tags/skills: {summary['original_skills']:,}
- Deterministic composite skill IDs: {summary['composite_skills']:,}
- Questions with multiple observed row-level skills: {summary['multi_skill_questions']:,}
- Interactions on those questions: {summary['multi_skill_question_interactions']:,} ({summary['multi_skill_question_interaction_percentage']:.6f}%)
- Missing target labels: 0
- Retained students (at least five rebuilt sessions): {summary['retained_students']:,}
- Excluded students: {summary['excluded_students']:,}
- Retained interactions: {summary['retained_interactions']:,}
- Retained sessions: {summary['retained_sessions']:,}
- Train interactions: {summary['split_interactions']['train']:,}
- Validation interactions: {summary['split_interactions']['validation']:,}
- Test interactions: {summary['split_interactions']['test']:,}

IDs are determined by numeric sorting of the complete source vocabularies, with
`0` reserved for padding. {skill_policy} Rows are ordered per student by
timestamp and {order}. Sessions are rebuilt at
gaps of at least 10 hours. Per-student sessions use chronological 60/20/20
splits with `floor(n/5)` validation and test sessions and earlier remainder
sessions in training. All positions and attempt counters are rebuilt afterward.
"""


def preprocess_flat(
    dataset: str,
    source_csv: Path,
    output_dir: Path,
    *,
    rows_per_shard: int = 1_000_000,
    verify_hash: bool = True,
) -> dict[str, object]:
    if dataset not in CONFIGS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(CONFIGS)}")
    config = CONFIGS[dataset]
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    events_dir = output_dir / "events"
    mappings_dir = output_dir / "mappings"
    reports_dir = output_dir / "reports"
    events_dir.mkdir()
    mappings_dir.mkdir()
    reports_dir.mkdir()
    try:
        source_hash = sha256_file(source_csv)
        if verify_hash and source_hash != config.expected_sha256:
            raise FlatDataValidationError(
                f"unexpected {dataset} SHA-256: {source_hash}; expected {config.expected_sha256}"
            )
        frame = pd.read_csv(source_csv, usecols=list(config.required_columns))
        if len(frame) != config.expected_rows and verify_hash:
            raise FlatDataValidationError(
                f"unexpected {dataset} row count: {len(frame)}; expected {config.expected_rows}"
            )
        if frame[list(config.required_columns)].isna().any().any():
            raise FlatDataValidationError("missing required source value")
        if not frame["correct"].isin([0, 1]).all():
            values = sorted(frame.loc[~frame["correct"].isin([0, 1]), "correct"].unique())
            raise FlatDataValidationError(f"ambiguous target encodings: {values!r}")
        for column in ("studentId", "skill", "problemId", "startTime"):
            if not pd.api.types.is_integer_dtype(frame[column]):
                raise FlatDataValidationError(f"{column} is not integer-valued")

        frame["source_row"] = np.arange(1, len(frame) + 1, dtype=np.int32)
        sort_columns = ["studentId", "startTime"]
        if config.event_order_column:
            if frame.duplicated(["studentId", config.event_order_column]).any():
                raise FlatDataValidationError(
                    f"duplicate authoritative {config.event_order_column} within student"
                )
            sort_columns.append(config.event_order_column)
        sort_columns.append("source_row")
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        frame = frame.rename(columns={"studentId": "source_student_id"})

        question_ids, question_map = _stable_ids(frame["problemId"])
        frame["question_id"] = question_ids
        question_map.columns = ["question_id", "source_question_id"]

        _, original_tag_map = _stable_ids(frame["skill"])
        original_tag_map.columns = ["tag_index", "source_tag_id"]
        source_question_tags = frame[["problemId", "skill"]].drop_duplicates()
        tag_counts = source_question_tags.groupby("problemId")["skill"].nunique()
        multi_skill_source_questions = set(tag_counts[tag_counts > 1].index.tolist())
        multi_skill_interaction_count = int(
            frame["problemId"].isin(multi_skill_source_questions).sum()
        )

        if dataset == "assist2017":
            skill_ids, source_skill_map = _stable_ids(frame["skill"])
            frame["skill_id"] = skill_ids
            source_skill_map.columns = ["skill_id", "source_skill_id"]
            skill_map = source_skill_map[["skill_id"]].copy()
            skill_map["canonical_skill_key"] = source_skill_map[
                "source_skill_id"
            ].astype(str)
            skill_map["original_tag_ids"] = source_skill_map[
                "source_skill_id"
            ].map(lambda value: json.dumps([int(value)], separators=(",", ":")))
            skill_map["tag_count"] = 1
            skill_map["is_untagged"] = False
            question_skill = (
                frame[["question_id", "skill_id"]]
                .drop_duplicates()
                .sort_values(["question_id", "skill_id"], kind="stable")
            )
        else:
            question_tag_sets = (
                source_question_tags.groupby("problemId", sort=True)["skill"]
                .agg(lambda values: tuple(sorted(set(int(value) for value in values))))
            )
            canonical_by_question = question_tag_sets.map(
                lambda values: ";".join(str(value) for value in values)
            )
            composite_keys = sorted(
                canonical_by_question.unique(),
                key=lambda key: tuple(int(token) for token in key.split(";")),
            )
            skill_id_by_key = {
                key: index + 1 for index, key in enumerate(composite_keys)
            }
            skill_map = pd.DataFrame(
                {
                    "skill_id": np.arange(1, len(composite_keys) + 1),
                    "canonical_skill_key": composite_keys,
                }
            )
            skill_map["original_tag_ids"] = skill_map["canonical_skill_key"].map(
                lambda key: json.dumps(
                    [int(token) for token in key.split(";")], separators=(",", ":")
                )
            )
            skill_map["tag_count"] = skill_map["canonical_skill_key"].map(
                lambda key: len(key.split(";"))
            )
            skill_map["is_untagged"] = False
            skill_by_source_question = canonical_by_question.map(skill_id_by_key)
            frame["skill_id"] = (
                frame["problemId"].map(skill_by_source_question).astype(np.int32)
            )
            question_skill = question_map.copy()
            question_skill["skill_id"] = question_skill["source_question_id"].map(
                skill_by_source_question
            )
            question_skill["canonical_skill_key"] = question_skill[
                "source_question_id"
            ].map(canonical_by_question)
            question_skill = question_skill.drop(columns="source_question_id")
        frame["student_id"] = frame["source_student_id"].astype(np.int32)
        frame["correct"] = frame["correct"].astype(np.uint8)
        frame["timestamp"] = frame["startTime"].astype(np.int64) * 1000
        frame["source_event_id"] = (
            frame[config.event_order_column].astype(np.int64)
            if config.event_order_column
            else np.full(len(frame), -1, dtype=np.int64)
        )

        source_students = int(frame["source_student_id"].nunique())
        retained = rebuild_sessions_and_splits(frame)
        retained_originals = np.sort(retained["source_student_id"].unique())
        student_map = pd.DataFrame(
            {
                "student_id": np.arange(1, len(retained_originals) + 1),
                "source_student_id": retained_originals,
            }
        )
        question_map.to_csv(mappings_dir / "questions.csv", index=False)
        original_tag_map.to_csv(mappings_dir / "original_tags.csv", index=False)
        skill_map.to_csv(mappings_dir / "composite_skills.csv", index=False)
        student_map.to_csv(mappings_dir / "students.csv", index=False)
        question_skill.to_csv(mappings_dir / "question_skills.csv", index=False)

        raw_distribution = frame.groupby("skill_id", sort=True).size()
        retained_distribution = retained.groupby("skill_id", sort=True).size()
        distribution = skill_map.copy()
        distribution["source_interaction_count"] = distribution["skill_id"].map(raw_distribution).fillna(0).astype(np.int64)
        distribution["retained_interaction_count"] = distribution["skill_id"].map(retained_distribution).fillna(0).astype(np.int64)
        distribution.to_csv(reports_dir / "composite_skill_distribution.csv", index=False)

        shard_count = _write_event_shards(retained, events_dir, rows_per_shard)
        session_rows = retained.drop_duplicates(["student_id", "session_id"])
        split_names = ("train", "validation", "test")
        split_interactions = {
            name: int((retained["split"] == label).sum())
            for label, name in enumerate(split_names)
        }
        split_sessions = {
            name: int((session_rows["split"] == label).sum())
            for label, name in enumerate(split_names)
        }
        summary: dict[str, object] = {
            "dataset": dataset,
            "source_sha256": source_hash,
            "source_rows": len(frame),
            "source_students": source_students,
            "questions": len(question_map),
            "original_skills": len(original_tag_map),
            "composite_skills": len(skill_map),
            "multi_skill_questions": len(multi_skill_source_questions),
            "multi_skill_question_interactions": multi_skill_interaction_count,
            "multi_skill_question_interaction_percentage": (
                100.0 * multi_skill_interaction_count / len(frame)
            ),
            "event_order_column": config.event_order_column,
            "retained_students": len(student_map),
            "excluded_students": source_students - len(student_map),
            "retained_interactions": len(retained),
            "retained_sessions": len(session_rows),
            "split_interactions": split_interactions,
            "split_sessions": split_sessions,
            "event_shards": shard_count,
            "session_gap_seconds": SESSION_GAP_SECONDS,
            "minimum_sessions": 5,
        }
        (reports_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (reports_dir / "dataset_report.md").write_text(
            _render_report(summary), encoding="utf-8"
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "event_schema": str(FLAT_EVENT_SCHEMA),
                    "event_shards": shard_count,
                    "retained_rows": len(retained),
                    "rows_per_shard": rows_per_shard,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "_SUCCESS").write_text("\n", encoding="utf-8")
        return summary
    except Exception:
        (output_dir / "_FAILED").write_text(
            "Preprocessing stopped before completion; inspect the raised error.\n",
            encoding="utf-8",
        )
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ASSIST2017 or full Junyi")
    parser.add_argument("dataset", choices=sorted(CONFIGS))
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    parser.add_argument("--skip-hash-check", action="store_true", help="Synthetic tests only")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = preprocess_flat(
        args.dataset,
        args.source_csv,
        args.output_dir,
        rows_per_shard=args.rows_per_shard,
        verify_hash=not args.skip_hash_check,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
