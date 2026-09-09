"""Deterministic, streaming preprocessing for the official EdNet-KT1 data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


UNTAGGED_KEY = "<UNTAGGED>"
MISSING_TAG_SENTINEL = "-1"
PAD_ID = 0
UNTAGGED_SKILL_ID = 1
SESSION_GAP_MS = 10 * 60 * 60 * 1000
EXPECTED_QUESTION_COLUMNS = (
    "question_id",
    "bundle_id",
    "explanation_id",
    "correct_answer",
    "part",
    "tags",
    "deployed_at",
)
EXPECTED_INTERACTION_COLUMNS = (
    "timestamp",
    "solving_id",
    "question_id",
    "user_answer",
    "elapsed_time",
)
ID_PATTERN = re.compile(r"^([qu])(\d+)$")
TAG_PATTERN = re.compile(r"^[0-9]+(?:;[0-9]+)*$")


class EdNetValidationError(ValueError):
    """Raised when source data violates the approved preprocessing contract."""


@dataclass(frozen=True)
class TagSet:
    canonical_key: str
    tags: tuple[int, ...]
    is_untagged: bool


@dataclass(frozen=True)
class MetadataMappings:
    questions: pd.DataFrame
    composites: pd.DataFrame
    original_tags: pd.DataFrame


def _numeric_prefixed_id(value: str, prefix: str, field: str) -> int:
    match = ID_PATTERN.fullmatch(value)
    if match is None or match.group(1) != prefix:
        raise EdNetValidationError(f"malformed {field}: {value!r}")
    return int(match.group(2))


def canonicalize_tags(raw_value: object) -> TagSet:
    """Validate and canonicalize one official EdNet question tag value.

    The only approved missing representation is the exact string ``-1``.
    Positive IDs are interpreted as a mathematical set: duplicates are removed
    before numeric sorting. Any other missing, negative, whitespace-padded, or
    malformed representation fails closed so it can be reviewed rather than
    inferred.
    """

    if raw_value is None or pd.isna(raw_value):
        raise EdNetValidationError("missing tag cell is not the approved -1 sentinel")
    raw = str(raw_value)
    if raw == MISSING_TAG_SENTINEL:
        return TagSet(UNTAGGED_KEY, (), True)
    if not TAG_PATTERN.fullmatch(raw):
        raise EdNetValidationError(f"unapproved tag representation: {raw!r}")
    tags = tuple(int(token) for token in raw.split(";"))
    if any(tag <= 0 for tag in tags):
        raise EdNetValidationError(f"non-positive genuine tag ID: {raw!r}")
    sorted_tags = tuple(sorted(set(tags)))
    return TagSet(";".join(str(tag) for tag in sorted_tags), sorted_tags, False)


def load_question_metadata(contents_zip: Path) -> pd.DataFrame:
    with ZipFile(contents_zip) as archive:
        member = "contents/questions.csv"
        if member not in archive.namelist():
            raise EdNetValidationError(f"missing {member!r} in {contents_zip}")
        with archive.open(member) as stream:
            questions = pd.read_csv(stream, dtype=str, keep_default_na=False)

    columns = tuple(questions.columns)
    if columns != EXPECTED_QUESTION_COLUMNS:
        raise EdNetValidationError(
            f"unexpected questions.csv columns: {columns!r}; "
            f"expected {EXPECTED_QUESTION_COLUMNS!r}"
        )
    if questions.empty:
        raise EdNetValidationError("questions.csv is empty")
    if questions["question_id"].duplicated().any():
        duplicates = questions.loc[
            questions["question_id"].duplicated(), "question_id"
        ].tolist()[:5]
        raise EdNetValidationError(f"duplicate question IDs: {duplicates!r}")
    return questions


def build_metadata_mappings(questions: pd.DataFrame) -> MetadataMappings:
    """Build stable question, genuine-tag, and composite-skill mappings."""

    records: list[dict[str, object]] = []
    genuine_tags: set[int] = set()
    question_numbers: set[int] = set()
    for row in questions.itertuples(index=False):
        question_number = _numeric_prefixed_id(
            str(row.question_id), "q", "question_id"
        )
        if question_number in question_numbers:
            raise EdNetValidationError(
                f"numeric duplicate question ID: q{question_number}"
            )
        question_numbers.add(question_number)
        tag_set = canonicalize_tags(row.tags)
        raw_tag_tokens = (
            []
            if tag_set.is_untagged
            else [int(token) for token in str(row.tags).split(";")]
        )
        genuine_tags.update(tag_set.tags)
        answer = str(row.correct_answer)
        if answer not in {"a", "b", "c", "d"}:
            raise EdNetValidationError(
                f"invalid correct_answer for {row.question_id}: {answer!r}"
            )
        records.append(
            {
                "question_key": str(row.question_id),
                "question_number": question_number,
                "correct_answer": answer,
                "canonical_skill_key": tag_set.canonical_key,
                "is_untagged": tag_set.is_untagged,
                "is_multitag": len(tag_set.tags) > 1,
                "had_duplicate_tags": len(raw_tag_tokens) != len(set(raw_tag_tokens)),
                "tag_count": len(tag_set.tags),
            }
        )

    question_frame = pd.DataFrame.from_records(records)
    genuine_keys = sorted(
        set(question_frame.loc[~question_frame["is_untagged"], "canonical_skill_key"]),
        key=lambda key: tuple(int(token) for token in key.split(";")),
    )
    skill_by_key = {UNTAGGED_KEY: UNTAGGED_SKILL_ID}
    skill_by_key.update({key: index + 2 for index, key in enumerate(genuine_keys)})

    question_frame["skill_id"] = question_frame["canonical_skill_key"].map(skill_by_key)
    question_frame = question_frame.sort_values("question_number", kind="stable")
    question_frame.insert(0, "question_id", np.arange(1, len(question_frame) + 1))
    question_frame = question_frame.reset_index(drop=True)

    question_counts = Counter(question_frame["canonical_skill_key"])
    composite_records = []
    for key, skill_id in sorted(skill_by_key.items(), key=lambda item: item[1]):
        is_untagged = key == UNTAGGED_KEY
        tag_ids = [] if is_untagged else [int(token) for token in key.split(";")]
        composite_records.append(
            {
                "skill_id": skill_id,
                "canonical_skill_key": key,
                "is_untagged": is_untagged,
                "tag_count": len(tag_ids),
                "original_tag_ids": json.dumps(tag_ids, separators=(",", ":")),
                "question_count": question_counts[key],
            }
        )
    composites = pd.DataFrame.from_records(composite_records)
    original_tags = pd.DataFrame(
        {
            "tag_index": np.arange(1, len(genuine_tags) + 1),
            "original_tag_id": sorted(genuine_tags),
        }
    )
    return MetadataMappings(question_frame, composites, original_tags)


def assign_sessions_and_splits(
    timestamps: Sequence[int], session_gap_ms: int = SESSION_GAP_MS
) -> tuple[np.ndarray, np.ndarray, int]:
    """Assign 1-based sessions and train/validation/test labels.

    Split labels are 0=train, 1=validation, and 2=test. With at least five
    sessions, each holdout receives ``floor(n_sessions / 5)`` sessions and the
    remainder stays in training, matching the established project convention.
    """

    if not timestamps:
        raise EdNetValidationError("cannot sessionize an empty interaction sequence")
    values = np.asarray(timestamps, dtype=np.int64)
    if np.any(np.diff(values) < 0):
        raise EdNetValidationError("timestamps must be chronologically sorted")
    starts = np.empty(len(values), dtype=np.bool_)
    starts[0] = True
    if len(values) > 1:
        starts[1:] = np.diff(values) >= session_gap_ms
    session_ids = np.cumsum(starts, dtype=np.int32)
    session_count = int(session_ids[-1])
    holdout = session_count // 5
    train_sessions = session_count - 2 * holdout
    splits = np.zeros(len(values), dtype=np.uint8)
    if holdout:
        splits[session_ids > train_sessions] = 1
        splits[session_ids > train_sessions + holdout] = 2
    return session_ids, splits, session_count


def _member_user_number(member: str) -> int:
    path = Path(member)
    if path.parent.as_posix() != "KT1" or path.suffix != ".csv":
        raise EdNetValidationError(f"unexpected KT1 archive member: {member!r}")
    return _numeric_prefixed_id(path.stem, "u", "student filename")


def sorted_student_members(archive: ZipFile) -> list[tuple[int, str]]:
    members = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        members.append((_member_user_number(info.filename), info.filename))
    members.sort(key=lambda item: item[0])
    user_numbers = [item[0] for item in members]
    if len(user_numbers) != len(set(user_numbers)):
        raise EdNetValidationError("duplicate numeric student filenames")
    return members


EVENT_SCHEMA = pa.schema(
    [
        ("student_id", pa.int32()),
        ("question_id", pa.int32()),
        ("skill_id", pa.int32()),
        ("correct", pa.uint8()),
        ("timestamp", pa.int64()),
        ("elapsed_time", pa.int32()),
        ("solving_id", pa.int32()),
        ("source_row", pa.int32()),
        ("session_id", pa.int32()),
        ("session_position", pa.int32()),
        ("question_attempt_no", pa.int32()),
        ("skill_attempt_no", pa.int32()),
        ("split", pa.uint8()),
    ]
)

UNANSWERED_SCHEMA = pa.schema(
    [
        ("original_student_id", pa.int32()),
        ("question_key", pa.string()),
        ("question_id", pa.int32()),
        ("skill_id", pa.int32()),
        ("timestamp", pa.int64()),
        ("elapsed_time", pa.int32()),
        ("solving_id", pa.int32()),
        ("source_row", pa.int32()),
        ("raw_session_id", pa.int32()),
        ("user_answer", pa.string()),
    ]
)


class ShardWriter:
    def __init__(
        self, directory: Path, rows_per_shard: int, schema: pa.Schema = EVENT_SCHEMA
    ) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.directory = directory
        self.rows_per_shard = rows_per_shard
        self.schema = schema
        self.columns: dict[str, list[object]] = {name: [] for name in schema.names}
        self.shard_index = 0
        self.total_rows = 0

    def append(self, values: dict[str, Sequence[object]]) -> None:
        lengths = {len(values[name]) for name in self.schema.names}
        if len(lengths) != 1:
            raise RuntimeError(f"event column length mismatch: {lengths!r}")
        row_count = lengths.pop()
        offset = 0
        while offset < row_count:
            capacity = self.rows_per_shard - len(self.columns[self.schema.names[0]])
            take = min(capacity, row_count - offset)
            for name in self.schema.names:
                self.columns[name].extend(values[name][offset : offset + take])
            offset += take
            if len(self.columns[self.schema.names[0]]) == self.rows_per_shard:
                self.flush()

    def flush(self) -> None:
        row_count = len(self.columns[self.schema.names[0]])
        if not row_count:
            return
        table = pa.Table.from_pydict(self.columns, schema=self.schema)
        target = self.directory / f"part-{self.shard_index:05d}.parquet"
        pq.write_table(
            table,
            target,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        self.total_rows += row_count
        self.shard_index += 1
        self.columns = {name: [] for name in self.schema.names}

    def close(self) -> None:
        self.flush()


def _read_student(
    archive: ZipFile,
    member: str,
    user_number: int,
    question_lookup: dict[str, tuple[int, int, str, bool, bool]],
) -> tuple[dict[str, list[int]], dict[str, list[object]], dict[str, int]]:
    records = []
    raw_nonmonotonic = 0
    previous_raw_timestamp: int | None = None
    with archive.open(member) as binary_stream:
        text_stream = (line.decode("utf-8") for line in binary_stream)
        reader = csv.reader(text_stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise EdNetValidationError(f"empty student file: {member}") from error
        if header != EXPECTED_INTERACTION_COLUMNS:
            raise EdNetValidationError(
                f"unexpected columns in {member}: {header!r}; "
                f"expected {EXPECTED_INTERACTION_COLUMNS!r}"
            )
        for source_row, row in enumerate(reader, start=1):
            if len(row) != len(EXPECTED_INTERACTION_COLUMNS):
                raise EdNetValidationError(
                    f"malformed row {source_row} in {member}: {row!r}"
                )
            try:
                timestamp = int(row[0])
                solving_id = int(row[1])
                elapsed_time = int(row[4])
            except ValueError as error:
                raise EdNetValidationError(
                    f"non-integer field at row {source_row} in {member}: {row!r}"
                ) from error
            if timestamp < 0 or solving_id < 1 or elapsed_time < 0:
                raise EdNetValidationError(
                    f"out-of-range field at row {source_row} in {member}: {row!r}"
                )
            question_key = row[2]
            if question_key not in question_lookup:
                raise EdNetValidationError(
                    f"unknown question {question_key!r} at row {source_row} in {member}"
                )
            answer = row[3]
            if answer not in {"", "a", "b", "c", "d"}:
                raise EdNetValidationError(
                    f"unapproved user_answer at row {source_row} in {member}: {answer!r}"
                )
            if previous_raw_timestamp is not None and timestamp < previous_raw_timestamp:
                raw_nonmonotonic += 1
            previous_raw_timestamp = timestamp
            question_id, skill_id, correct_answer, is_multitag, is_untagged = (
                question_lookup[question_key]
            )
            records.append(
                (
                    timestamp,
                    source_row,
                    question_id,
                    skill_id,
                    None if answer == "" else int(answer == correct_answer),
                    elapsed_time,
                    solving_id,
                    int(is_multitag),
                    int(is_untagged),
                    answer,
                    question_key,
                )
            )
    if not records:
        raise EdNetValidationError(f"student file has no interactions: {member}")

    records.sort(key=lambda row: (row[0], row[1]))
    raw_session_ids, _, raw_session_count = assign_sessions_and_splits(
        [row[0] for row in records]
    )
    answered_indices = [index for index, row in enumerate(records) if row[9] != ""]
    answered_records = [records[index] for index in answered_indices]
    values: dict[str, list[int]] = {
        "timestamp": [row[0] for row in answered_records],
        "source_row": [row[1] for row in answered_records],
        "question_id": [row[2] for row in answered_records],
        "skill_id": [row[3] for row in answered_records],
        "correct": [row[4] for row in answered_records],
        "elapsed_time": [row[5] for row in answered_records],
        "solving_id": [row[6] for row in answered_records],
        "raw_session_id": [int(raw_session_ids[index]) for index in answered_indices],
    }
    unanswered_indices = [index for index, row in enumerate(records) if row[9] == ""]
    unanswered: dict[str, list[object]] = {
        "original_student_id": [user_number] * len(unanswered_indices),
        "question_key": [records[index][10] for index in unanswered_indices],
        "question_id": [records[index][2] for index in unanswered_indices],
        "skill_id": [records[index][3] for index in unanswered_indices],
        "timestamp": [records[index][0] for index in unanswered_indices],
        "elapsed_time": [records[index][5] for index in unanswered_indices],
        "solving_id": [records[index][6] for index in unanswered_indices],
        "source_row": [records[index][1] for index in unanswered_indices],
        "raw_session_id": [int(raw_session_ids[index]) for index in unanswered_indices],
        "user_answer": [records[index][9] for index in unanswered_indices],
    }
    stats = {
        "raw_interactions": len(records),
        "raw_multitag_interactions": sum(row[7] for row in records),
        "raw_untagged_interactions": sum(row[8] for row in records),
        "unanswered_interactions": len(unanswered_indices),
        "supervised_interactions": len(answered_records),
        "supervised_multitag_interactions": sum(row[7] for row in answered_records),
        "supervised_untagged_interactions": sum(row[8] for row in answered_records),
        "raw_session_count": raw_session_count,
        "raw_timestamp_ties": sum(
            left[0] == right[0] for left, right in zip(records, records[1:])
        ),
        "raw_nonmonotonic_transitions": raw_nonmonotonic,
        "user_number": user_number,
    }
    return values, unanswered, stats


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(value: object, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_report(summary: dict[str, object]) -> str:
    scope = (
        "the genuine full official EdNet-KT1 archive"
        if summary["is_full_dataset"]
        else "a bounded smoke subset of the genuine official EdNet-KT1 archive"
    )
    return f"""# EdNet-KT1 preprocessing report

Generated from {scope}.

## Tag and composite-skill mapping

- Genuine original tags (excluding `-1`): {summary['genuine_original_tags']:,}
- Questions with missing tag metadata (`-1`): {summary['untagged_questions']:,}
- Questions whose repeated tag IDs were deduplicated: {summary['duplicate_tag_questions']:,}
- Multi-tag questions: {summary['multitag_questions']:,}
- Unique composite skill IDs (including `<UNTAGGED>`): {summary['composite_skills']:,}
- `<UNTAGGED>` composite skill ID: {UNTAGGED_SKILL_ID}

Every genuine tag set is numerically sorted and serialized with semicolons. One
stable contiguous composite ID represents the entire set. No interaction is
expanded and no primary tag is selected. The literal `-1` is missing metadata,
not a genuine tag; all such questions use the one `<UNTAGGED>` skill ID.

## Full source interactions

- Students/files: {summary['raw_students']:,}
- Interactions: {summary['raw_interactions']:,}
- Unanswered interactions excluded from supervision: {summary['unanswered_interactions']:,}
- Students with at least one unanswered interaction: {summary['unanswered_affected_students']:,}
- Full interaction exclusion percentage: {summary['unanswered_percentage']:.6f}%
- First-1,000-student sample interactions: {summary['sample_interactions']:,}
- First-1,000-student sample unanswered interactions: {summary['sample_unanswered_interactions']:,}
- Sample exclusion percentage: {summary['sample_unanswered_percentage']:.6f}%
- Interactions on multi-tag questions: {summary['raw_multitag_interactions']:,}
- Interactions on `<UNTAGGED>` questions: {summary['raw_untagged_interactions']:,}
- Equal-timestamp adjacent interactions after stable ordering: {summary['raw_timestamp_ties']:,}
- Nonmonotonic timestamp transitions in original file order: {summary['raw_nonmonotonic_transitions']:,}

Unanswered rows are not labeled incorrect, are not supervised targets, and are
not included in model history. Their exact source fields are preserved under
`audit/unanswered_events/`. Sessions and sequence counters are rebuilt only
after this filtering step.

## Filtering effect on sessions and sequences

- Supervised interactions after unanswered filtering: {summary['supervised_interactions']:,}
- Sequence length reduction: {summary['unanswered_interactions']:,} interactions
- Raw sessions before filtering: {summary['raw_sessions']:,}
- Supervised sessions after filtering: {summary['supervised_sessions']:,}
- Students whose session count changed: {summary['session_count_changed_students']:,}
- Students whose answered-event grouping changed: {summary['session_grouping_changed_students']:,}
- Net session-count change: {summary['session_count_change_net']:+,}

## Benchmark population

- Retained students (at least five sessions): {summary['retained_students']:,}
- Excluded students: {summary['excluded_students']:,}
- Retained interactions: {summary['retained_interactions']:,}
- Retained sessions: {summary['retained_sessions']:,}
- Retained multi-tag interactions: {summary['retained_multitag_interactions']:,}
- Retained `<UNTAGGED>` interactions: {summary['retained_untagged_interactions']:,}
- Training interactions: {summary['split_interactions']['train']:,}
- Validation interactions: {summary['split_interactions']['validation']:,}
- Test interactions: {summary['split_interactions']['test']:,}

Sessions start when the preceding interaction gap is at least 10 hours. Each
student's sessions are split chronologically with `floor(n/5)` validation and
test sessions and the remaining earlier sessions assigned to training. Event
rows retain full chronology so downstream rolling-history examples can use only
observations strictly before each target.

Detailed question, composite, user, and distribution tables are saved under
`mappings/` and `reports/`. Supervised event shards are under `events/`; removed
source rows and per-student session effects are under `audit/`.
"""


def preprocess_ednet(
    kt1_zip: Path,
    contents_zip: Path,
    output_dir: Path,
    *,
    rows_per_shard: int = 1_000_000,
    max_students: int | None = None,
) -> dict[str, object]:
    """Preprocess EdNet-KT1, returning the generated summary."""

    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a new path"
        )
    if max_students is not None and max_students <= 0:
        raise ValueError("max_students must be positive")

    output_dir.mkdir(parents=True)
    events_dir = output_dir / "events"
    mappings_dir = output_dir / "mappings"
    reports_dir = output_dir / "reports"
    audit_dir = output_dir / "audit"
    unanswered_dir = audit_dir / "unanswered_events"
    events_dir.mkdir()
    mappings_dir.mkdir()
    reports_dir.mkdir()
    audit_dir.mkdir()
    unanswered_dir.mkdir()
    try:
        questions = load_question_metadata(contents_zip)
        mappings = build_metadata_mappings(questions)
        _write_csv(mappings.original_tags, mappings_dir / "original_tags.csv")
        _write_csv(mappings.composites, mappings_dir / "composite_skills.csv")
        _write_csv(mappings.questions, mappings_dir / "questions.csv")

        question_lookup = {
            row.question_key: (
                int(row.question_id),
                int(row.skill_id),
                row.correct_answer,
                bool(row.is_multitag),
                bool(row.is_untagged),
            )
            for row in mappings.questions.itertuples(index=False)
        }
        composite_count = len(mappings.composites)
        raw_skill_interactions = np.zeros(composite_count + 1, dtype=np.int64)
        supervised_skill_interactions = np.zeros(composite_count + 1, dtype=np.int64)
        retained_skill_interactions = np.zeros(composite_count + 1, dtype=np.int64)
        summary: dict[str, object] = {
            "is_full_dataset": max_students is None,
            "genuine_original_tags": len(mappings.original_tags),
            "untagged_questions": int(mappings.questions["is_untagged"].sum()),
            "duplicate_tag_questions": int(
                mappings.questions["had_duplicate_tags"].sum()
            ),
            "multitag_questions": int(mappings.questions["is_multitag"].sum()),
            "composite_skills": composite_count,
            "raw_students": 0,
            "raw_interactions": 0,
            "unanswered_interactions": 0,
            "unanswered_affected_students": 0,
            "supervised_interactions": 0,
            "sample_interactions": 0,
            "sample_unanswered_interactions": 0,
            "sample_unanswered_affected_students": 0,
            "raw_multitag_interactions": 0,
            "raw_untagged_interactions": 0,
            "raw_timestamp_ties": 0,
            "raw_nonmonotonic_transitions": 0,
            "raw_sessions": 0,
            "supervised_sessions": 0,
            "session_count_changed_students": 0,
            "session_grouping_changed_students": 0,
            "session_count_change_net": 0,
            "retained_students": 0,
            "excluded_students": 0,
            "excluded_interactions": 0,
            "retained_interactions": 0,
            "retained_sessions": 0,
            "retained_multitag_interactions": 0,
            "retained_untagged_interactions": 0,
            "split_interactions": {"train": 0, "validation": 0, "test": 0},
            "split_sessions": {"train": 0, "validation": 0, "test": 0},
        }
        retained_users: list[dict[str, int]] = []
        session_impacts: list[dict[str, int]] = []
        writer = ShardWriter(events_dir, rows_per_shard)
        unanswered_writer = ShardWriter(
            unanswered_dir, rows_per_shard, schema=UNANSWERED_SCHEMA
        )

        with ZipFile(kt1_zip) as archive:
            members = sorted_student_members(archive)
            if max_students is not None:
                members = members[:max_students]
            for member_index, (user_number, member) in enumerate(members, start=1):
                values, unanswered, student_stats = _read_student(
                    archive, member, user_number, question_lookup
                )
                summary["raw_students"] += 1
                for name in (
                    "raw_interactions",
                    "unanswered_interactions",
                    "supervised_interactions",
                    "raw_multitag_interactions",
                    "raw_untagged_interactions",
                    "raw_timestamp_ties",
                    "raw_nonmonotonic_transitions",
                ):
                    summary[name] += student_stats[name]
                summary["raw_sessions"] += student_stats["raw_session_count"]
                if student_stats["unanswered_interactions"]:
                    summary["unanswered_affected_students"] += 1
                    unanswered_writer.append(unanswered)
                if member_index <= 1_000:
                    summary["sample_interactions"] += student_stats["raw_interactions"]
                    summary["sample_unanswered_interactions"] += student_stats[
                        "unanswered_interactions"
                    ]
                    if student_stats["unanswered_interactions"]:
                        summary["sample_unanswered_affected_students"] += 1
                np.add.at(raw_skill_interactions, values["skill_id"], 1)
                np.add.at(raw_skill_interactions, unanswered["skill_id"], 1)
                np.add.at(supervised_skill_interactions, values["skill_id"], 1)

                if values["timestamp"]:
                    session_ids, splits, session_count = assign_sessions_and_splits(
                        values["timestamp"]
                    )
                    raw_answered = values["raw_session_id"]
                    normalized_raw = []
                    previous = None
                    normalized_id = 0
                    for raw_session_id in raw_answered:
                        if raw_session_id != previous:
                            normalized_id += 1
                            previous = raw_session_id
                        normalized_raw.append(normalized_id)
                    grouping_changed = normalized_raw != session_ids.tolist()
                else:
                    session_ids = np.asarray([], dtype=np.int32)
                    splits = np.asarray([], dtype=np.uint8)
                    session_count = 0
                    grouping_changed = False
                summary["supervised_sessions"] += session_count
                count_changed = session_count != student_stats["raw_session_count"]
                if count_changed:
                    summary["session_count_changed_students"] += 1
                if grouping_changed:
                    summary["session_grouping_changed_students"] += 1
                summary["session_count_change_net"] += (
                    session_count - student_stats["raw_session_count"]
                )
                if student_stats["unanswered_interactions"]:
                    session_impacts.append(
                        {
                            "original_student_id": user_number,
                            "raw_interactions": student_stats["raw_interactions"],
                            "unanswered_interactions": student_stats[
                                "unanswered_interactions"
                            ],
                            "supervised_interactions": student_stats[
                                "supervised_interactions"
                            ],
                            "raw_session_count": student_stats["raw_session_count"],
                            "supervised_session_count": session_count,
                            "session_count_delta": session_count
                            - student_stats["raw_session_count"],
                            "answered_grouping_changed": int(grouping_changed),
                        }
                    )
                if session_count < 5:
                    summary["excluded_students"] += 1
                    summary["excluded_interactions"] += len(values["timestamp"])
                    continue

                student_id = int(summary["retained_students"]) + 1
                summary["retained_students"] = student_id
                summary["retained_interactions"] += len(values["timestamp"])
                summary["retained_sessions"] += session_count
                summary["retained_multitag_interactions"] += student_stats[
                    "supervised_multitag_interactions"
                ]
                summary["retained_untagged_interactions"] += student_stats[
                    "supervised_untagged_interactions"
                ]
                np.add.at(retained_skill_interactions, values["skill_id"], 1)

                holdout = session_count // 5
                split_session_counts = (session_count - 2 * holdout, holdout, holdout)
                for label, name in enumerate(("train", "validation", "test")):
                    summary["split_interactions"][name] += int(np.sum(splits == label))
                    summary["split_sessions"][name] += split_session_counts[label]

                question_attempts: Counter[int] = Counter()
                skill_attempts: Counter[int] = Counter()
                session_positions: Counter[int] = Counter()
                question_attempt_no = []
                skill_attempt_no = []
                session_position = []
                for question_id, skill_id, session_id in zip(
                    values["question_id"], values["skill_id"], session_ids
                ):
                    question_attempts[question_id] += 1
                    skill_attempts[skill_id] += 1
                    session_positions[int(session_id)] += 1
                    question_attempt_no.append(question_attempts[question_id])
                    skill_attempt_no.append(skill_attempts[skill_id])
                    session_position.append(session_positions[int(session_id)])

                row_count = len(values["timestamp"])
                event_values = {
                    "student_id": [student_id] * row_count,
                    "question_id": values["question_id"],
                    "skill_id": values["skill_id"],
                    "correct": values["correct"],
                    "timestamp": values["timestamp"],
                    "elapsed_time": values["elapsed_time"],
                    "solving_id": values["solving_id"],
                    "source_row": values["source_row"],
                    "session_id": session_ids.tolist(),
                    "session_position": session_position,
                    "question_attempt_no": question_attempt_no,
                    "skill_attempt_no": skill_attempt_no,
                    "split": splits.tolist(),
                }
                writer.append(event_values)
                retained_users.append(
                    {
                        "student_id": student_id,
                        "original_student_id": user_number,
                        "interaction_count": row_count,
                        "session_count": session_count,
                    }
                )
                if member_index % 10_000 == 0:
                    print(
                        f"processed {member_index:,}/{len(members):,} students; "
                        f"retained {student_id:,}; rows {writer.total_rows + len(writer.columns['student_id']):,}",
                        flush=True,
                    )
        writer.close()
        unanswered_writer.close()

        if writer.total_rows != summary["retained_interactions"]:
            raise RuntimeError(
                f"written row mismatch: {writer.total_rows} != "
                f"{summary['retained_interactions']}"
            )
        _write_csv(pd.DataFrame.from_records(retained_users), mappings_dir / "students.csv")
        _write_csv(pd.DataFrame.from_records(session_impacts), audit_dir / "session_impacts.csv")

        distribution = mappings.composites.copy()
        distribution["raw_interaction_count"] = raw_skill_interactions[1:]
        distribution["supervised_interaction_count"] = supervised_skill_interactions[1:]
        distribution["retained_interaction_count"] = retained_skill_interactions[1:]
        _write_csv(distribution, reports_dir / "composite_skill_distribution.csv")
        if unanswered_writer.total_rows != summary["unanswered_interactions"]:
            raise RuntimeError(
                f"unanswered audit row mismatch: {unanswered_writer.total_rows} != "
                f"{summary['unanswered_interactions']}"
            )
        summary["unanswered_percentage"] = (
            100.0 * summary["unanswered_interactions"] / summary["raw_interactions"]
        )
        summary["sample_unanswered_percentage"] = (
            100.0
            * summary["sample_unanswered_interactions"]
            / summary["sample_interactions"]
        )
        summary["event_shards"] = writer.shard_index
        summary["unanswered_audit_shards"] = unanswered_writer.shard_index
        summary["session_gap_ms"] = SESSION_GAP_MS
        summary["minimum_sessions"] = 5
        summary["split_label_mapping"] = {"0": "train", "1": "validation", "2": "test"}
        _write_json(summary, reports_dir / "summary.json")
        (reports_dir / "dataset_report.md").write_text(
            _render_report(summary), encoding="utf-8"
        )
        _write_json(
            {
                "complete": True,
                "event_schema": str(EVENT_SCHEMA),
                "event_shards": writer.shard_index,
                "unanswered_audit_schema": str(UNANSWERED_SCHEMA),
                "unanswered_audit_shards": unanswered_writer.shard_index,
                "retained_rows": writer.total_rows,
                "unanswered_audit_rows": unanswered_writer.total_rows,
                "rows_per_shard": rows_per_shard,
            },
            output_dir / "manifest.json",
        )
        (output_dir / "_SUCCESS").write_text("\n", encoding="utf-8")
        return summary
    except Exception:
        failure_path = output_dir / "_FAILED"
        failure_path.write_text(
            "Preprocessing stopped before completion. Inspect the raised error; "
            "this directory is not a valid dataset.\n",
            encoding="utf-8",
        )
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess genuine official EdNet-KT1 without row expansion"
    )
    parser.add_argument("--kt1-zip", type=Path, required=True)
    parser.add_argument("--contents-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    parser.add_argument(
        "--max-students",
        type=int,
        default=None,
        help="Smoke-test only; omit for the final full dataset",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = preprocess_ednet(
        args.kt1_zip,
        args.contents_zip,
        args.output_dir,
        rows_per_shard=args.rows_per_shard,
        max_students=args.max_students,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
