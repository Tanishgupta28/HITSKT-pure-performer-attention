#!/usr/bin/env python3
"""Verify the official development release and build the Algebra session store."""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ktbench.data.ednet import assign_sessions_and_splits
from ktbench.data.flat import sha256_file
from ktbench.data.session_store import SPLIT_NAMES, build_session_store

SOURCE_URL = "https://kdd.org/cupfiles/algebra_2005_2006.zip"
ARCHIVE_SHA256 = "a38d1fb55b139a38dfb019391417824855aef9104f51291d93fba614dacc9c84"
NAME = "algebra_2005_2006"
STUDENT = "Anon Student Id"
QUESTION = ["Problem Hierarchy", "Problem Name", "Step Name"]
LABEL = "Correct First Attempt"
KC = "KC(Default)"
TIMES = ["Step Start Time", "First Transaction Time", "Correct Transaction Time", "Step End Time"]
CLOCK = TIMES[1]
EXPECTED_ROWS = {"train": 809694, "master": 3967, "test": 3967, "submission": 3967}
SCHEMA = ["Row", STUDENT, *QUESTION[:2], "Problem View", QUESTION[2], *TIMES,
          "Step Duration (sec)", "Correct Step Duration (sec)", "Error Step Duration (sec)",
          LABEL, "Incorrects", "Hints", "Corrects", KC, "Opportunity(Default)"]
MINIMUM_SESSIONS = 5


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inspect_tsv(path, columns):
    """Validate the physical format before parsing values into a table."""
    missing = dict.fromkeys(columns, 0)
    with path.open(encoding="ascii", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        if next(reader) != columns:
            raise ValueError(f"unexpected TSV schema in {path}")
        count = 0
        for row in reader:
            if len(row) != len(columns):
                raise ValueError(f"wrong field count in {path}:{reader.line_num}")
            count += 1
            for name, value in zip(columns, row, strict=True):
                if value == "":
                    missing[name] += 1
        line_endings = stream.newlines
    return {"columns": columns, "records": count, "missing": missing, "encoding": "ASCII",
            "line_endings": line_endings,
            "delimiter": "tab", "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_release(archive, upload):
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("archive does not match the verified SIGKDD development release")
    frames, summaries = {}, {}
    with zipfile.ZipFile(archive) as zipped:
        expected = {f"{NAME}{'_' + part if part != 'submission' else ''}.txt": part
                    for part in EXPECTED_ROWS}
        if set(zipped.namelist()) != set(expected):
            raise ValueError("unexpected archive inventory")
        for member, part in expected.items():
            path = archive.parent / member
            if not path.exists():
                zipped.extract(member, archive.parent)
            # Compare bytes even when the original extracted files already exist.
            with zipped.open(member) as source, path.open("rb") as extracted:
                for block in iter(lambda: source.read(1 << 20), b""):
                    if block != extracted.read(len(block)):
                        raise ValueError(f"extracted copy differs from the archive: {member}")
                if extracted.read(1):
                    raise ValueError(f"extra extracted bytes: {member}")
            columns = ["Row", LABEL] if part == "submission" else SCHEMA
            info = inspect_tsv(path, columns)
            info.update(member=member, archive_timestamp=list(zipped.getinfo(member).date_time))
            if info["records"] != EXPECTED_ROWS[part]:
                raise ValueError(f"unexpected record count in {member}")
            table = pd.read_csv(path, sep="\t", encoding="ascii", dtype=str,
                                keep_default_na=False, quoting=csv.QUOTE_NONE)
            table["Row"] = pd.to_numeric(table["Row"]).astype("int64")
            if table["Row"].duplicated().any() or table["Row"].le(0).any():
                raise ValueError(f"invalid Row IDs in {member}")
            frames[part], summaries[part] = table, info
    train, master, test, submission = (frames[name] for name in EXPECTED_ROWS)
    if not master["Row"].equals(test["Row"]) or not test["Row"].equals(submission["Row"]):
        raise ValueError("master, masked test and submission Row IDs differ")
    identity = ["Row", STUDENT, *QUESTION, "Problem View", KC, "Opportunity(Default)"]
    if not master[identity].equals(test[identity]) or not submission[LABEL].eq("").all():
        raise ValueError("the official masked test and master identities do not agree")
    data = pd.concat((train, master), ignore_index=True, verify_integrity=True)
    if data["Row"].duplicated().any() or data[STUDENT].nunique() != 575:
        raise ValueError("labeled union does not match the expected official release")
    if upload is not None:
        summaries["upload"] = inspect_tsv(upload, ["Row", LABEL])
        if summaries["upload"]["sha256"] != summaries["submission"]["sha256"]:
            raise ValueError("the supplied upload does not match the blank submission member")
    return data, summaries


def measure_timestamps(data):
    audit = {}
    for field in TIMES:
        raw = data[field]
        parsed = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        audit[field] = {
            "missing": int(raw.eq("").sum()),
            "invalid": int((raw.ne("") & parsed.isna()).sum()),
            "source_row_reversals": int(parsed.groupby(data[STUDENT]).diff().lt(pd.Timedelta(0)).sum()),
            "minimum": str(parsed.min()), "maximum": str(parsed.max()),
        }
        data[field] = parsed
    order = {f"{left} > {right}": int(data[left].gt(data[right]).sum())
             for left, right in combinations(TIMES, 2)}
    return {"fields": audit, "ordering_violations": order,
            "format": "YYYY-MM-DD HH:MM:SS.f; offset unspecified in the source"}


def categories(values):
    vocabulary = sorted(set(values))
    mapping = {value: index for index, value in enumerate(vocabulary, 1)}
    return np.fromiter((mapping[value] for value in values), dtype=np.int32), vocabulary


def preprocess(archive, root, upload=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "_SUCCESS").unlink(missing_ok=True)
    data, files = load_release(archive, upload)
    data.sort_values([STUDENT, "Row"], inplace=True)
    timestamp_audit = measure_timestamps(data)
    valid_label = data[LABEL].isin(("0", "1"))
    valid_identity = data[[STUDENT, *QUESTION]].ne("").all(axis=1)
    valid_clock = data[CLOCK].notna()
    valid = valid_label & valid_identity & valid_clock
    full_questions = data[QUESTION].drop_duplicates()
    shared_pairs = full_questions.duplicated(QUESTION[1:], keep=False)
    raw = {
        "students": int(data[STUDENT].nunique()), "interactions": len(data),
        "problem_names": int(data["Problem Name"].nunique()),
        "problem_step_pairs": len(data[QUESTION[1:]].drop_duplicates()),
        "hierarchy_problem_steps": len(data[QUESTION].drop_duplicates()),
        "problem_step_pairs_in_multiple_hierarchies":
            len(full_questions.loc[shared_pairs, QUESTION[1:]].drop_duplicates()),
        "repeated_student_hierarchy_problem_steps": int(data.duplicated([STUDENT, *QUESTION]).sum()),
        "correct": int(data[LABEL].eq("1").sum()),
        "invalid_or_missing_label": int((~valid_label).sum()),
        "missing_identity": int((~valid_identity).sum()),
        "invalid_or_missing_first_transaction": int((~valid_clock).sum()),
        "union_of_invalid_required_fields": int((~valid).sum()),
    }
    data = data.loc[valid].sort_values([STUDENT, CLOCK, "Row"]).copy()
    data["timestamp"] = data[CLOCK].to_numpy(dtype="datetime64[ms]").astype(np.int64)
    groups = []
    for _, student in data.groupby(STUDENT, sort=False):
        ids, splits, count = assign_sessions_and_splits(student["timestamp"].tolist())
        groups.append((student, ids, splits, count))
    retained = [(student, ids, splits) for student, ids, splits, count in groups
                if count >= MINIMUM_SESSIONS]
    events = pd.concat([student.assign(session_id=ids, split=splits)
                        for student, ids, splits in retained], ignore_index=True)
    tags = [tuple(sorted({tag.strip() for tag in value.split("~~") if tag.strip()}))
            for value in events[KC]]
    events["student_id"], students = categories(events[STUDENT].tolist())
    events["question_id"], questions = categories(list(events[QUESTION].itertuples(index=False, name=None)))
    events["skill_id"], skills = categories(tags)
    events["correct"] = events[LABEL].astype("uint8")
    grouped = events.groupby(["student_id", "session_id"], sort=False)
    events["session_position"] = grouped.cumcount().astype("int32") + 1
    ties = events.duplicated(["student_id", "timestamp"])
    summary = {
        "dataset": "algebra2005",
        "protocol": "algebra2005_first_transaction_10h_sessions_v1",
        "source_url": SOURCE_URL, "archive_sha256": ARCHIVE_SHA256,
        "source_version": "KDD Cup 2010 Algebra I 2005–2006 development archive",
        "files": files, "raw": raw, "timestamps": timestamp_audit,
        "timestamp_field": CLOCK, "processed_timestamp_unit": "milliseconds",
        "minimum_sessions": MINIMUM_SESSIONS, "session_gap_ms": 36_000_000,
        "source_sessions": sum(count for _, _, _, count in groups),
        "retained_students": len(students), "retained_interactions": len(events),
        "retained_sessions": grouped.ngroups, "maximum_session_length": int(grouped.size().max()),
        "questions": len(questions), "skills": len(skills),
        "atomic_skills": len({tag for signature in skills for tag in signature}),
        "untagged_interactions": tags.count(()),
        "multi_skill_interactions": sum(len(signature) > 1 for signature in tags),
        "retained_repeated_question_events": int(events.duplicated(["student_id", "question_id"]).sum()),
        "same_clock_additional_events": int(ties.sum()),
        "students_excluded_by_session_eligibility": len(groups) - len(students),
        "events_excluded_by_session_eligibility": len(data) - len(events),
        "split_interactions": {name: int(events["split"].eq(number).sum())
                               for name, number in SPLIT_NAMES.items()},
        "split_sessions": {name: int((events["split"].eq(number) &
                                          events["session_position"].eq(1)).sum())
                           for name, number in SPLIT_NAMES.items()},
        "split_correct": {name: int(events.loc[events["split"].eq(number), "correct"].sum())
                          for name, number in SPLIT_NAMES.items()},
    }
    (root / "reports").mkdir(exist_ok=True)
    (root / "events").mkdir(exist_ok=True)
    (root / "mappings").mkdir(exist_ok=True)
    for name, vocabulary in (("students", students), ("questions", questions), ("skills", skills)):
        write_json(root / "mappings" / f"{name}.json", vocabulary)
    columns = ["Row", "student_id", "question_id", "skill_id", "correct",
               "timestamp", "session_id", "session_position", "split"]
    pq.write_table(pa.Table.from_pandas(events[columns], preserve_index=False),
                   root / "events" / "part-00000.parquet")
    write_json(root / "reports" / "summary.json", summary)
    (root / "_SUCCESS").write_text("complete\n")
    try:
        store = root / "session_store"
        build_session_store(root, store)
        metadata_path = store / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(strict_timestamp_history=True, protocol=summary["protocol"],
                        source_archive_sha256=ARCHIVE_SHA256,
                        events_sha256=sha256_file(root / "events" / "part-00000.parquet"))
        write_json(metadata_path, metadata)
    except Exception:
        (root / "_SUCCESS").unlink()
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upload", type=Path)
    args = parser.parse_args()
    print(json.dumps(preprocess(args.archive, args.output, args.upload), indent=2))


if __name__ == "__main__":
    main()
