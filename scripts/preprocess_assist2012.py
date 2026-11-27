#!/usr/bin/env python3
"""Build the full official ASSIST2012 first-action benchmark."""
from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ktbench.data.ednet import SESSION_GAP_MS
from ktbench.data.flat import sha256_file
from ktbench.data.session_store import SPLIT_NAMES, build_session_store

ARCHIVE_SHA256 = "7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445"
MEMBER = "2012-2013-data-with-predictions-4-final.csv"
SOURCE = "https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect"
DOWNLOAD = "https://drive.google.com/file/d/1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY/view"
PROTOCOL = "assist2012_first_action_10h_sessions_v1"
START = r"(?m)^- - start\r?\n +-[ ]*(\d{13})\s*$"
ACTION = r"(?m)^- - (answer|hint|scaffold)\r?\n +-[ ]*(-?\d+)\r?\n(?: +-[ ]*(true|false)\s*$)?"
CHUNK_ROWS = 200_000
REQUIRED = ("problem_log_id", "user_id", "problem_id", "skill_id", "correct",
            "actions", "start_time", "end_time", "ms_first_response", "problem_type")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_source(archive):
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("archive does not match the official ASSIST2012 release")
    raw = archive.parent / MEMBER
    with zipfile.ZipFile(archive) as zipped:
        if zipped.namelist() != [MEMBER]:
            raise ValueError("unexpected publisher ZIP contents")
        if not raw.exists():
            with zipped.open(MEMBER) as source, raw.open("xb") as extracted:
                shutil.copyfileobj(source, extracted)
        sha = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")()
        utf8 = True
        non_ascii = False
        with zipped.open(MEMBER) as source, raw.open("rb") as extracted:
            for block in iter(lambda: source.read(1 << 22), b""):
                if extracted.read(len(block)) != block:
                    raise ValueError("extracted CSV differs from publisher ZIP member")
                sha.update(block)
                non_ascii |= not block.isascii()
                if utf8:
                    try:
                        decoder.decode(block)
                    except UnicodeDecodeError:
                        utf8 = False
            if extracted.read(1):
                raise ValueError("unexpected bytes after the archive member")
        if utf8:
            decoder.decode(b"", final=True)
        member = zipped.getinfo(MEMBER)
    encoding = "utf-8" if utf8 else "latin-1"
    with raw.open(encoding=encoding, newline="") as stream:
        header = next(csv.reader(stream))
    if len(header) != len(set(header)) or not set(REQUIRED).issubset(header):
        raise ValueError("unexpected ASSIST2012 schema")
    return raw, encoding, {"archive_sha256": ARCHIVE_SHA256, "csv_sha256": sha.hexdigest(),
                           "archive_member_timestamp": list(member.date_time), "csv_bytes": member.file_size,
                           "archive_crc32": f"{member.CRC:08x}", "encoding_used": encoding,
                           "strict_utf8_valid": utf8, "ascii_only": not non_ascii,
                           "delimiter": ",", "columns": header}


def int_id(values, *, allow_missing=False):
    valid = values.str.fullmatch(r"[0-9]+") & values.ne("0")
    if not allow_missing and not valid.all():
        raise ValueError(f"invalid required identity in {values.name}")
    return pd.to_numeric(values.where(valid, "0")).astype(np.int64)


def load(raw, encoding, header):
    frames, audit, missing, types = [], Counter(), Counter(), Counter()
    minimum, maximum = None, None
    for chunk in pd.read_csv(raw, dtype=str, chunksize=CHUNK_ROWS, encoding=encoding,
                             keep_default_na=False, on_bad_lines="error"):
        if list(chunk.columns) != header or isinstance(chunk.index, pd.MultiIndex):
            raise ValueError("CSV contains malformed record width")
        missing.update({field: int(chunk[field].eq("").sum()) for field in header})
        audit["source_rows"] += len(chunk)
        nums = {name: int_id(chunk[name]) for name in REQUIRED[:3]}
        score = pd.to_numeric(chunk["correct"], errors="coerce")
        valid_score = score.between(0, 1)
        correct = score.eq(1)
        start = pd.to_datetime(chunk["start_time"], format="ISO8601", errors="coerce")
        end = pd.to_datetime(chunk["end_time"], format="ISO8601", errors="coerce")
        recorded_start = pd.to_numeric(chunk["actions"].str.extract(START, expand=False), errors="coerce")
        action = chunk["actions"].str.extract(ACTION)
        delay_ms = pd.to_numeric(action[1], errors="coerce")
        wall_start = pd.to_datetime(recorded_start, unit="ms", utc=True).dt.tz_convert(
            "America/New_York").dt.tz_localize(None)
        start_delta = start - wall_start
        start_error = start_delta.abs()
        # Relative action offsets follow the exported start clock. Anchor
        # that clock to UTC with the action log's explicit Unix start.
        wall_action = start + pd.to_timedelta(delay_ms, unit="ms")
        timestamp_ms = recorded_start + start_delta.dt.total_seconds() * 1_000 + delay_ms
        plausible = (start_error <= pd.Timedelta(seconds=1)) & delay_ms.ge(0)
        # Source end times are recorded on the same local clock. The offset
        # and recorded start alone define the model's event timestamp.
        in_bounds = wall_action.le(end + pd.Timedelta(seconds=1))
        observed = action[0].isin(("hint", "scaffold")) | (action[0].eq("answer") & action[2].isin(("true", "false")))
        first_correct = action[0].eq("answer") & action[2].eq("true")
        eligible = valid_score & plausible & in_bounds & observed & first_correct.eq(correct)
        skills = int_id(chunk["skill_id"], allow_missing=True)
        if (skills.eq(0) & chunk["skill_id"].ne("")).any():
            raise ValueError("unexpected nonnumeric skill annotation")
        for name, mask in {
            "invalid_score": ~valid_score, "fractional_score": valid_score & score.ne(0) & ~correct,
            "invalid_start_time": start.isna(), "invalid_end_time": end.isna(),
            "end_before_start_time": end.lt(start), "missing_logged_start": recorded_start.isna(),
            "missing_logged_first_action": delay_ms.isna(), "negative_first_action_delay": delay_ms.lt(0),
            "start_clock_mismatch": start_error.gt(pd.Timedelta(seconds=1)),
            "first_action_after_end": wall_action.notna() & end.notna() & ~in_bounds,
            "first_action_delay_differs_from_ms_first_response": delay_ms.notna() &
                delay_ms.ne(pd.to_numeric(chunk["ms_first_response"], errors="coerce")),
            "first_action_outcome_differs_from_binary_label": observed & valid_score & ~first_correct.eq(correct),
            "valid_timed_response": eligible,
        }.items():
            audit[name] += int(mask.sum())
        types.update(chunk["problem_type"].value_counts().to_dict())
        if wall_action[eligible].notna().any():
            earliest, latest = wall_action[eligible].min(), wall_action[eligible].max()
            minimum = earliest if minimum is None else min(minimum, earliest)
            maximum = latest if maximum is None else max(maximum, latest)
        frame = pd.DataFrame({**nums, "raw_skill": skills, "correct": correct.astype(np.uint8),
                              "timestamp": timestamp_ms.fillna(0).round().astype(np.int64),
                              "valid": eligible})
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    unique_events = data["problem_log_id"].nunique()
    if len(data) != 6_123_270 or unique_events != len(data):
        raise ValueError("unexpected source cardinality or duplicate response keys")
    audit.update({"unique_response_ids": unique_events, "students": data["user_id"].nunique(),
                  "questions": data["problem_id"].nunique(),
                  "atomic_skills": data.loc[data["raw_skill"].gt(0), "raw_skill"].nunique(),
                  "multiple_skill_event_keys": 0})
    return data, {**audit, "missing": dict(missing), "problem_types": dict(types),
                  "first_action_local_min": str(minimum), "first_action_local_max": str(maximum)}


def distribution(values):
    return {"count": len(values), "mean": float(np.mean(values)),
            "percentiles": {str(p): float(v) for p, v in zip(
                [0, 25, 50, 75, 90, 95, 99, 100], np.percentile(values, [0, 25, 50, 75, 90, 95, 99, 100]),
                strict=True)}}


def preprocess(archive, root):
    raw, encoding, provenance = verify_source(archive)
    data, raw_stats = load(raw, encoding, provenance["columns"])
    valid = data.loc[data["valid"]].sort_values(["user_id", "timestamp", "problem_log_id"]).copy()
    new_student = valid["user_id"].ne(valid["user_id"].shift())
    gap = valid["timestamp"].diff()
    session_start = new_student | gap.ge(SESSION_GAP_MS)
    valid["session_id"] = session_start.groupby(valid["user_id"], sort=False).cumsum().astype(np.int32)
    count = valid.groupby("user_id", sort=False)["session_id"].max()
    keep = valid["user_id"].isin(count.index[count.ge(5)])
    events = valid.loc[keep].copy()
    per_student_sessions = events["user_id"].map(count).to_numpy()
    holdout = per_student_sessions // 5
    sessions = events["session_id"].to_numpy()
    events["split"] = ((sessions > per_student_sessions - 2 * holdout).astype(np.uint8)
                       + (sessions > per_student_sessions - holdout).astype(np.uint8))
    mapping = {}
    for source, target in (("user_id", "student_id"), ("problem_id", "question_id"), ("raw_skill", "skill_id")):
        catalog = np.sort(events[source].unique())
        events[target] = 1 + np.searchsorted(catalog, events[source]).astype(np.int32)
        mapping[target] = catalog.tolist()
    grouped = events.groupby(["student_id", "session_id"], sort=False)
    events["session_position"] = grouped.cumcount().astype(np.int32) + 1
    same_time = events.duplicated(["student_id", "timestamp"]).sum()
    summary = {
        "dataset": "assist2012", "protocol": PROTOCOL, "source": SOURCE, "download": DOWNLOAD,
        "raw": raw_stats, "provenance": provenance,
        "timestamp_field": "exported problem start anchored by logged Unix time plus first graded action offset",
        "minimum_sessions": 5, "session_gap_ms": SESSION_GAP_MS,
        "timestamp_eligible_interactions": len(valid), "timestamp_eligible_students": len(count),
        "source_sessions": int(count.sum()), "students_at_least_five_sessions": int(count.ge(5).sum()),
        "retained_students": len(mapping["student_id"]), "retained_interactions": len(events),
        "retained_sessions": grouped.ngroups, "questions": len(mapping["question_id"]),
        "skills": len(mapping["skill_id"]), "maximum_session_length": int(grouped.size().max()),
        "untagged_interactions": int(events["raw_skill"].eq(0).sum()),
        "same_clock_additional_events": int(same_time),
        "repeated_student_question_events": int(events.duplicated(["student_id", "question_id"]).sum()),
        "retained_fraction": len(events) / len(data), "excluded_fraction": 1 - len(events) / len(data),
        "excluded_for_session_eligibility": len(valid) - len(events),
        "distributions": {
            "all_sessions_per_student": distribution(count.to_numpy()),
            "sessions_per_retained_student": distribution(count[count.ge(5)].to_numpy()),
            "retained_interactions_per_session": distribution(grouped.size().to_numpy()),
            "retained_interaction_history_length_per_student": distribution(events["student_id"].value_counts().to_numpy()),
            "inter_session_gap_hours": distribution(gap.loc[session_start & ~new_student & keep].to_numpy() / 3_600_000),
            "questions_frequency": distribution(events["question_id"].value_counts().to_numpy()),
            "skills_frequency": distribution(events["skill_id"].value_counts().to_numpy()),
        },
        "split_interactions": {}, "split_correct": {}, "split_sessions": {},
    }
    for name, number in SPLIT_NAMES.items():
        split = events.loc[events["split"].eq(number)]
        summary["split_interactions"][name] = len(split)
        summary["split_correct"][name] = int(split["correct"].sum())
        summary["split_sessions"][name] = int(split["session_position"].eq(1).sum())
    for folder in ("reports", "events", "mappings"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "_SUCCESS").unlink(missing_ok=True)
    write_json(root / "mappings" / "catalogs.json", mapping)
    write_json(root / "reports" / "summary.json", summary)
    columns = ["problem_log_id", "student_id", "question_id", "skill_id", "correct",
               "timestamp", "session_id", "session_position", "split"]
    parquet = root / "events" / "part-00000.parquet"
    pq.write_table(pa.Table.from_pandas(events[columns], preserve_index=False), parquet)
    (root / "_SUCCESS").write_text("complete\n")
    try:
        store = root / "session_store"
        build_session_store(root, store)
        metadata_path = store / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(strict_timestamp_history=True, protocol=PROTOCOL,
                        source_archive_sha256=ARCHIVE_SHA256, events_sha256=sha256_file(parquet))
        write_json(metadata_path, metadata)
    except Exception:
        (root / "_SUCCESS").unlink()
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(preprocess(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
