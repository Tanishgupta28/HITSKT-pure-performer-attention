"""Recover only Student 1's approved prefix from the published Junyi source.

Run with NumPy and pandas installed. Reproduce and hash the complete array
layout in memory, then write only the private pre-test slice after every
array and the student crosswalk match the independently published digests.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import JUNYI_ASSETS, ROOT
from app.junyi_bank import digest, load_mappings
from app.junyi_model import GAP, STORE_SHA256, read_baseline


_COLUMNS = {"studentId": "int32", "problemId": "int32", "startTime": "int64", "correct": "uint8"}
_REPORTS = ROOT / "reports" / "independent_correctness_audit"
_EXPECTED_ROWS = 14_660_217
_RAW_SHA256 = "8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a"
_CHUNK_SIZE = 262_144


def _scan(path, source_questions, question_ids):
    columns = {name: np.empty(_EXPECTED_ROWS, dtype=dtype) for name, dtype in _COLUMNS.items()}
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if hashlib.file_digest(stream, "sha256").hexdigest() != _RAW_SHA256:
            raise ValueError("Raw source does not match the accepted Junyi digest")
        stream.seek(0)
        offset = 0
        chunks = pd.read_csv(stream, usecols=list(_COLUMNS), dtype=_COLUMNS, chunksize=_CHUNK_SIZE)
        for chunk in chunks:
            stop = offset + len(chunk)
            if stop > _EXPECTED_ROWS:
                raise ValueError("Raw source exceeds the tracked interaction count")
            for name, destination in columns.items():
                values = chunk[name].to_numpy()
                if name == "problemId":
                    indices = np.searchsorted(source_questions, values)
                    if np.any(indices == len(source_questions)) or not np.array_equal(source_questions[indices], values):
                        raise ValueError("Raw question is missing from the published mapping")
                    values = question_ids[indices]
                elif name == "correct" and np.any(values > 1):
                    raise ValueError("Raw correctness is not binary")
                destination[offset:stop] = values
            offset = stop
        after = os.fstat(stream.fileno())
        if offset != _EXPECTED_ROWS or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Raw source changed during reconstruction")
    return columns


def _layout(students, seconds):
    # lexsort is stable, so ties in source student and time keep original CSV row order.
    order = np.lexsort((seconds, students))
    boundaries = np.empty(len(order), dtype=bool)
    boundaries[0] = True
    for start in range(1, len(order), _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, len(order))
        prior, following = order[start - 1:stop - 1], order[start:stop]
        boundaries[start:stop] = (students[following] != students[prior]) | (seconds[following] - seconds[prior] >= GAP // 1_000)
    starts = np.flatnonzero(boundaries).astype("<i8", copy=False)
    source_students = students[order[starts]]
    student_starts = np.flatnonzero(np.r_[True, source_students[1:] != source_students[:-1]]).astype("<i8", copy=False)
    offsets = np.append(student_starts, len(starts)).astype("<i8", copy=False)
    counts = np.diff(offsets)
    if np.any(counts < 5):
        raise ValueError("Accepted Junyi should retain every source student")
    ids = np.arange(1, len(counts) + 1, dtype="<i4")
    numbers = (np.arange(len(starts)) - np.repeat(student_starts, counts) + 1).astype("<i4")
    holdout = counts // 5
    splits = (numbers > np.repeat(counts - 2 * holdout, counts)).astype("u1")
    splits[numbers > np.repeat(counts - holdout, counts)] = 2
    return order, starts, offsets, ids, source_students[student_starts], np.repeat(ids, counts), numbers, splits


def _npy_digest(dtype, count, chunks):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": np.dtype(dtype).str, "shape": (count,), "fortran_order": False})
    verifier = hashlib.sha256(header.getvalue())
    seen = 0
    for values in chunks:
        array = np.ascontiguousarray(values, dtype=dtype)
        verifier.update(memoryview(array))
        seen += len(array)
    if seen != count:
        raise ValueError("Rebuilt NumPy stream has the wrong length")
    return verifier.hexdigest()


def _store_digests(columns, layout, skills_by_id, mappings):
    order, starts, student_offsets, _, _, session_students, numbers, splits = layout
    size = len(order)
    session_offsets = np.append(starts, size).astype("<i8", copy=False)

    def ordered(name, transform=lambda values: values):
        for start in range(0, size, _CHUNK_SIZE):
            yield transform(columns[name][order[start:start + _CHUNK_SIZE]])

    metadata = {"format_version": 2, "interactions": size, "sessions": len(starts), "students": len(student_offsets) - 1,
                "num_questions": len(mappings.pairs), "num_skills": len(mappings.composite_tags),
                "maximum_session_length": int(np.diff(session_offsets).max()),
                "split_sessions": {name: int(np.count_nonzero(splits == i)) for i, name in enumerate(("train", "validation", "test"))},
                "processed_timestamp_unit": "milliseconds", "timestamp_hours_divisor": 3_600_000,
                "action_truncation": False, "action_chunking": False}
    arrays = {"session_number": ("<i4", numbers), "session_offsets": ("<i8", session_offsets), "session_split": ("u1", splits),
              "session_student": ("<i4", session_students), "student_offsets": ("<i8", student_offsets)}
    hashes = {f"{name}.npy": _npy_digest(dtype, len(values), (values,)) for name, (dtype, values) in arrays.items()}
    hashes.update({"question.npy": _npy_digest("<i4", size, ordered("problemId")),
                   "skill.npy": _npy_digest("<i4", size, ordered("problemId", lambda ids: skills_by_id[ids])),
                   "correct.npy": _npy_digest("u1", size, ordered("correct")),
                   "timestamp_ms.npy": _npy_digest("<i8", size, ordered("startTime", lambda times: times * 1_000)),
                   "metadata.json": digest((json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()),
                   "_SUCCESS": digest(b"complete\n")})
    return hashes


def recover(raw, output, ranking):
    mappings = load_mappings(ROOT / "reports" / "datasets" / "junyi", ranking)
    question_order = sorted((int(source), qid) for qid, source in mappings.source_ids.items())
    source_questions, question_ids = np.asarray(question_order, dtype="<i4").T
    skills_by_id = np.zeros(len(question_ids) + 1, dtype="<i4")
    for question_id, item in mappings.pairs.items():
        skills_by_id[question_id] = item.skill_id

    columns = _scan(raw, source_questions, question_ids)
    layout = _layout(columns["studentId"], columns["startTime"])
    order, starts, student_offsets, ids, raw_students, _, _, splits = layout
    expected = json.loads((_REPORTS / "junyi_evaluation_manifest.json").read_text(encoding="utf-8"))["session_store_sha256"]
    observed = _store_digests(columns, layout, skills_by_id, mappings)
    if observed != expected:
        mismatches = sorted(name for name in set(expected) | set(observed) if observed.get(name) != expected.get(name))
        raise ValueError(f"Reconstructed SessionStore differs from the original: {', '.join(mismatches)}")

    source_manifest = json.loads((_REPORTS / "junyi_data_manifest.json").read_text(encoding="utf-8"))
    crosswalk = next(entry for entry in source_manifest["processed_file_checksums"] if entry["path"].endswith("/students.csv"))["sha256"]
    crosswalk_text = io.StringIO(newline="")
    writer = csv.writer(crosswalk_text, lineterminator="\n")
    writer.writerow(("student_id", "source_student_id"))
    writer.writerows(zip(ids, raw_students, strict=True))
    if digest(crosswalk_text.getvalue().encode()) != crosswalk or int(raw_students[0]) != 1 or int(ids[0]) != 1:
        raise ValueError("Rebuilt student crosswalk does not match the original Student 1 mapping")

    begin, end = map(int, student_offsets[:2])
    excluded = begin + int(np.flatnonzero(splits[begin:end] == 2)[0])
    cutoff = int(columns["startTime"][order[starts[excluded]]]) * 1_000
    sessions = []
    for session in range(begin, excluded):
        rows = order[starts[session]:starts[session + 1]]
        questions = columns["problemId"][rows]
        sessions.append({"question_ids": questions.tolist(), "skill_ids": skills_by_id[questions].tolist(),
                         "correct": columns["correct"][rows].tolist(), "timestamp_ms": (columns["startTime"][rows] * 1_000).tolist()})
    baseline = {"source_student_id": 1, "processed_student_id": 1, "first_test_timestamp_ms": cutoff,
                "source_session_store_sha256": STORE_SHA256, "sessions": sessions,
                "provenance": {"provider": "published_junyi_raw_reconstruction", "raw_sha256": _RAW_SHA256,
                               "student_crosswalk_sha256": crosswalk, "verified_store_members": observed}}
    data = (json.dumps(baseline, indent=2, sort_keys=True) + "\n").encode()
    version = digest(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(f"{output.name}.candidate")
    with staged.open("xb") as stream:
        stream.write(data)
    try:
        verified = read_baseline(staged, version, mappings)
        if output.exists() and output.read_bytes() != data:
            raise FileExistsError("An existing history file differs from this reconstruction")
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    return {"verified_store_members": len(observed), "student_crosswalk_verified": True, "sessions": len(verified.sessions),
            "interactions": sum(map(len, verified.sessions)), "history_sha256": version}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_junyi_csv", type=Path)
    parser.add_argument("--ranking", type=Path, default=JUNYI_ASSETS / "representative_questions.csv")
    args = parser.parse_args()
    print(json.dumps(recover(args.raw_junyi_csv, JUNYI_ASSETS / "student1_pre_test_history.json", args.ranking)))


if __name__ == "__main__":
    main()
