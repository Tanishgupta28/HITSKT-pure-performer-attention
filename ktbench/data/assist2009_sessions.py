"""ASSIST2009 assignment-run hierarchy on the existing interaction split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ktbench.batching import ExampleShape, Session, SessionExample
from ktbench.data.assist2009 import Assist2009Store, SPLITS
from ktbench.data.flat import sha256_file
from ktbench.data.session_store import RollingSessionDataset


SESSION_PROTOCOL = "contiguous_assignment_runs_v1"
FIELDS = ("assignment_id", "sequence_id", "base_sequence_id", "assistment_id",
          "problem_id", "student_class_id", "position")


def starts_for(students, keys):
    changed = np.any(keys[1:] != keys[:-1], axis=1)
    return np.r_[0, np.flatnonzero(changed | (students[1:] != students[:-1])) + 1, len(students)]


def describe(lengths, counts, student_lengths):
    eligible = counts >= 5
    return {
        "sessions": len(lengths),
        "mean_length": float(lengths.mean()),
        "median_length": float(np.median(lengths)),
        "length_percentiles": {str(p): float(np.percentile(lengths, p)) for p in (0, 25, 50, 75, 90, 95, 99, 100)},
        "length_distribution": {f"{a}-{b}": int(((lengths >= a) & (lengths <= b)).sum())
                                for a, b in ((1, 1), (2, 3), (4, 5), (6, 10), (11, 20),
                                             (21, 50), (51, 100), (101, int(max(101, lengths.max()))))},
        "mean_sessions_per_student": float(counts.mean()),
        "students_at_least_five": int(eligible.sum()),
        "interactions_if_five_session_filter": int(student_lengths[eligible].sum()),
        "interactions_excluded_by_five_session_filter": int(student_lengths[~eligible].sum()),
    }


def prepare(source: Path, ordered: Assist2009Store, output: Path) -> dict:
    if sha256_file(source) != ordered.metadata["source_sha256"]:
        raise ValueError("source does not match ordered ASSIST2009 benchmark")
    raw = pd.read_csv(source, usecols=["order_id", "user_id", *FIELDS], encoding="latin1")
    counts = raw.groupby("order_id", sort=False)[list(FIELDS)].nunique(dropna=False)
    conflicts = {k: int(v) for k, v in counts.gt(1).sum().items()}
    if conflicts["assignment_id"]:
        raise ValueError("conflicting assignment ids for the same event")
    records = raw.drop_duplicates("order_id").set_index("order_id").loc[ordered.order_id]
    students = np.asarray(ordered.student)
    student_lengths = np.diff(ordered.student_offsets)
    keys = {
        "assignment": ["assignment_id"], "sequence": ["sequence_id"],
        "base_sequence": ["base_sequence_id"], "assignment_and_sequence": ["assignment_id", "sequence_id"],
        "class": ["student_class_id"], "assignment_position": ["position"],
        "assistment": ["assistment_id"], "problem": ["problem_id"],
    }
    report = {
        "selected_protocol": SESSION_PROTOCOL, "dataset_provenance": ordered.provenance,
        "duplicate_event_field_conflicts": conflicts,
        "missing_field_counts": records[list(FIELDS)].isna().sum().astype(int).to_dict(),
        "distinct_source_field_values": records[list(FIELDS)].nunique().astype(int).to_dict(),
        "sequence_ids_used_by_multiple_assignments": int(records.groupby("sequence_id").assignment_id.nunique().gt(1).sum()),
        "assignment_ids_with_multiple_sequences": int(records.groupby("assignment_id").sequence_id.nunique().gt(1).sum()),
        "alternatives": {},
    }
    for name, columns in keys.items():
        offsets = starts_for(students, records[columns].to_numpy())
        lengths = np.diff(offsets)
        run_keys = records.iloc[offsets[:-1]][["user_id", *columns]]
        repeated = int(run_keys.duplicated().sum())
        run_count = np.bincount(students[offsets[:-1]], minlength=len(student_lengths) + 1)[1:]
        grouped = records.groupby(["user_id", *columns], dropna=False).size()
        group_count = grouped.groupby(level=0).size().to_numpy()
        report["alternatives"][name] = {
            "contiguous_runs": {
                **describe(lengths, run_count, student_lengths),
                "reopened_runs": repeated,
                "cross_split_runs": int(np.count_nonzero(ordered.split[offsets[:-1]] != ordered.split[offsets[1:] - 1])),
                "retained_without_minimum": len(students),
                "excluded_without_minimum": 0,
            },
            "global_group_by_student_and_key": {
                **describe(grouped.to_numpy(), group_count, student_lengths),
                "groups_spanning_other_keys": int(run_keys.duplicated(keep=False).groupby(
                    [run_keys[c] for c in run_keys.columns], dropna=False).any().sum()),
                "preserves_flattened_chronology": repeated == 0,
            },
        }
    offsets = starts_for(students, records[["assignment_id"]].to_numpy())
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "session_offsets.npy", offsets)
    report["session_offsets_sha256"] = sha256_file(output / "session_offsets.npy")
    proxy = AssignmentRunStore(ordered, offsets)
    report["split_windows"] = {}
    for name in SPLITS:
        dataset = AssignmentTargets(proxy, name)
        report["split_windows"][name] = {
            "windows": len(dataset),
            "targets": dataset.target_count,
            "windows_with_unscored_prefix": int(np.count_nonzero(dataset.metric_start)),
            "targets_in_first_assignment_run": int(dataset.lengths[proxy.session_number[dataset.targets] == 1].sum()),
        }
    (output / "session_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


class AssignmentRunStore:
    """Real assignment boundaries on the unchanged, ordered interaction arrays."""

    def __init__(self, ordered: Assist2009Store, offsets: np.ndarray) -> None:
        self.ordered = ordered
        self.root = ordered.root
        self.provenance = ordered.provenance
        self.question, self.skill, self.correct = ordered.question, ordered.skill, ordered.correct
        self.session_offsets = offsets
        self.session_student = ordered.student[offsets[:-1]]
        self.student_offsets = np.searchsorted(offsets, ordered.student_offsets)
        if (offsets[0] != 0 or offsets[-1] != len(self.question) or np.any(np.diff(offsets) <= 0)
                or not np.array_equal(offsets[self.student_offsets], ordered.student_offsets)):
            raise ValueError("assignment runs do not partition the student interaction arrays")
        self.session_number = (np.arange(len(offsets) - 1)
                               - self.student_offsets[self.session_student - 1] + 1)
        self.metadata = {**ordered.metadata, "sessions": len(offsets) - 1,
                         "session_protocol": SESSION_PROTOCOL}

    @classmethod
    def open(cls, ordered: Assist2009Store, root: Path) -> AssignmentRunStore:
        report = json.loads((root / "session_audit.json").read_text())
        offsets = root / "session_offsets.npy"
        if (report["dataset_provenance"] != ordered.provenance
                or report["selected_protocol"] != SESSION_PROTOCOL
                or report["session_offsets_sha256"] != sha256_file(offsets)):
            raise ValueError("assignment run offsets do not match the dataset provenance")
        store = cls(ordered, np.load(offsets, mmap_mode="r"))
        store.session_offsets_sha256 = report["session_offsets_sha256"]
        return store

    def session(self, index: int) -> Session:
        return self.actions(int(self.session_offsets[index]), int(self.session_offsets[index + 1]))

    def actions(self, start: int, stop: int) -> Session:
        return Session(self.question[start:stop], self.skill[start:stop], self.correct[start:stop])


class AssignmentTargets(RollingSessionDataset):
    """Score one split's events per real run, with online unscored prefixes."""

    def __init__(self, store: AssignmentRunStore, split: str, *, history_sessions: int = 15):
        self.store = store
        self.split = SPLITS[split]
        self.history_sessions = history_sessions
        self.events = np.flatnonzero(store.ordered.split == self.split)
        self.targets, first, self.lengths = np.unique(
            np.searchsorted(store.session_offsets, self.events, side="right") - 1,
            return_index=True, return_counts=True,
        )
        starts = store.session_offsets[self.targets]
        self.metric_start = self.events[first] - starts
        self.window_stop = self.events[first + self.lengths - 1] + 1
        if np.any(self.window_stop - starts - self.metric_start != self.lengths):
            raise ValueError("noncontiguous metric targets in assignment run")
        self.target_count = len(self.events)
        self.shapes = tuple(self._shape(i) for i in range(len(self)))

    def __getitem__(self, index: int) -> SessionExample:
        target = int(self.targets[index])
        start = int(self.store.session_offsets[target])
        h_start, h_stop = self._history_bounds(target)
        history = [self.store.session(i) for i in range(h_start, h_stop)]
        return SessionExample(history, self.store.actions(start, int(self.window_stop[index])),
                              self.split, metric_start=int(self.metric_start[index]),
                              event_start=start)

    def shape(self, index: int) -> ExampleShape:
        return self.shapes[index]

    def _shape(self, index: int) -> ExampleShape:
        target = int(self.targets[index])
        h_start, h_stop = self._history_bounds(target)
        offsets = self.store.session_offsets
        return ExampleShape(tuple(np.diff(offsets[h_start:h_stop + 1]).tolist()),
                            int(self.window_stop[index] - offsets[target]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("ordered", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, Assist2009Store(args.ordered), args.output), indent=2))


if __name__ == "__main__":
    main()
