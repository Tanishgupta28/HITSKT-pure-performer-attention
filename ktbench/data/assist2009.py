"""ASSIST2009 event-order preprocessing for timestamp-independent baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ktbench.data.flat import _stable_ids, sha256_file
from ktbench.data.rolling_targets import RollingTargetDataset


PROTOCOL = "assist2009_ordered_interactions_60_20_20_v1"
SOURCE = "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010"
SPLITS = RollingTargetDataset.SPLITS


def preprocess(source: Path, output: Path) -> dict:
    raw = pd.read_csv(source, encoding="latin1", low_memory=False)
    identity = ["order_id", "user_id", "problem_id", "assignment_id", "sequence_id"]
    required = identity + ["correct", "skill_id"]
    missing = raw.isna().sum().astype(int).to_dict()
    data = raw[required].apply(pd.to_numeric, errors="coerce")
    invalid = (~np.isfinite(data[identity]) | data[identity].mod(1).ne(0)
               | data[identity].le(0)).any(axis=1) | ~data.correct.isin([0, 1])
    if invalid.any():
        raise ValueError(f"{invalid.sum()} rows with invalid required identifiers or correctness")
    groups = data.groupby("order_id", sort=False)
    conflicts = groups[identity[1:] + ["correct"]].nunique().gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"{conflicts.sum()} problem logs have conflicting identifiers or labels")
    invalid_skill = data.skill_id.notna() & (
        ~np.isfinite(data.skill_id) | data.skill_id.mod(1).ne(0) | data.skill_id.le(0)
    )
    invalid_skill |= raw.skill_id.notna() & data.skill_id.isna()
    data.loc[invalid_skill, "skill_id"] = np.nan
    tags = groups.skill_id.agg(lambda values: tuple(sorted({int(v) for v in values.dropna()})))

    events = (data.drop_duplicates("order_id").drop(columns="skill_id")
              .sort_values(["user_id", "order_id"]).reset_index(drop=True))
    events["student"], students = _stable_ids(events.user_id)
    events["question"], questions = _stable_ids(events.problem_id)
    tag_keys = [tags.loc[order] for order in events.order_id]
    vocabulary = sorted(set(tag_keys))
    skill_ids = {key: index for index, key in enumerate(vocabulary, 1)}
    events["skill"] = np.fromiter((skill_ids[key] for key in tag_keys), dtype=np.int32)
    by_student = events.groupby("student", sort=False)
    position = by_student.cumcount().to_numpy()
    holdout = by_student.order_id.transform("size").to_numpy() // 5
    lengths = by_student.order_id.transform("size").to_numpy()
    events["split"] = np.where(position < lengths - 2 * holdout, 0,
                               np.where(position < lengths - holdout, 1, 2)).astype(np.uint8)
    offsets = np.r_[np.flatnonzero(position == 0), len(events)]
    counts = {name: int(events.split.eq(code).sum()) for name, code in SPLITS.items()}
    fields = ("order_id", "student", "question", "skill", "correct", "split")
    summary = {
        "dataset": "assist2009", "protocol": PROTOCOL, "source": SOURCE,
        "source_csv": str(source), "source_sha256": sha256_file(source),
        "source_columns": raw.columns.tolist(), "raw_rows": len(raw),
        "missing_by_column": missing, "invalid_required_rows": int(invalid.sum()),
        "invalid_skill_rows": int(invalid_skill.sum()), "conflicting_order_ids": int(conflicts.sum()),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "rows_collapsed_by_order_id": len(raw) - len(events),
        "rows_repeating_same_order_skill_pair": int(data.duplicated(["order_id", "skill_id"]).sum()),
        "raw_atomic_skills": int(data.skill_id.nunique()),
        "multi_skill_interactions": sum(len(key) > 1 for key in tag_keys),
        "untagged_interactions": tag_keys.count(()),
        "students": len(students), "interactions": len(events),
        "num_questions": len(questions), "num_skills": len(vocabulary), "sessions": None,
        "target_counts": counts,
        "target_ratios": {name: count / len(events) for name, count in counts.items()},
        "correct_targets": {name: int(events.loc[events.split.eq(code), "correct"].sum())
                            for name, code in SPLITS.items()},
        "students_with_train_only": int(np.count_nonzero(np.diff(offsets) < 5)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".assist2009-") as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        for name in fields:
            np.save(staging / f"{name}.npy", events[name].to_numpy())
        np.save(staging / "student_offsets.npy", offsets)
        for mapping, name in ((students, "student"), (questions, "question")):
            mapping.to_csv(staging / f"{name}_mapping.csv", index=False)
        pd.DataFrame({
            "skill_id": list(skill_ids.values()),
            "original_tag_ids": [json.dumps(key) for key in vocabulary],
            "is_untagged": [not key for key in vocabulary],
        }).to_csv(staging / "skill_mapping.csv", index=False)
        summary["array_sha256"] = {
            p.name: sha256_file(p) for p in sorted(staging.glob("*.npy"))
        }
        (staging / "metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (staging / "_SUCCESS").write_text("complete\n")
        staging.rename(output)
    return summary


class Assist2009Store:
    """Memory-mapped ordered interactions with no timestamp or session representation."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if not (self.root / "_SUCCESS").is_file():
            raise ValueError(f"incomplete ASSIST2009 store: {root}")
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata["protocol"] != PROTOCOL:
            raise ValueError("unexpected ASSIST2009 preprocessing protocol")
        for name, digest in self.metadata["array_sha256"].items():
            if sha256_file(self.root / name) != digest:
                raise ValueError(f"ASSIST2009 corruption in {name}")
        for name in ("order_id", "student", "question", "skill", "correct", "split", "student_offsets"):
            setattr(self, name, np.load(self.root / f"{name}.npy", mmap_mode="r"))
        self.provenance = {
            "protocol": PROTOCOL,
            "source_sha256": self.metadata["source_sha256"],
            "metadata_sha256": sha256_file(self.root / "metadata.json"),
            "target_order_id_sha256": {
                name: hashlib.sha256(self.order_id[self.split == code].astype("<i8").tobytes()).hexdigest()
                for name, code in SPLITS.items()
            },
        }

    def targets(self, split: str, history_length: int) -> OrderedTargets:
        return OrderedTargets(self, split, history_length)


class OrderedTargets:
    """One target per interaction and strictly earlier same-student histories."""

    def __init__(self, store: Assist2009Store, split: str, history_length: int):
        self.store = store
        self.split = SPLITS[split]
        self.history_length = history_length
        self.events = np.flatnonzero(store.split == self.split)

    def __len__(self) -> int:
        return len(self.events)

    def history_lengths(self, indices: np.ndarray) -> np.ndarray:
        events = self.events[indices]
        first = self.store.student_offsets[self.store.student[events] - 1]
        return np.minimum(events - first, self.history_length)

    def vectorized_batch(self, indices):
        from ktbench.data.fast_rolling import materialize_batch

        events = self.events[indices]
        return materialize_batch(self.store, events, self.store.student[events],
                                 self.history_lengths(indices), self.split,
                                 order=self.store.order_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(preprocess(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
