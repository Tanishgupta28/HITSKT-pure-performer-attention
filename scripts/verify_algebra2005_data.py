#!/usr/bin/env python3
"""Audit the full Algebra store and the target views used by all five models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ktbench.batching import collate_hitskt
from ktbench.data.algebra2005 import AlgebraSessionTargets
from ktbench.data.fast_rolling import vectorized_batch
from ktbench.data.flat import sha256_file
from ktbench.data.rolling_targets import RollingTargetDataset
from ktbench.data.session_store import SessionStore, SPLIT_NAMES
from ktbench.rkt.data import RKTTargetDataset


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    store = SessionStore(root / "session_store")
    parquet = root / "events" / "part-00000.parquet"
    require(store.strict_timestamp_history and sha256_file(parquet) == store.metadata["events_sha256"],
            "store provenance mismatch")
    table = pq.read_table(parquet)
    arrays = {name: table[name].to_numpy() for name in table.column_names}
    row = arrays["Row"]
    require(len(np.unique(row)) == len(row), "duplicate raw event identity")
    for disk, source in (("question", "question_id"), ("skill", "skill_id"),
                         ("correct", "correct"), ("timestamp_ms", "timestamp")):
        require(np.array_equal(getattr(store, disk), arrays[source]), f"{disk} differs from parquet")
    counts = np.diff(store.session_offsets)
    for disk, source in (("session_student", "student_id"), ("session_number", "session_id"),
                         ("session_split", "split")):
        require(np.array_equal(np.repeat(getattr(store, disk), counts), arrays[source]),
                f"{disk} differs from parquet")
    change_student = arrays["student_id"][1:] != arrays["student_id"][:-1]
    gaps = np.diff(store.timestamp_ms)
    starts = np.r_[0, 1 + np.flatnonzero(change_student | (gaps >= 36_000_000)), len(row)]
    require(np.array_equal(starts, store.session_offsets), "wrong ten-hour session boundaries")
    tied = ~change_student & (gaps == 0)
    require(np.all(change_student | (gaps >= 0)) and np.all(np.diff(row)[tied] > 0),
            "timestamps or tie ordering differ")
    history_ends = store.history_stop(np.arange(len(row)))
    has_history = history_ends > np.repeat(
        store.session_offsets[store.student_offsets[:-1]],
        np.diff(store.session_offsets[store.student_offsets]),
    )
    require(np.all(store.timestamp_ms[history_ends[has_history] - 1] < store.timestamp_ms[has_history]),
            "non-past event in a strict-time history")
    for start, stop in zip(store.student_offsets[:-1], store.student_offsets[1:], strict=True):
        holdout = (stop - start) // 5
        expected = np.repeat([0, 1, 2], [stop - start - 2 * holdout, holdout, holdout])
        require(holdout > 0 and np.array_equal(store.session_split[start:stop], expected),
                "invalid student cohort or chronological split")

    hashes = {}
    for split, index in SPLIT_NAMES.items():
        expected = np.flatnonzero(arrays["split"] == index)
        for klass, settings in ((RollingTargetDataset, {"history_length": 199}),
                                (RollingTargetDataset, {"history_length": 99}), (RKTTargetDataset, {})):
            dataset = klass(store, split, **settings)
            _, events = dataset._events_for_indices(np.arange(len(dataset)))
            require(np.array_equal(events, expected), f"{klass.__name__} targets differ for {split}")
            indices = np.r_[np.arange(min(6, len(events))), np.flatnonzero(history_ends[events] != events)]
            for target_index in indices:
                example = dataset[int(target_index)]
                require(np.all(example.history_timestamp_ms < example.target_timestamp_ms),
                        f"{klass.__name__} exposes simultaneous correctness")
            if klass is RollingTargetDataset:
                batch = vectorized_batch(dataset, indices)
                for j, target_index in enumerate(indices):
                    example = dataset[int(target_index)]
                    mask = batch.history_mask[j]
                    require(np.array_equal(batch.history_correct[j, mask], example.history_correct),
                            "optimized batch differs from reference history")
        hierarchical = AlgebraSessionTargets(store, split)
        observed = []
        for start in range(0, len(hierarchical), 8):
            examples = [hierarchical[i] for i in range(start, min(start + 8, len(hierarchical)))]
            batch = collate_hitskt(examples, num_questions=store.metadata["num_questions"],
                                   num_skills=store.metadata["num_skills"])
            observed.extend(batch.target_event_ids[batch.target_metric_mask].tolist())
            for i, example in enumerate(examples):
                length = len(example.target)
                ids = batch.target_event_ids[i, :length].numpy()
                scored = batch.target_metric_mask[i, :length].numpy()
                require(np.all(np.diff(store.timestamp_ms[ids])[scored[1:]] > 0),
                        "hierarchical decoder can see a same-clock outcome")
        require(np.array_equal(np.sort(observed), expected), f"HiTSKT targets differ for {split}")
        keys = np.column_stack((row[expected], store.correct[expected])).astype("<i8")
        hashes[split] = {"targets": len(expected), "source_row_and_label_sha256":
                         hashlib.sha256(keys.tobytes()).hexdigest()}
    return {"verified": True, "session_store_metadata_sha256": sha256_file(store.root / "metadata.json"),
            "source_target_manifests": hashes, "same_clock_additional_events": int(tied.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.processed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
