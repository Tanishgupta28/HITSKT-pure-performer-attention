#!/usr/bin/env python3
"""Verify the processed ASSIST2009 targets against the uploaded and published releases."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ktbench.data.assist2009 import Assist2009Store, SPLITS
from ktbench.data.flat import sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(source, processed, published):
    keys = ["order_id", "user_id", "problem_id", "correct"]
    raw = pd.read_csv(source, encoding="latin1", usecols=keys + ["assignment_id", "skill_id"])
    expected = raw.drop_duplicates("order_id").sort_values(["user_id", "order_id"])
    store = Assist2009Store(processed)
    require(store.metadata["source_sha256"] == sha256_file(source), "uploaded SHA-256 differs")
    student_map = pd.read_csv(processed / "student_mapping.csv").set_index("id").source_id
    question_map = pd.read_csv(processed / "question_mapping.csv").set_index("id").source_id
    skill_map = (pd.read_csv(processed / "skill_mapping.csv").set_index("skill_id")
                 .original_tag_ids.map(lambda value: frozenset(json.loads(value))))
    for key, values in zip(keys, (store.order_id, student_map.loc[store.student],
                                  question_map.loc[store.question], store.correct), strict=True):
        require(np.array_equal(expected[key], values), f"source {key} misalignment")
    tags = raw.dropna(subset="skill_id").groupby("order_id").skill_id.agg(frozenset).to_dict()
    for order, skill in zip(store.order_id, store.skill, strict=True):
        require(skill_map.loc[skill] == tags.get(order, frozenset()), f"wrong skills for {order}")
    for start, stop in zip(store.student_offsets[:-1], store.student_offsets[1:], strict=True):
        order = store.order_id[start:stop]
        n = stop - start
        holdout = n // 5
        require((np.diff(order) > 0).all(), "source interactions are not strictly ordered")
        require(np.array_equal(store.split[start:stop], np.repeat([0, 1, 2],
                                [n - 2*holdout, holdout, holdout])), "wrong split assignment")
    for name, code in SPLITS.items():
        a, b = store.targets(name, 199), store.targets(name, 99)
        require(np.array_equal(a.events, b.events) and len(a) == (store.split == code).sum(),
                f"unequal rolling targets for {name}")
        for dataset in (a, b):
            for start in range(0, len(dataset), 256):
                indices = np.arange(start, min(start + 256, len(dataset)))
                batch = dataset.vectorized_batch(indices)
                event = dataset.events[indices]
                positions = event[:, None] - np.arange(batch.history_mask.shape[1], 0, -1)
                valid = batch.history_mask.numpy()
                history = positions[valid]
                target = np.broadcast_to(event[:, None], valid.shape)[valid]
                require((store.student[history] == store.student[target]).all(), "cross-student history")
                require((store.order_id[history] < store.order_id[target]).all(), "future history")
                require(np.array_equal(batch.history_correct.numpy()[valid], store.correct[history]),
                        "wrong historical correctness")
                require(np.array_equal(batch.target_labels.numpy(), store.correct[event]),
                        "wrong target correctness")

    assignment_boundary = (expected.user_id.ne(expected.user_id.shift())
                           | expected.assignment_id.ne(expected.assignment_id.shift()))
    runs = expected.loc[assignment_boundary, ["user_id", "assignment_id"]]
    result = {
        "verified": True, "provenance": store.provenance,
        "verified_unique_targets": len(expected),
        "observed_assignment_runs": len(runs),
        "assignment_runs_revisiting_same_student_and_assignment": int(runs.duplicated().sum()),
        "split_statistics": {
            name: {
                "questions_absent_from_training": int(np.setdiff1d(store.question[store.split == code],
                                                                     store.question[store.split == 0]).size),
                "targets_with_question_absent_from_training": int(np.count_nonzero(
                    ~np.isin(store.question[store.split == code], store.question[store.split == 0]))),
                "skills_absent_from_training": int(np.setdiff1d(store.skill[store.split == code],
                                                                  store.skill[store.split == 0]).size),
            } for name, code in SPLITS.items()
        },
    }
    if published:
        corrected = published / "official_corrected_collapsed.csv"
        official = pd.read_csv(corrected, encoding="latin1", low_memory=False)
        aligned = expected[keys].merge(official[keys], on="order_id", how="outer",
                                       suffixes=("_uploaded", "_official"), indicator=True,
                                       validate="one_to_one")
        combined = published / "official_combined.zip"
        with zipfile.ZipFile(combined) as archive, archive.open(archive.namelist()[0]) as stream:
            combined_columns = pd.read_csv(stream, nrows=0, encoding="latin1").columns.tolist()
        result["published_release_checks"] = {
            "combined_zip_sha256": sha256_file(combined),
            "combined_columns": combined_columns,
            "corrected_sha256": sha256_file(corrected),
            "corrected_columns": official.columns.tolist(),
            "corrected_unique_interactions": len(official),
            "source_alignment_by_order_id": aligned["_merge"].value_counts().astype(int).to_dict(),
            "shared_order_id_field_mismatches": {
                key: int((aligned[f"{key}_uploaded"].ne(aligned[f"{key}_official"])
                          & aligned["_merge"].eq("both")).sum()) for key in keys[1:]
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("processed", type=Path)
    parser.add_argument("--published", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.processed, args.published), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
