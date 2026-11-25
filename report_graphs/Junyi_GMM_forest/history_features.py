"""Extract pre-assessment features by learner and composite skill on Junyi."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def extract(raw, export, repo, output):
    fields = ["student_id", "original_source_student_id", "target_session_id",
              "target_position", "target_question_id", "history_session_count",
              "history_interaction_count"]
    events = pd.read_csv(export, usecols=fields)
    first = (events[events.target_position == 1].sort_values(["student_id", "target_session_id"])
             .drop_duplicates("student_id"))
    first = first[(first.history_session_count >= 4) & (first.history_interaction_count >= 20)]
    first = first.set_index("original_source_student_id")
    del events
    maps = repo / "reports/datasets/junyi"
    questions = pd.read_csv(maps/"questions.csv").set_index("source_question_id").question_id
    skills = pd.read_csv(maps/"question_skills.csv").set_index("question_id").skill_id
    types = dict(studentId="int32", startTime="int64", problemId="int32", correct="uint8")
    chunks = pd.read_csv(raw, usecols=list(types), dtype=types, chunksize=300000)
    data = pd.concat([c[c.studentId.isin(first.index)] for c in chunks], ignore_index=True)
    data.sort_values(["studentId", "startTime"], kind="stable", inplace=True)
    results = []
    for source, group in data.groupby("studentId", sort=False):
        meta = first.loc[source]
        times = group.startTime.to_numpy()
        starts = np.r_[0, 1 + np.flatnonzero(np.diff(times) >= 36000), len(times)]
        cutoff = int(meta.target_session_id) - 1
        assert cutoff == len(starts)-1-(len(starts)-1)//5
        start, end = starts[max(0, cutoff-15)], starts[cutoff]
        assert end-start == meta.history_interaction_count
        assert questions[group.problemId.iloc[end]] == meta.target_question_id
        history = group.iloc[start:end]
        ids = skills.loc[questions.loc[history.problemId].to_numpy()].to_numpy()
        distinct, index = np.unique(ids, return_inverse=True)
        count = np.bincount(index)
        right = np.bincount(index, weights=history.correct)
        last = np.zeros(len(distinct), dtype="int64")
        np.maximum.at(last, index, times[start:end])
        days = (times[end-1]-last)/86400
        results.extend(zip([int(meta.student_id)]*len(count), distinct, count, right/count, days))
    frame = pd.DataFrame(results, columns=["student", "skill", "past_attempts", "past_accuracy", "days_since"])
    assert not frame.duplicated(["student", "skill"]).any()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["raw", "export", "repo", "output"]:
        parser.add_argument("--"+key, type=Path, required=True)
    extract(**vars(parser.parse_args()))
