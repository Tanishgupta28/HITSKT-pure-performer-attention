#!/usr/bin/env python3
"""Generate rolling HiTSKT knowledge-state figures and inference evidence on CPU."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, replace
from functools import lru_cache, partial
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ktbench.batching import CORRECT_BOS_ID, CORRECT_PAD_ID, Session, SessionExample, collate_hitskt
from ktbench.data.session_store import RollingSessionDataset, SessionStore
from ktbench.models.hitskt import HiTSKT
from ktbench.models.performer import CausalLinearAttention, sinusoidal_positions
from report_graphs.Junyi_original_ranking.replay_scores import last_action

NAMES = {"assist2012": "ASSIST2012", "junyi": "Junyi", "ednet_kt1": "EdNet-KT1"}
AUDIT = ROOT / "reports/independent_correctness_audit"
RESULTS = ROOT / "reports/final_results.csv"
SEED = 42
DPI = 330
ATOL = 1e-5
TITLE = "Knowledge State Prediction of a Student Using HiTSKT"
SELECTION = (
    "Test sessions only. Rank students by SHA-256 of seed 42 and the processed student ID. "
    "Prefer 50 consecutive rows with >=4 completed sessions at entry, both labels, and five "
    "train-backed skills with >=2 attempts each. Search by student rank and earliest window. "
    "Fallback tiers relax to >=1 prior session, then 50/49…40/51…60 rows and 3–5 repeated "
    "skills. Window boundaries must contain whole availability groups. Exclude EdNet <UNTAGGED> "
    "from skill rows. Skill ties use ascending processed IDs."
)
CONVENTION = {
    "assist2012": "Post first graded action. Equal timestamps update together at the final group row.",
    "junyi": "Post logged response group. Equal timestamps update together at the final group row.",
    "ednet_kt1": (
        "Post bundle at its final row, with closure estimated by the maximum delivery time plus "
        "elapsed time. Earlier rows show the state before the bundle. Merge equal delivery times. "
        "Require previous groups to have closed by the delivery time of each plotted group."
    ),
}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def require(test, detail):
    if not test:
        raise ValueError(detail)


def audit_path(dataset):
    return AUDIT / f"{dataset}_evaluation_guarded_manifest.json"


def checkpoint_info(dataset):
    with RESULTS.open(newline="") as stream:
        row, = [r for r in csv.DictReader(stream) if (r["dataset"], r["model"]) == (dataset, "HiTSKT")]
    path = ROOT / row["experiment_directory"]
    require(path == ROOT / "experiments/hitskt" / dataset / "full", "Unexpected HiTSKT experiment path")
    config = read_json(path / "config.json")
    results = read_json(path / "final_results.json")
    stats = read_json(path / "dataset_stats.json")
    require((path / "_SUCCESS").is_file() and results["status"] == "complete"
            and results["best_checkpoint_reloaded"], "Expected a completed best-checkpoint run")
    for field in ("dataset", "model", "best_epoch", "epochs_completed", "best_validation_auc",
                  "seed", "total_parameters"):
        require(str(results[field]) == row[field] and config[field] == results[field],
                f"Experiment disagrees with final_results.csv on {field}")
    require(float(row["test_auc"]) == results["test"]["roc_auc"], "Test ROC-AUC mismatch")
    require(int(row["test_examples"]) == results["test"]["targets"] == stats["target_counts"]["test"],
            "Test target count mismatch")
    require(config["history_sessions"] == 15 and results["seed"] == SEED
            and config["attention"] == "pure causal ELU+1 Performer linear attention",
            "Unexpected attention, context size or benchmark seed")
    cp = path / "best_model.pt"
    recorded = (read_json(audit_path(dataset))["historical_artifact_sha256"]["best_model.pt"] if dataset == "junyi"
                else read_json(path / "independent_evaluation.json")["checkpoint_sha256"])
    sha = digest(cp)
    require(sha == recorded, "Checkpoint hash disagrees with historical evaluation")
    payload = torch.load(cp, map_location="cpu", mmap=True, weights_only=True)
    require(payload["format_version"] == 1 and payload["epoch"] == results["best_epoch"]
            and payload["validation_roc_auc"] == results["best_validation_auc"]
            and payload["seed"] == SEED, "Checkpoint metadata disagrees with experiment")
    model_config = payload["config"]
    for field, value in model_config.items():
        expected = (stats[{"num_questions": "questions", "num_skills": "skills"}[field]]
                    if field in ("num_questions", "num_skills") else config[field])
        require(value == expected, f"Checkpoint configuration mismatch for {field}")
    if dataset != "junyi":
        evaluation = read_json(path / "independent_evaluation.json")
        require(evaluation["verified"] and evaluation["test"] == results["test"],
                "Historical independent replay did not match results")
    metrics_path = path / "metrics.csv"
    with metrics_path.open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    validation = [row for row in metrics if row["split"] == "validation"]
    require([int(row["epoch"]) for row in validation] == list(range(1, results["epochs_completed"] + 1)),
            "Training log epoch count mismatch")
    best = max(validation, key=lambda row: float(row["roc_auc"]))
    require(int(best["epoch"]) == payload["epoch"] and float(best["roc_auc"]) == payload["validation_roc_auc"],
            "Training log best validation epoch mismatch")
    test, = [row for row in metrics if row["split"] == "test"]
    require(int(test["epoch"]) == payload["epoch"] and float(test["roc_auc"]) == results["test"]["roc_auc"],
            "Training log best-checkpoint test mismatch")
    return cp, {
        "checkpoint": str(cp.relative_to(ROOT)), "checkpoint_sha256": sha,
        "checkpoint_epoch": payload["epoch"], "epochs_completed": results["epochs_completed"],
        "test_roc_auc": results["test"]["roc_auc"], "validation_roc_auc": payload["validation_roc_auc"],
        "checkpoint_config": model_config, "experiment_config": config, "dataset_stats": stats,
        "benchmark_row_matched": True,
        "historical_evidence": str((audit_path(dataset) if dataset == "junyi" else
                                    path / "independent_evaluation.json").relative_to(ROOT)),
    }


def open_store(dataset, path, checkpoint):
    store = SessionStore(path)
    for stat, key in (("interactions", "interactions"), ("students", "students"),
                      ("questions", "num_questions"), ("skills", "num_skills")):
        require(checkpoint["dataset_stats"][stat] == store.metadata[key],
                f"Store differs from benchmark on {stat}")
    if dataset == "assist2012":
        expected = {"metadata.json": checkpoint["experiment_config"]["session_store_metadata_sha256"]}
        require(store.strict_timestamp_history, "ASSIST2012 requires the strict timestamp session store")
        event_file = path.parent / "events/part-00000.parquet"
        require(digest(event_file) == store.metadata["events_sha256"], "ASSIST2012 event file checksum mismatch")
    else:
        expected = read_json(audit_path(dataset))["session_store_sha256"]
    hashes = {p.name: digest(p) for p in sorted(path.iterdir()) if p.suffix == ".npy"
              or p.name in ("_SUCCESS", "metadata.json")}
    require(all(hashes.get(k) == v for k, v in expected.items()), "Store checksum mismatch")
    targets = RollingSessionDataset(store, "test", history_sessions=15)
    test_sessions = np.flatnonzero(store.session_split == 2) if store.strict_timestamp_history else targets.targets
    count = int(np.sum(np.diff(store.session_offsets)[test_sessions]))
    require(count == checkpoint["dataset_stats"]["target_counts"]["test"], "Test cohort count mismatch")
    source = {
        "assist2012": ("https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect",
                       store.metadata.get("source_archive_sha256")),
        "junyi": ("https://github.com/pokerme7777/HiTSKT/tree/main/dataset",
                  "8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a"),
        "ednet_kt1": ("https://github.com/riiid/ednet",
                     "0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535"),
    }
    url, source_sha = source[dataset]
    return store, targets, {"session_store": store.metadata, "files_sha256": hashes,
                            "historically_matched_files": sorted(expected), "test_targets": count,
                            "source": url, "source_sha256": source_sha}


def question_banks(store, dataset):
    """Count processed (skill, question) pairs over every original training row."""
    stride = int(store.metadata["num_questions"]) + 1
    train_ranges = np.flatnonzero(store.session_split == 0)
    counts = Counter()
    for start in range(0, len(store.question), 1_000_000):
        stop = min(start + 1_000_000, len(store.question))
        offsets = store.session_offsets
        first = int(np.searchsorted(offsets, start, side="right")) - 1
        last = int(np.searchsorted(offsets, stop, side="left"))
        left, right = np.searchsorted(train_ranges, [first, last])
        sessions = train_ranges[left:right]
        changes = np.zeros(stop - start + 1, dtype=np.int8)
        changes[np.maximum(offsets[sessions], start) - start] += 1
        changes[np.minimum(offsets[sessions + 1], stop) - start] -= 1
        train = changes[:-1].cumsum(dtype=np.int32) > 0
        pairs = store.skill[start:stop][train].astype(np.int64) * stride + store.question[start:stop][train]
        values, occurrences = np.unique(pairs, return_counts=True)
        counts.update(dict(zip(map(int, values), map(int, occurrences))))
    grouped = defaultdict(list)
    for pair, frequency in counts.items():
        skill, question = divmod(pair, stride)
        if (dataset, skill) != ("ednet_kt1", 1):
            grouped[skill].append((question, frequency))
    banks = {skill: sorted(items, key=lambda x: (-x[1], x[0]))[:5] for skill, items in grouped.items()}
    require(sum(counts.values()) == int(np.diff(store.session_offsets)[train_ranges].sum()),
            "Training bank coverage mismatch")
    return banks, sum(counts.values())


class EdNetTiming:
    """Read auxiliary KT1 timing for candidate learners and check alignment to the store."""

    def __init__(self, events):
        self.files = sorted(events.glob("*.parquet"))
        require(bool(self.files), f"EdNet timing requires the benchmark event shards at {events}")
        self.starts = [0]
        self.groups = []
        for file in self.files:
            metadata = pq.ParquetFile(file).metadata
            for group in range(metadata.num_row_groups):
                self.groups.append((file, group))
                self.starts.append(self.starts[-1] + metadata.row_group(group).num_rows)

    @lru_cache(maxsize=8)
    def read(self, index):
        file, group = self.groups[index]
        columns = ["timestamp", "question_id", "skill_id", "correct", "elapsed_time", "solving_id"]
        table = pq.ParquetFile(file).read_row_group(group, columns=columns)
        return {key: table[key].to_numpy() for key in columns}

    def for_student(self, store, lo, hi):
        require(self.starts[-1] == len(store.correct), "EdNet timing shard row count mismatch")
        arrays = defaultdict(list)
        for index in range(bisect_right(self.starts, lo) - 1, bisect_right(self.starts, hi - 1)):
            begin, stop = self.starts[index:index + 2]
            for key, value in self.read(index).items():
                arrays[key].append(value[max(lo - begin, 0):min(stop, hi) - begin])
        fields = {key: np.concatenate(value) for key, value in arrays.items()}
        for field, source in (("timestamp", store.timestamp_ms), ("question_id", store.question),
                              ("skill_id", store.skill), ("correct", store.correct)):
            require(np.array_equal(fields[field], source[lo:hi]), f"EdNet timing is misaligned on {field}")
        require(np.all(fields["elapsed_time"] >= 0), "Negative EdNet elapsed time")
        return fields["timestamp"] + fields["elapsed_time"], fields["solving_id"]


@dataclass
class Student:
    lo: int
    hi: int
    groups: np.ndarray
    group_start: np.ndarray
    group_stop: np.ndarray
    overlap_prefix: np.ndarray


def profile(store, learner, timing):
    first, stop = map(int, store.student_offsets[learner:learner + 2])
    lo, hi = int(store.session_offsets[first]), int(store.session_offsets[stop])
    times = store.timestamp_ms[lo:hi]
    if np.any(times[1:] < times[:-1]):
        return None
    joined = times[1:] == times[:-1]
    ready = times
    if timing is not None:
        ready, bundle = timing.for_student(store, lo, hi)
        bundle_starts = bundle[np.r_[True, bundle[1:] != bundle[:-1]]]
        if len(np.unique(bundle_starts)) != len(bundle_starts):
            return None
        joined |= bundle[1:] == bundle[:-1]
    breaks = np.r_[0, np.flatnonzero(~joined) + 1, hi - lo]
    if not np.isin(store.session_offsets[first:stop + 1] - lo, breaks).all():
        return None
    group_ready = np.maximum.reduceat(ready, breaks[:-1])
    overlaps = np.r_[False, group_ready[:-1] >= times[breaks[1:-1]]]
    groups = np.repeat(np.arange(len(breaks) - 1), np.diff(breaks))
    return Student(lo, hi, groups, breaks[:-1], breaks[1:],
                   np.r_[0, np.cumsum(overlaps)])


@dataclass
class Window:
    student: Student
    start: int
    stop: int
    skills: list[int]
    tier: str


def choose_students(store, targets, banks, timing):
    test_starts = targets.targets[np.r_[True, np.diff(store.session_student[targets.targets]) != 0]]
    firsts = test_starts - store.session_number[test_starts] + 1
    learners = np.searchsorted(store.student_offsets, firsts)
    cohort = sorted(zip(learners.tolist(), test_starts.tolist()), key=lambda x: (
        hashlib.sha256(f"{SEED}:{int(store.session_student[store.student_offsets[x[0]]])}".encode()).digest(),
        x[0],
    ))
    selected = {}
    tiers = [(4, [50], 5, "preferred"), (1, [50], 5, "fewer preceding sessions"),
             (1, [50, *range(49, 39, -1), *range(51, 61)], 3, "shorter/longer window or fewer skills")]
    for preceding, lengths, skill_min, tier in tiers:
        for learner, first_test in cohort:
            if learner in selected:
                continue
            first, stop = map(int, store.student_offsets[learner:learner + 2])
            first_allowed = max(first_test, first + preceding)
            if first_allowed >= stop or store.session_offsets[stop] - store.session_offsets[first_allowed] < min(lengths):
                continue
            student = profile(store, learner, timing)
            if student is None:
                continue
            low = int(store.session_offsets[first_allowed]) - student.lo
            skills = store.skill[student.lo:student.hi]
            answers = store.correct[student.lo:student.hi]
            group_edges = set(student.group_stop.tolist())
            found = None
            for length in lengths:
                for relative in student.group_start[student.group_start >= low]:
                    end = relative + length
                    if end > len(skills):
                        break
                    if end not in group_edges or not 0 < answers[relative:end].sum() < length:
                        continue
                    last_group = student.groups[end - 1]
                    # Require observability from the earliest session used by the first probe.
                    session = int(np.searchsorted(store.session_offsets, student.lo + relative, side="right")) - 1
                    context_lo = int(store.session_offsets[max(first, session - 15)]) - student.lo
                    context_group = student.groups[context_lo]
                    overlap = student.overlap_prefix[last_group + 1] - student.overlap_prefix[context_group + 1]
                    if overlap:
                        continue
                    frequencies = Counter(skills[relative:end].tolist())
                    ranked = [k for k in sorted(frequencies, key=lambda k: (-frequencies[k], k))
                              if k in banks and frequencies[k] >= 2][:5]
                    if len(ranked) >= skill_min:
                        found = Window(student, int(student.lo + relative), int(student.lo + end), ranked, tier)
                        break
                if found is not None:
                    selected[learner] = found
                    break
            if len(selected) == 5:
                return list(selected.values())
    raise ValueError(f"Only {len(selected)} eligible test students after selection fallbacks")


def probe_example(store, targets, session, visible_stop):
    first, stop = targets._history_bounds(session)
    history = tuple(store.session(i) for i in range(first, stop))
    start = int(store.session_offsets[session])
    return history, (
        store.question[start:visible_stop].tolist(),
        store.skill[start:visible_stop].tolist(),
        store.correct[start:visible_stop].tolist(),
    )


def examples(history, prefix, items, *, dummy=0, tail=()):
    q, k, y = prefix
    return [SessionExample(history, Session([*q, question, *(r[0] for r in tail)],
                                            [*k, skill, *(r[1] for r in tail)],
                                            [*y, dummy, *(r[2] for r in tail)]), 2)
            for question, skill in items]


def forward_probes(model, batch, history, items, position):
    """Share fixed-history memory, as in the Junyi ranking replay."""
    batch.target_correct_input[:, position + 1:] = CORRECT_PAD_ID
    batch.target_attention_mask[:, position + 1:] = False
    batch.target_labels[:, position:] = CORRECT_PAD_ID
    batch.target_metric_mask.zero_()
    memory = model.correct_padding_encoder(history, batch)
    prefix_elements = (position + 2) * model.config.width ** 2 // model.config.heads
    chunk_size = max(1, min(25, 8_388_608 // prefix_elements))
    probabilities = []
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        q, k = torch.tensor(chunk, dtype=torch.long).T
        n = len(chunk)
        question = batch.target_question.expand(n, -1).clone()
        skill = batch.target_skill.expand(n, -1).clone()
        question[:, position] = q
        skill[:, position] = k
        target = replace(batch, target_question=question, target_skill=skill,
                         target_attention_mask=batch.target_attention_mask.expand(n, -1))
        decoded = model.decoder(memory.expand(n, -1, -1), target)
        probabilities.extend(model.prediction(decoded[:, position]).squeeze(-1).sigmoid().tolist())
    result = np.array(probabilities)
    require(np.isfinite(result).all() and np.all((0 <= result) & (result <= 1)), "Invalid probe probability")
    return result


def close(a, b, name):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    require(error <= ATOL, f"{name} failed (absolute error {error:.8g})")
    return error


def verify_probes(model, collate, history, prefix, items, probs, tail):
    # Run the unmodified evaluation forward path, one probe per example, with padding.
    position = len(prefix[2])
    direct = []
    changed = []
    future = [(q, k, 1 - y) for q, k, y in tail]
    for item in items:
        plain = collate(examples(history, prefix, [item]))
        require(plain.target_correct_input[0, :position + 1].tolist() == [CORRECT_BOS_ID, *prefix[2]],
                "Current-session correctness shift mismatch")
        require(int(plain.history_count[0]) == len(history), "Historical session count mismatch")
        require(plain.target_attention_mask[0].all().item(), "Invalid attention mask on real tokens")
        # Pair unequal lengths to exercise padding. Both labels at and after the probe change.
        first, = examples(history, prefix, [item], tail=tail)
        second, = examples(history, prefix, [item], dummy=1, tail=future)
        padded = collate([second, *examples(history, prefix, [item])])
        require(not padded.target_attention_mask[1, position + 2:].any().item(), "Target padding mask mismatch")
        reference = model(plain).sigmoid()[0, position].item()
        direct.append(reference)
        changed.append(model(collate([first])).sigmoid()[0, position].item())
        changed.extend(model(padded).sigmoid()[:, position].tolist())
    return {
        "probes_compared": len(items),
        "direct_max_abs_error": close(direct, probs, "Optimized vs direct"),
        "alternate_future_max_abs_error": close(np.repeat(direct, 3), changed, "Target/future correctness causality"),
    }


@torch.inference_mode()
def predict_student(store, targets, model, window, banks):
    collate = partial(collate_hitskt, num_questions=model.config.num_questions, num_skills=model.config.num_skills)
    items = [(q, k) for k in window.skills for q, _ in banks[k]]
    student = window.student
    encoded = {}
    rows = []
    scores = []
    verification = []
    prior_session = None
    verify_positions = {window.start, (window.start + window.stop) // 2, window.stop - 1}
    for event in range(window.start, window.stop):
        session = int(np.searchsorted(store.session_offsets, event, side="right")) - 1
        relative = event - student.lo
        group = student.groups[relative]
        closes = relative + 1 == student.group_stop[group]
        visible = event + 1 if closes else int(student.lo + student.group_start[group])
        start = int(store.session_offsets[session])
        require(start <= visible <= event + 1, "Probe prefix escaped its current observed session")
        require(store.session_split[session] == 2, "Non-test interaction in selected window")
        history, prefix = probe_example(store, targets, session, visible)
        position = len(prefix[2])
        if session not in encoded:
            verify_positions.add(event)
            base = collate(examples(history, prefix, [items[0]]))
            actions = last_action(model, base, sinusoidal_positions)
            # Include the existing ActionEncoder in parity at each real session transition.
            direct_actions = torch.cat([
                model.action_encoder(collate(examples([observed], ([], [], []), [items[0]])))
                for observed in history
            ])
            action_error = close(actions.numpy(), direct_actions.numpy(), "Action encoder EOS")
            encoded[session] = model.session_encoder(actions, base)
        base = collate(examples(history, prefix, [items[0]]))
        probs = forward_probes(model, base, encoded[session], items, position)
        if event in verify_positions:
            future_stop = min(event + 4, student.hi, int(store.session_offsets[session + 1]))
            tail = list(zip(store.question[event + 1:future_stop].tolist(),
                            store.skill[event + 1:future_stop].tolist(),
                            store.correct[event + 1:future_stop].tolist()))
            if not tail:
                tail = [(*items[-1], 0)]
            indexes = sorted({0, len(items) // 2, len(items) - 1})
            check = verify_probes(model, collate, history, prefix, [items[i] for i in indexes], probs[indexes], tail)
            verification.append(dict(check, column=event - window.start + 1, action_max_abs_error=action_error))
        rows.append({
            "column": event - window.start + 1,
            "chronological_position": relative + 1,
            "observed_skill_id": int(store.skill[event]),
            "observed_correctness": int(store.correct[event]),
            "session_number": int(store.session_number[session]),
            "session_start": event == start,
            "session_change_in_figure": prior_session is not None and session != prior_session,
            "completed_history_sessions": len(history),
            "visible_current_session_responses": position,
            "outcome_group_end": bool(closes),
        })
        scores.append(probs)
        prior_session = session
    output = np.array(scores).T
    counts = [len(banks[k]) for k in window.skills]
    matrix = np.stack([values.mean(axis=0) for values in np.split(output, np.cumsum(counts)[:-1])])
    return rows, items, output, matrix, verification


@dataclass
class Panel:
    dataset: str
    ID: str
    skills: list[int]
    rows: list[dict]
    matrix: np.ndarray


def write_matrix(path, skills, matrix):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["skill_id", *range(1, matrix.shape[1] + 1)])
        writer.writerows([skill, *values.tolist()] for skill, values in zip(skills, matrix))
    reloaded = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    require(reloaded[:, 0].tolist() == skills and np.array_equal(reloaded[:, 1:], matrix),
            f"Matrix CSV round-trip failed: {path.name}")
    return reloaded[:, 1:]


def write_combined_matrix(path, panels):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["dataset", "student", "skill_id", "column", "probability"])
        for panel in panels:
            for k, values in zip(panel.skills, panel.matrix):
                writer.writerows((panel.dataset, panel.ID, k, i, value)
                                 for i, value in enumerate(values.tolist(), 1))


    with path.open(newline="") as stream:
        restored = list(csv.DictReader(stream))
    start = 0
    for panel in panels:
        size = panel.matrix.size
        values = [float(r["probability"]) for r in restored[start:start + size]]
        require(np.array_equal(np.asarray(values).reshape(panel.matrix.shape), panel.matrix),
                f"Combined matrix CSV round-trip failed: {path.name}")
        start += size
    require(start == len(restored), "Combined matrix CSV has an unexpected row count")


def render(path, panels, *, private):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 10, "savefig.facecolor": "white"}):
        fig = plt.figure(figsize=(12.1, 0.65 + len(panels) * (2.95 if private else 2.45)))
        try:
            grid = fig.add_gridspec(len(panels), 2, width_ratios=(46, 1), hspace=0.7 if private else 0.62,
                                    left=0.073, right=0.907, bottom=(0.28 if private else 0.19) / len(panels),
                                    top=1 - 0.63 / fig.get_figheight(), wspace=0.07)
            fig.suptitle(TITLE, fontsize=13, y=1 - 0.14 / fig.get_figheight())
            for i, panel in enumerate(panels):
                ax = fig.add_subplot(grid[i, 0])
                bar_ax = fig.add_subplot(grid[i, 1])
                matrix = panel.matrix
                image = ax.imshow(matrix, cmap="Blues", norm=Normalize(vmin=0, vmax=1),
                                  aspect="auto", interpolation="none")
                require(np.array_equal(image.get_array(), matrix), "Plot matrix differs from CSV")
                bar = fig.colorbar(image, cax=bar_ax, ticks=np.linspace(0, 1, 6), format="%.1f")
                bar.set_label("Predicted Correctness Probability", fontsize=8)
                bar.ax.tick_params(labelsize=8)
                bar.outline.set_linewidth(0.5)
                ax.set_title(f"{NAMES[panel.dataset]} · Student {panel.ID[-2:]}", pad=7)
                ax.set_ylabel("Skill ID")
                ax.set_yticks(range(len(panel.skills)), labels=panel.skills)
                ax.set_xlabel("Student Interaction Sequence")
                if private:
                    labels = [f"({r['observed_skill_id']},{r['observed_correctness']})" for r in panel.rows]
                    ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=7)
                else:
                    ticks = [0, *range(4, matrix.shape[1], 5)]
                    ax.set_xticks(ticks, labels=[i + 1 for i in ticks])
                ax.tick_params(length=2, width=0.5, pad=3)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
            fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.13)
        finally:
            plt.close(fig)


def save_panels(root, panels, *, private):
    subdir = root / "knowledge_state_heatmaps" / panels[0].dataset
    subdir.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        stem = subdir / panel.ID
        checked = write_matrix(stem.with_suffix(".csv"), panel.skills, panel.matrix)
        render(stem.with_suffix(".png"), [replace(panel, matrix=checked)], private=private)
    write_combined_matrix(subdir / "combined_students.csv", panels)
    render(subdir / "combined_students.png", panels, private=private)
    for extension in ("png", "csv"):
        shutil.copyfile(subdir / f"student_01.{extension}",
                        root / f"{panels[0].dataset}_hitskt_knowledge_state.{extension}")


def run_dataset(dataset, cp, verified, processed, private, public):
    folder = processed / dataset / "full"
    store, targets, provenance = open_store(dataset, folder / "session_store", verified)
    banks, training_count = question_banks(store, dataset)
    timing = EdNetTiming(folder / "events") if dataset == "ednet_kt1" else None
    windows = choose_students(store, targets, banks, timing)
    repeated = choose_students(store, targets, banks, timing)
    signature = lambda windows: [(w.start, w.stop, w.skills, w.tier) for w in windows]
    require(signature(repeated) == signature(windows), "Student selection was not deterministic")
    model = HiTSKT.from_checkpoint(cp, map_location="cpu").eval()
    require(sum(p.numel() for p in model.parameters()) == verified["experiment_config"]["total_parameters"],
            "Loaded parameter count differs from benchmark")
    require(not any(m.training for m in model.modules()), "HiTSKT is not entirely in evaluation mode")
    require(any(isinstance(m, CausalLinearAttention) for m in model.modules()), "Causal linear attention missing")
    root = private / "knowledge_state_heatmaps" / dataset
    root.mkdir(parents=True, exist_ok=True)
    panels, summaries = [], []
    for index, window in enumerate(windows, 1):
        name = f"student_{index:02d}"
        print(f"{dataset}: {name}, {window.stop - window.start} columns, {window.tier}", flush=True)
        rows, items, per_question, matrix, verification = predict_student(store, targets, model, window, banks)
        summaries.append({"student": name, "skills": window.skills, "columns": matrix.shape[1],
                          "selection_tier": window.tier, "direct_and_causality_checks": verification,
                          "probability_bounds": [float(matrix.min()), float(matrix.max())]})
        panels.append(Panel(dataset, name, window.skills, rows, matrix))
        write_json(root / f"{name}_metadata.json", {
            "dataset": dataset, "student": name, "skills": window.skills, "interactions": rows,
            "question_bank": {str(k): [{"question_id": q, "training_frequency": n} for q, n in banks[k]]
                              for k in window.skills},
            "selection": SELECTION, "selection_tier": window.tier, "convention": CONVENTION[dataset],
            "checkpoint": verified, "dataset_provenance": provenance,
        })
        with (root / f"{name}_question_probabilities.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["skill_id", "question_id", *range(1, matrix.shape[1] + 1)])
            writer.writerows((k, q, *values.tolist()) for (q, k), values in zip(items, per_question))
    save_panels(private, panels, private=True)
    if public is not None:
        save_panels(public, panels, private=False)
    return panels[0], {
        **verified, "status": "generated", "dataset_provenance": provenance, "training_bank_rows": training_count,
        "model_eval_and_dropout_disabled": True, "torch_inference_mode": True,
        "deterministic_selection_repeated": True, "numerical_matrix_and_plot_checks": True,
        "plotting_convention": CONVENTION[dataset], "students": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=["all", *NAMES], default=["all"])
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--private-output", type=Path, default=ROOT / ".inspection/hitskt_knowledge_state/report_graphs")
    parser.add_argument("--public-output", type=Path, help="Optional export with ordinal-only interaction labels")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    datasets = list(NAMES) if "all" in args.dataset else list(dict.fromkeys(args.dataset))
    private, public = args.private_output.resolve(), args.public_output
    if public is not None:
        public = public.resolve()
        require(not private.is_relative_to(public) and not public.is_relative_to(private),
                "Use separate private and public output locations")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = {"software": {"torch": torch.__version__, "numpy": np.__version__,
                             "matplotlib": matplotlib.__version__, "python": sys.version.split()[0]},
                "device": "cpu", "threads": args.threads, "seed": SEED, "dpi": DPI,
                "probability_absolute_tolerance": ATOL, "student_selection": SELECTION,
                "source_sha256": {name: digest(ROOT / name) for name in (
                    str(Path(__file__).resolve().relative_to(ROOT)), "ktbench/models/hitskt.py",
                    "ktbench/models/performer.py", "ktbench/batching.py", "ktbench/data/session_store.py",
                    "report_graphs/Junyi_original_ranking/replay_scores.py", "reports/final_results.csv")},
                "datasets": {}}
    panels = []
    for dataset in datasets:
        print(f"{dataset}: verifying checkpoint", flush=True)
        cp, verified = checkpoint_info(dataset)
        store = args.processed_root / dataset / "full/session_store"
        if not (store / "_SUCCESS").is_file():
            manifest["datasets"][dataset] = dict(verified, status="blocked",
                                                  reason="Original processed session store is unavailable.")
            print(f"{dataset}: original session store unavailable", file=sys.stderr, flush=True)
            continue
        panel, result = run_dataset(dataset, cp, verified, args.processed_root, private, public)
        panels.append(panel)
        manifest["datasets"][dataset] = result
    blocked = len(panels) != len(datasets)
    manifest["status"] = "incomplete" if blocked else "verified"
    for output, restricted in ((private, True), (public, False)):
        if output is None:
            continue
        data_dir = output / "knowledge_state_heatmaps"
        data_dir.mkdir(parents=True, exist_ok=True)
        if len(panels) >= 2:
            name = "hitskt_knowledge_state_all_datasets" if len(panels) == len(NAMES) else "hitskt_knowledge_state_available_datasets"
            write_combined_matrix(output / f"{name}.csv", panels)
            render(output / f"{name}.png", panels, private=restricted)
        output_manifest = json.loads(json.dumps(manifest))
        if not restricted:
            for result in output_manifest["datasets"].values():
                for student in result.get("students", []):
                    checks = student.pop("direct_and_causality_checks")
                    student["direct_and_causality_summary"] = {
                        "positions_checked": len(checks),
                        "probes_compared": sum(check["probes_compared"] for check in checks),
                        **{key: max(check[key] for check in checks) for key in (
                            "action_max_abs_error", "direct_max_abs_error", "alternate_future_max_abs_error")},
                    }
        outputs = [p for panel in panels for p in (
            *(data_dir / panel.dataset).iterdir(),
            *(output.glob(f"{panel.dataset}_hitskt_knowledge_state.*")),
        )]
        if len(panels) >= 2:
            outputs.extend(output.glob(f"{name}.*"))
        output_manifest["files_sha256"] = {
            str(p.relative_to(output)): digest(p) for p in sorted(outputs)
            if p.is_file() and (p.suffix in (".csv", ".png") or p.name.endswith("_metadata.json"))
        }
        write_json(data_dir / "reproducibility_manifest.json", output_manifest)
    return int(blocked)


if __name__ == "__main__":
    raise SystemExit(main())
