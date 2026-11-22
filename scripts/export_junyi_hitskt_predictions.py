#!/usr/bin/env python3
"""Export or replay 15 genuine Junyi test predictions from the original HiTSKT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, fields, replace
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from ktbench.batching import (  # noqa: E402
    CORRECT_BOS_ID,
    CORRECT_PAD_ID,
    HiTSKTBatch,
    collate_hitskt,
)
from ktbench.data.session_store import SPLIT_NAMES, RollingSessionDataset, SessionStore  # noqa: E402
from ktbench.models.hitskt import HiTSKT  # noqa: E402

EXPERIMENT = ROOT / "experiments/hitskt/junyi/full"
CHECKPOINT = EXPERIMENT / "best_model.pt"
PROCESSED = ROOT / "data/processed/junyi/full"
AUDIT = ROOT / "reports/independent_correctness_audit"
RESULTS = ROOT / "reports/final_results.csv"
DEFAULT_OUTPUT = ROOT / "reports/junyi_prediction_samples_20261122"
CHECKPOINT_ID = "hitskt-junyi-full-epoch-032"
CSV_NAME = "junyi_hitskt_15_predictions.csv"
INPUTS_NAME = "junyi_hitskt_15_inputs.json"
SEED = 42
SAMPLE_COUNT = 15
THRESHOLD = 0.5
COLUMNS = (
    "sample_id", "dataset", "dataset_split", "model_name", "checkpoint_id",
    "student_id", "target_session_id", "target_position", "history_session_count",
    "history_question_ids", "history_skill_ids", "history_correctness",
    "target_question_id", "target_skill_id", "target_correct_input_at_position",
    "previous_target_correctness", "true_correctness", "raw_logit",
    "predicted_probability", "predicted_correctness", "prediction_matches_actual",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(directory: Path, expected: dict[str, str]) -> dict[str, str]:
    actual = {name: sha256(directory / name) for name in expected}
    for name, digest in actual.items():
        require(digest == expected[name], f"Historical SHA-256 mismatch: {directory / name}")
    return actual


def load_original() -> HiTSKT:
    model = HiTSKT.from_checkpoint(CHECKPOINT, map_location="cpu").eval()
    require(all(p.device.type == "cpu" for p in model.parameters()), "Expected CPU model")
    return model


def verified_experiment() -> tuple[HiTSKT, RollingSessionDataset, dict[int, int], dict]:
    evaluation_audit = AUDIT / "junyi_evaluation_manifest.json"
    data_audit = AUDIT / "junyi_data_manifest.json"
    audit = read_json(evaluation_audit)
    data = read_json(data_audit)
    experiment_hashes = verify_hashes(EXPERIMENT, audit["historical_artifact_sha256"])
    store_hashes = verify_hashes(PROCESSED / "session_store", audit["session_store_sha256"])
    # Leave the student crosswalk unread. The processed store IDs are sufficient.
    mapping_hashes = verify_hashes(ROOT, {
        entry["path"]: entry["sha256"] for entry in data["processed_file_checksums"]
        if "/mappings/" in entry["path"] and not entry["path"].endswith("/students.csv")
    })
    require(len(mapping_hashes) == 4, "Missing historical Junyi vocabulary mapping hashes")
    mapping_file = PROCESSED / "mappings/question_skills.csv"
    with mapping_file.open(encoding="utf-8", newline="") as stream:
        skill_by_question = {
            int(row["question_id"]): int(row["skill_id"]) for row in csv.DictReader(stream)
        }
    raw = ROOT / "data/raw/junyi/Junyi.csv"
    raw_hash = sha256(raw) if raw.is_file() else None
    if raw_hash is not None:
        require(raw_hash == data["raw_source_sha256"], "Raw Junyi source hash mismatch")

    with RESULTS.open(encoding="utf-8", newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["dataset"] == "junyi" and row["model"] == "HiTSKT"]
    require(len(matches) == 1, "Expected one Junyi HiTSKT benchmark row")
    row = matches[0]
    config = read_json(EXPERIMENT / "config.json")
    results = read_json(EXPERIMENT / "final_results.json")
    stats = read_json(EXPERIMENT / "dataset_stats.json")
    require(results["status"] == "complete" and results["best_checkpoint_reloaded"],
            "Missing completed benchmark with best checkpoint reload")
    require(row["experiment_directory"] == EXPERIMENT.relative_to(ROOT).as_posix()
            and float(row["test_auc"]) == results["test"]["roc_auc"] == 0.7978262055191263
            and int(row["best_epoch"]) == config["best_epoch"] == results["best_epoch"] == 32
            and float(row["best_validation_auc"]) == config["best_validation_auc"]
            == results["best_validation_auc"], "Benchmark identity disagrees across artifacts")

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    require(payload["format_version"] == 1 and "variant" not in payload
            and payload["epoch"] == results["best_epoch"]
            and payload["validation_roc_auc"] == results["best_validation_auc"],
            "Checkpoint is not the original validation-selected Junyi artifact")
    model = load_original()
    model_config = asdict(model.config)
    require(model_config == payload["config"]
            and all(config[k] == v for k, v in model_config.items() if k in config)
            and all(torch.equal(value, payload["state_dict"][name])
                    for name, value in model.state_dict().items()), "Saved weights/config mismatch")
    del payload

    store = SessionStore(PROCESSED / "session_store")
    dataset = RollingSessionDataset(store, "test", history_sessions=config["history_sessions"])
    require(len(dataset) == stats["target_sessions"]["test"]
            and int(np.diff(store.session_offsets)[dataset.targets].sum())
            == stats["target_counts"]["test"] == results["test"]["targets"]
            and model_config["num_questions"] == store.metadata["num_questions"]
            == len(skill_by_question)
            and model_config["num_skills"] == store.metadata["num_skills"],
            "Dataset split or vocabulary disagrees with the benchmark")
    provenance = {
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_format_version": 1,
        "checkpoint_sha256": experiment_hashes["best_model.pt"],
        "architecture": config["attention"],
        "model_config": model_config,
        "best_epoch": results["best_epoch"],
        "validation_selection_metric": config["early_stopping_metric"],
        "best_validation_auc": results["best_validation_auc"],
        "historical_full_test_auc_for_identification_only": results["test"]["roc_auc"],
        "historical_test_targets": results["test"]["targets"],
        "historical_test_sessions": len(dataset),
        "history_sessions": config["history_sessions"],
        "session_store_metadata": store.metadata,
        "raw_source_expected_sha256": data["raw_source_sha256"],
        "local_raw_source_verified_sha256": raw_hash,
        "experiment_sha256": experiment_hashes,
        "session_store_sha256": store_hashes,
        "mapping_sha256": mapping_hashes,
        "benchmark_csv_sha256": sha256(RESULTS),
        "provenance_references": [p.relative_to(ROOT).as_posix() for p in
                                  (evaluation_audit, data_audit, ROOT / "docs/data/junyi_provenance.md")],
        "inference_source_sha256": {
            p: sha256(ROOT / p) for p in ("ktbench/models/hitskt.py", "ktbench/models/performer.py",
                                        "ktbench/batching.py", "ktbench/data/session_store.py")
        },
        "runtime": {"device": "cpu", "threads": 1, "dtype": "float32",
                    "python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": np.__version__},
    }
    return model, dataset, skill_by_question, provenance


def prefix_only(batch: HiTSKTBatch, position: int) -> HiTSKTBatch:
    stop = position + 1
    question = batch.target_question[:, :stop].contiguous().clone()
    return replace(
        batch, target_question=question,
        target_skill=batch.target_skill[:, :stop].contiguous().clone(),
        target_correct_input=batch.target_correct_input[:, :stop].contiguous().clone(),
        target_attention_mask=batch.target_attention_mask[:, :stop].contiguous().clone(),
        target_labels=torch.full_like(question, CORRECT_PAD_ID),
        target_metric_mask=torch.zeros_like(question, dtype=torch.bool),
        padded_tokens=batch.history_question.numel() + question.numel(),
    )


@torch.inference_mode()
def score(model: HiTSKT, batch: HiTSKTBatch, position: int) -> tuple[float, float]:
    require(not model.training, "HiTSKT must be in evaluation mode")
    logit = model(batch)[0, position]
    return logit.item(), torch.sigmoid(logit).item()


def encode_batch(batch: HiTSKTBatch) -> dict:
    return {name: {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
            if isinstance(value, torch.Tensor) else value for name, value in vars(batch).items()}


def decode_batch(encoded: dict) -> HiTSKTBatch:
    dtypes = {str(dtype): dtype for dtype in (torch.long, torch.bool, torch.uint8)}
    values = {name: torch.tensor(item["values"], dtype=dtypes[item["dtype"]]).reshape(item["shape"])
              if isinstance(item, dict) else item for name, item in encoded.items()}
    return HiTSKTBatch(**values)


def equal_batches(left: HiTSKTBatch, right: HiTSKTBatch) -> bool:
    for field in fields(HiTSKTBatch):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, torch.Tensor):
            if not torch.equal(a, b):
                return False
        elif a != b:
            return False
    return True


def observed_mapping(session, skill_by_question: dict[int, int]) -> bool:
    return all(skill_by_question.get(int(q)) == int(skill)
               for q, skill in zip(session.question_ids, session.skill_ids, strict=True))


def sample_predictions(model, dataset, skill_by_question) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(SEED)
    indexes = rng.choice(len(dataset), SAMPLE_COUNT, replace=False)
    store = dataset.store
    rows, inputs = [], []
    def collate(item):
        return collate_hitskt([item], num_questions=model.config.num_questions,
                              num_skills=model.config.num_skills)

    for ordinal, index in enumerate(indexes, start=1):
        example = dataset[int(index)]
        target = example.target
        position = int(rng.integers(len(target)))
        session_id = int(dataset.targets[index])
        student_id = int(store.session_student[session_id])
        history_start = session_id - len(example.history)
        require(example.split == store.session_split[session_id] == SPLIT_NAMES["test"]
                and np.all(store.session_student[history_start:session_id] == student_id),
                f"Test or learner history mismatch in sample {ordinal}")
        require(all(observed_mapping(session, skill_by_question)
                    for session in (*example.history, target)), f"Vocabulary mismatch in sample {ordinal}")
        original = collate(example)
        require(bool(original.target_metric_mask[0, position]), "Selected position is masked")
        q, skill, actual = (int(value[position]) for value in
                            (target.question_ids, target.skill_ids, target.correct))
        previous = [int(value) for value in target.correct[:position]]
        batch = prefix_only(original, position)
        require(batch.target_correct_input[0].tolist() == [CORRECT_BOS_ID, *previous]
                and batch.target_question[0, position].item() == q
                and batch.target_skill[0, position].item() == skill
                and original.target_labels[0, position].item() == actual,
                f"Shifted correctness or target alignment mismatch in sample {ordinal}")
        logit, probability = score(model, batch, position)
        require(math.isfinite(logit) and 0 <= probability <= 1, "Invalid model output")
        # These mutations only test the preprocessing boundary. Neither the
        # selected answer nor any later answer is passed into the forward call.
        for affected in (slice(position, position + 1), slice(position + 1, None)):
            altered_correct = np.array(target.correct, copy=True)
            altered_correct[affected] = 1 - altered_correct[affected]
            altered = replace(example, target=replace(target, correct=altered_correct))
            altered_batch = collate(altered)
            if len(altered_correct[affected]):
                require(not torch.equal(original.target_correct_input, altered_batch.target_correct_input),
                        f"Response mutation did not reach the excluded suffix in sample {ordinal}")
            perturbed = prefix_only(altered_batch, position)
            require(equal_batches(batch, perturbed) and score(model, perturbed, position) == (logit, probability),
                    f"Excluded answer affected the model input or result in sample {ordinal}")
        metric_probe = replace(batch, target_labels=torch.full_like(batch.target_labels, -1),
                               target_metric_mask=torch.ones_like(batch.target_metric_mask))
        require(score(model, metric_probe, position) == (logit, probability),
                f"Evaluation-only fields affected sample {ordinal}")
        if position > 0:
            # Probe the model itself using later tokens within the safe prefix.
            changed_prefix = batch.target_correct_input.clone()
            changed_prefix[:, 1:] = 1 - changed_prefix[:, 1:]
            causal_probe = replace(batch, target_correct_input=changed_prefix)
            require(score(model, causal_probe, 0) == score(model, batch, 0),
                    f"A later correctness input affected the first logit in sample {ordinal}")
        prediction = int(probability >= THRESHOLD)
        rows.append(dict(zip(COLUMNS, (
            f"junyi-{ordinal:02d}", "junyi", "test", "HiTSKT", CHECKPOINT_ID,
            student_id, session_id, position + 1, len(example.history),
            [[int(v) for v in s.question_ids] for s in example.history],
            [[int(v) for v in s.skill_ids] for s in example.history],
            [[int(v) for v in s.correct] for s in example.history],
            q, skill, batch.target_correct_input[0, position].item(), previous, actual,
            logit, probability, prediction, int(prediction == actual),
        ), strict=True)))
        inputs.append({"sample_id": rows[-1]["sample_id"], "dataset_index_zero_based": int(index),
                       "student_id_processed": student_id, "target_session_store_index": session_id,
                       "student_session_number_one_based": int(store.session_number[session_id]),
                       "history_session_store_indexes": list(range(history_start, session_id)),
                       "original_target_session_length": len(target), "read_logit_at": [0, position],
                       "model_inputs": encode_batch(batch)})
    require(len({row["target_session_id"] for row in rows}) == SAMPLE_COUNT,
            "Expected 15 distinct test sessions and target positions")
    return rows, inputs


def replay(directory: Path) -> dict:
    provenance = read_json(directory / "checkpoint_provenance.json")
    require(sha256(CHECKPOINT) == provenance["checkpoint_sha256"], "Replay checkpoint hash mismatch")
    inputs = read_json(directory / INPUTS_NAME)["samples"]
    with (directory / CSV_NAME).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(inputs) == SAMPLE_COUNT, "Replay requires exactly 15 rows and inputs")
    model = load_original()
    for row, item in zip(rows, inputs, strict=True):
        batch = decode_batch(item["model_inputs"])
        _, position = item["read_logit_at"]
        require(row["sample_id"] == item["sample_id"]
                and int(row["target_position"]) == position + 1
                and int(row["target_question_id"]) == batch.target_question[0, position].item()
                and int(row["target_skill_id"]) == batch.target_skill[0, position].item()
                and batch.target_correct_input[0].tolist() ==
                [CORRECT_BOS_ID, *json.loads(row["previous_target_correctness"])],
                "Exported tensor and CSV alignment mismatch")
        logit, probability = score(model, batch, position)
        predicted = int(probability >= THRESHOLD)
        require(logit == float(row["raw_logit"]) and probability == float(row["predicted_probability"])
                and predicted == int(row["predicted_correctness"])
                and int(predicted == int(row["true_correctness"])) == int(row["prediction_matches_actual"]),
                f"Exact CSV replay mismatch: {row['sample_id']}")
    return {"status": "PASS_EXACT", "rows": len(rows), "maximum_logit_delta": 0.0,
            "maximum_probability_delta": 0.0}


def explanation(first: dict, item: dict, provenance: dict) -> str:
    tensors = item["model_inputs"]
    history = tensors["history_question"]["shape"]
    target = tensors["target_question"]["shape"]
    correct = tensors["target_correct_input"]["values"][0]
    target_questions = tensors["target_question"]["values"][0]
    target_skills = tensors["target_skill"]["values"][0]
    return f"""# Original HiTSKT predictions on 15 Junyi test questions

The epoch-32 **ELU+1 HiTSKT** checkpoint has SHA-256
`{provenance['checkpoint_sha256']}`. The published `0.7978262055191263`
is the historical aggregate test AUC and identifies the experiment. The export
contains 15 fresh FP32 CPU predictions. It does not recompute AUC or recover
the original per-question CUDA outputs.

## Selection and input format

NumPy `default_rng(42)` samples 15 rolling test-session indexes without
replacement using `choice`, then draws a uniform interaction position in
each selected session in draw order using `integers`. Selection does not
depend on labels or predictions. The 15 sessions have their original rolling
history of up to 15 complete earlier sessions. This is retrospective
teacher-forced context with recorded grades of earlier interactions.
Real-time availability of those grades at question presentation was not
established by the Junyi source.

`student_id` is the genuine **processed student ID** from `session_student.npy`.
`target_session_id` is the zero-based global session-store index.
`target_position` is one-based within the target session. Question and skill IDs
are the original model vocabulary indexes. CSV history cells are JSON arrays
of per-session genuine actions in chronological order, before EOS and padding.
The companion JSON contains the exact batch values, dtypes and shapes after
collation and exclusion of the target-session suffix. Its evaluation-only
tensors contain the correctness PAD sentinel and a disabled metric mask.
The current grade is read separately for comparison. Future grades are absent
from all forwarded tensors. Keep the row-level artifacts restricted because
they contain detailed learner trajectories and processed identifiers.

## Trace of {first['sample_id']}

1. `ActionEncoder` uses all {first['history_session_count']} earlier sessions.
   Question, skill, and past correctness IDs are shaped `{history}` with
   historical EOS and right padding. It adds embeddings and positions, applies
   masked causal attention, and takes the EOS vector from each session.
2. `SessionEncoder` packs the vectors using `history_owner`, `history_order`,
   and `history_count`. Its learned session EOS summarizes the learner's
   earlier sessions in a 128-dimensional vector.
3. `CorrectPaddingEncoder` adds the session summary to the shifted correctness
   embeddings. This prefix is `{correct}`. `4` is BOS. The recorded earlier
   responses are `{first['previous_target_correctness']}`.
4. `Decoder` embeds target question IDs `{target_questions}` and skill IDs
   `{target_skills}`. These `{target}` ID tensors query the aligned correctness
   memory through causal cross-attention. The decoder returns `[1, {target[1]}, 128]`
   hidden representations. Each is a vector, while the prediction head returns
   one scalar logit per position.
5. The linear head's logit at `[0, {item['read_logit_at'][1]}]` is
   `{first['raw_logit']!r}`. `torch.sigmoid` returned
   `{first['predicted_probability']!r}`. Using `>= 0.5` gives
   `{first['predicted_correctness']}`. The held-out recorded result was
   `{first['true_correctness']}`.

## Reproduction

From the repository root, regenerate the complete export into an unused path
or replay the saved tensors and verify the CSV in a new CPU process:

```sh
python scripts/export_junyi_hitskt_predictions.py --output-dir /path/to/new-export
python scripts/export_junyi_hitskt_predictions.py --replay reports/junyi_prediction_samples_20261122
```

The causal collation boundary was checked by separately flipping the excluded
current and future responses before collation and verifying that the forwarded
batch and prediction stayed identical. An independent forward with changed
evaluation-only fields also preserved the result. For each prefix with more
than one action, perturbing later correctness inputs preserved the first
logit. Fresh-process replay compares every CSV logit and probability exactly.

Source: [model](../../ktbench/models/hitskt.py),
[collator](../../ktbench/batching.py), [rolling loader](../../ktbench/data/session_store.py),
and [Junyi provenance](../../docs/data/junyi_provenance.md).
"""


def show_table(rows: list[dict]) -> None:
    print(f"{'Sample':<10} {'Question':>8} {'Skill':>5} {'Previous correctness':<62} {'Probability':>13} {'Pred':>4} {'Actual':>6}")
    for row in rows:
        print(f"{row['sample_id']:<10} {row['target_question_id']:>8} {row['target_skill_id']:>5} "
              f"{json.dumps(row['previous_target_correctness'], separators=(',', ':')):<62} "
              f"{row['predicted_probability']:>13.10f} {row['predicted_correctness']:>4} "
              f"{row['true_correctness']:>6}")


def export(directory: Path) -> None:
    require(not directory.exists(), f"Output already exists: {directory}")
    model, dataset, skills, provenance = verified_experiment()
    rows, inputs = sample_predictions(model, dataset, skills)
    del model
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / CSV_NAME).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, list) else v
                             for k, v in row.items()})
    write_json(directory / INPUTS_NAME, {"format_version": 1, "selection_seed": SEED,
                                         "selection_rng": "numpy.default_rng / PCG64", "samples": inputs})
    write_json(directory / "checkpoint_provenance.json", provenance)
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay", str(directory)],
                            check=True, capture_output=True, text=True)
    write_json(directory / "verification_results.json", {
        "status": "PASS", "rows": len(rows), "distinct_test_sessions": len(rows),
        "distinct_target_positions": len(rows), "real_vocabulary_and_target_alignment_checks": len(rows),
        "checkpoint_strictly_loaded_and_weights_equal": True,
        "model_forward_device": "cpu", "optimizer_updates": 0, "training_calls": 0,
        "full_test_inference_runs": 0, "threshold": THRESHOLD,
        "probabilities_finite_and_in_unit_interval": True,
        "shifted_correctness_prefix_checks": len(rows),
        "current_response_exclusion_checks": len(rows), "future_response_exclusion_checks": len(rows),
        "evaluation_fields_invariance_checks": len(rows),
        "model_causal_prefix_checks": sum(row["target_position"] > 1 for row in rows),
        "fresh_process_csv_replay": json.loads(result.stdout),
        "csv_sha256": sha256(directory / CSV_NAME), "input_json_sha256": sha256(directory / INPUTS_NAME),
    })
    with (directory / "prediction_explanation.md").open("x", encoding="utf-8") as stream:
        stream.write(explanation(rows[0], inputs[0], provenance))
    show_table(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    action.add_argument("--replay", type=Path, help="Read saved tensors and verify the existing CSV")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.replay is not None:
        print(json.dumps(replay(args.replay.resolve())))
    else:
        export(args.output_dir.resolve())


if __name__ == "__main__":
    main()
