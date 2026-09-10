import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.session_store import SessionStore, build_session_store
from ktbench.metrics import binary_metrics
from ktbench.models import PaperFaithfulRKT, RKTConfig
from ktbench.rkt.data import (
    MILLISECONDS_PER_HOUR,
    RKTTargetDataset,
    build_memory_strength_initialization,
    collate_rkt,
)
from ktbench.rkt.phi import (
    CrossFittedPhiRepository,
    SparsePhiMatrix,
    _student_pair_outcomes,
    assign_student_folds,
    build_cross_fitted_phi_cache,
)


def _make_store(root: Path, *, mutate_heldout: bool = False) -> SessionStore:
    rows = []
    event = 0
    # Each student has five sessions; first three are train, then val/test.
    # Student 10 has no positive training gap and therefore exercises fallback.
    for student in range(1, 11):
        for session in range(1, 6):
            split = 0 if session <= 3 else session - 3
            length = 60 if student == 1 and session == 1 else 2
            for position in range(1, length + 1):
                timestamp = 1_600_000_000_000 + session * 40_000_000 + position * 3_600_000
                if student == 10 and split == 0:
                    timestamp = 1_600_000_000_000
                label = (student + session + position) % 2
                if mutate_heldout and split > 0:
                    label = 1 - label
                rows.append(
                    {
                        "student_id": student,
                        "question_id": (position + session) % 5 + 1,
                        "skill_id": (position + session) % 3 + 1,
                        "correct": label,
                        "timestamp": timestamp,
                        "session_id": session,
                        "session_position": position,
                        "split": split,
                    }
                )
                event += 1
    processed = root / "processed"
    (processed / "events").mkdir(parents=True)
    (processed / "reports").mkdir()
    frame = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(frame), processed / "events" / "part-00000.parquet")
    (processed / "reports" / "summary.json").write_text(
        json.dumps(
            {
                "retained_interactions": len(frame),
                "retained_sessions": 50,
                "retained_students": 10,
            }
        )
    )
    (processed / "_SUCCESS").write_text("complete\n")
    build_session_store(processed, root / "store")
    return SessionStore(root / "store")


def test_timestamp_hours_memory_initialization_and_fallback(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "data")
    report = build_memory_strength_initialization(store, tmp_path / "memory")
    values = np.load(tmp_path / "memory" / "initial_s_hours.npy")

    assert MILLISECONDS_PER_HOUR == 3_600_000.0
    assert report["validation_or_test_used"] is False
    assert report["students_using_fallback"] == 1
    assert values[1] > 0
    assert values[10] == report["global_training_only_fallback_hours"]


def test_fold_assignment_is_seeded_balanced_and_persisted(tmp_path: Path) -> None:
    first = assign_student_folds(range(1, 13), seed=PROJECT_SEED)
    second = assign_student_folds(reversed(range(1, 13)), seed=PROJECT_SEED)
    assert first.equals(second)
    counts = first.groupby("fold").size()
    assert counts.max() - counts.min() <= 1

    store = _make_store(tmp_path / "data")
    metadata = build_cross_fitted_phi_cache(store, tmp_path / "phi", chunk_values=20)
    persisted = pd.read_csv(tmp_path / "phi" / "student_folds.csv")
    expected = assign_student_folds(range(1, 11), seed=PROJECT_SEED)
    assert np.array_equal(persisted.to_numpy(), expected.to_numpy())
    assert metadata["seed"] == 42
    assert metadata["folds_balanced_within_one"] is True


def test_latest_prior_occurrence_and_undefined_phi_zero() -> None:
    # At target q=4, the q=1 response at index 0 is superseded by index 2.
    questions = np.array([1, 2, 1, 4])
    correct = np.array([0, 1, 1, 0], dtype=np.uint8)
    codes = _student_pair_outcomes(questions, correct, vocabulary_size=5, history_length=49)
    pair_for_4_1 = (4 * 6 + 1) * 4
    outcomes = codes[(codes // 4) == (pair_for_4_1 // 4)] % 4
    assert outcomes.tolist() == [2]  # latest prior correct=1, target correct=0

    matrix = SparsePhiMatrix(
        keys=np.array([4 * 6 + 1]),
        counts=np.array([[1, 0, 0, 0]], dtype=np.int64),
        vocabulary_size=5,
    )
    # A one-cell contingency is undefined, and the second pair is unobserved.
    result = matrix.lookup(np.array([4, 5]), np.array([1, 2]))
    assert result.tolist() == [0.0, 0.0]


def test_cross_fitting_excludes_own_fold_and_heldout_data(tmp_path: Path) -> None:
    original = _make_store(tmp_path / "original")
    changed = _make_store(tmp_path / "changed", mutate_heldout=True)
    build_cross_fitted_phi_cache(original, tmp_path / "phi_original", chunk_values=20)
    build_cross_fitted_phi_cache(changed, tmp_path / "phi_changed", chunk_values=20)

    # Changing every validation/test label cannot alter any train-only cache.
    for filename in ["phi_all_train.npz"] + [
        f"phi_excluding_fold_{fold}.npz" for fold in range(5)
    ]:
        left = np.load(tmp_path / "phi_original" / filename)
        right = np.load(tmp_path / "phi_changed" / filename)
        assert np.array_equal(left["keys"], right["keys"])
        assert np.array_equal(left["counts"], right["counts"])

    repository = CrossFittedPhiRepository(tmp_path / "phi_original", 5)
    student = 1
    fold = int(repository.fold_by_student[student])
    own = np.load(tmp_path / "phi_original" / f"counts_fold_{fold}.npz")
    all_train = np.load(tmp_path / "phi_original" / "phi_all_train.npz")
    excluded = repository.excluding[fold]
    # Complement cache is exactly global training counts minus the target
    # student's entire fold, including their target labels and future rows.
    union = np.union1d(all_train["keys"], own["keys"])
    global_counts = np.zeros((len(union), 4), dtype=np.int64)
    own_counts = np.zeros_like(global_counts)
    excluded_counts = np.zeros_like(global_counts)
    global_counts[np.searchsorted(union, all_train["keys"])] = all_train["counts"]
    own_counts[np.searchsorted(union, own["keys"])] = own["counts"]
    excluded_counts[np.searchsorted(union, excluded.keys)] = excluded.counts
    assert np.array_equal(excluded_counts, global_counts - own_counts)


def test_rolling_history_target_uniqueness_shapes_and_causality(tmp_path: Path) -> None:
    store = _make_store(tmp_path / "data")
    build_cross_fitted_phi_cache(store, tmp_path / "phi", chunk_values=20)
    repository = CrossFittedPhiRepository(tmp_path / "phi", 5)
    datasets = [RKTTargetDataset(store, split) for split in ("train", "validation", "test")]
    target_ids = [example.target_event_id for dataset in datasets for example in dataset]
    assert len(target_ids) == len(set(target_ids)) == store.metadata["interactions"]

    long_target = datasets[0][59]
    assert len(long_target.history_question) == 49
    assert all(value <= long_target.target_timestamp_ms for value in long_target.history_timestamp_ms)
    batch = collate_rkt([datasets[0][0], long_target], phi_repository=repository)
    assert batch.history_question.shape == (2, 49)
    assert batch.history_mask.sum(dim=1).tolist() == [0, 49]
    assert batch.target_labels.shape == (2,)
    assert batch.delta_hours[1, -1].item() == 1.0

    # The target label is never consulted while making historical features.
    changed_target = type(long_target)(
        **{**vars(long_target), "target_correct": 1 - long_target.target_correct}
    )
    changed = collate_rkt([changed_target], phi_repository=repository)
    original = collate_rkt([long_target], phi_repository=repository)
    assert torch.equal(changed.history_question, original.history_question)
    assert torch.equal(changed.delta_hours, original.delta_hours)
    assert torch.equal(changed.phi, original.phi)


def test_s_u_lambda_temporal_learning_freeze_metrics_and_checkpoint(tmp_path: Path) -> None:
    seed_everything()
    store = _make_store(tmp_path / "data")
    build_cross_fitted_phi_cache(store, tmp_path / "phi", chunk_values=20)
    build_memory_strength_initialization(store, tmp_path / "memory")
    repository = CrossFittedPhiRepository(tmp_path / "phi", 5)
    initial = np.load(tmp_path / "memory" / "initial_s_hours.npy")
    model = PaperFaithfulRKT(RKTConfig(5, 10), initial).train()
    assert torch.all(model.memory_strength(torch.arange(1, 11)) > 0)
    assert model.fusion_lambda().item() == 0.5

    dataset = RKTTargetDataset(store, "train")
    batch = collate_rkt([dataset[index] for index in range(1, 8)], phi_repository=repository)
    logits, details = model(batch, return_attention=True)
    assert logits.shape == (7,)
    expected = torch.exp(-batch.delta_hours / details["memory_strength_hours"].unsqueeze(-1))
    assert torch.allclose(details["temporal"], expected * batch.history_mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.target_labels.float()
    )
    before_rho = model.rho.weight.detach().clone()
    before_eta = model.eta.detach().clone()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before_rho, model.rho.weight)
    assert not torch.equal(before_eta, model.eta)

    model.set_relation_parameters_trainable(False)
    frozen_rho = model.rho.weight.detach().clone()
    frozen_eta = model.eta.detach().clone()
    assert not model.rho.weight.requires_grad and not model.eta.requires_grad
    with torch.no_grad():
        evaluation_logits = model(batch)
    assert torch.equal(frozen_rho, model.rho.weight)
    assert torch.equal(frozen_eta, model.eta)

    metrics = binary_metrics(
        torch.sigmoid(evaluation_logits), batch.target_labels, loss=float(loss.detach())
    )
    assert set(metrics) == {
        "targets", "loss", "roc_auc", "accuracy", "precision", "recall", "f1", "mse", "threshold"
    }
    checkpoint = tmp_path / "rkt.pt"
    torch.save(model.checkpoint(), checkpoint)
    model.eval()
    restored = PaperFaithfulRKT.from_checkpoint(checkpoint).eval()
    restored.set_relation_parameters_trainable(False)
    with torch.no_grad():
        assert torch.equal(model(batch), restored(batch))
