import numpy as np
import pandas as pd
import torch

from ktbench.batching import collate_hitskt, padded_token_cost
from ktbench.data.assist2009 import Assist2009Store, preprocess
from ktbench.data.assist2009_sessions import AssignmentRunStore, AssignmentTargets, prepare
from ktbench.data.session_store import DatasetShapeSequence
from ktbench.models import PaperFaithfulRKT, RKTConfig
from ktbench.rkt.data import OrderedRKTTargets
from ktbench.rkt.phi import CrossFittedPhiRepository, build_cross_fitted_phi_cache


def make_data(tmp_path):
    source = tmp_path / "source.csv"
    # Assignment 8 reopens after 9, which shares its sequence. The reopened
    # contiguous assignment-8 run crosses both held-out cutoffs.
    assignment = [8, 8, 9, 8, 8, 8, 8, 8, 8, 8]
    rows = []
    for student in range(1, 6):
        for i, block in enumerate(assignment):
            rows.append((100 * student + i, student, i + 1, block, 1, i % 2, 1, 1, i + 1, 1, 1))
    columns = ["order_id", "user_id", "problem_id", "assignment_id", "sequence_id",
               "correct", "skill_id", "base_sequence_id", "assistment_id",
               "student_class_id", "position"]
    pd.DataFrame(rows[::-1], columns=columns).to_csv(source, index=False)
    root = tmp_path / "ordered"
    preprocess(source, root)
    return source, Assist2009Store(root)


def test_proxy_sessions_keep_reopened_assignment_separate_and_targets_identical(tmp_path):
    source, flat = make_data(tmp_path)
    root = tmp_path / "assignment"
    audit = prepare(source, flat, root)
    store = AssignmentRunStore.open(flat, root)
    assert audit["alternatives"]["assignment"]["contiguous_runs"]["reopened_runs"] == 5
    assert np.array_equal(store.session_offsets[:4], [0, 2, 3, 10])
    train = AssignmentTargets(store, "train")
    validation = AssignmentTargets(store, "validation")
    test = AssignmentTargets(store, "test")
    assert train[0].history == []
    assert train[2].event_start == validation[0].event_start == test[0].event_start == 3
    assert [len(split[pos].target) for split, pos in [(train, 2), (validation, 0), (test, 0)]] == [3, 5, 7]
    assert [split[pos].metric_start for split, pos in [(train, 2), (validation, 0), (test, 0)]] == [0, 3, 5]
    assert len(test[0].history) == 2
    for data in (train, validation, test):
        examples = [data[i] for i in range(len(data))]
        batch = collate_hitskt(examples, num_questions=10, num_skills=1)
        np.testing.assert_array_equal(batch.target_event_ids[batch.target_metric_mask], data.events)
        assert batch.padded_tokens == padded_token_cost(data.shapes)
        assert DatasetShapeSequence(data).sort_lengths().tolist() == [shape.sort_length for shape in data.shapes]


def test_phi_only_batches_and_checkpoint_use_shared_split_without_time(tmp_path):
    _, flat = make_data(tmp_path)
    root = tmp_path / "phi"
    metadata = build_cross_fitted_phi_cache(flat, root)
    assert metadata["all_dataset_targets"] == {"train": 30, "validation": 10, "test": 10}
    assert metadata["dataset_provenance"] == flat.provenance
    repo = CrossFittedPhiRepository(root, 10)
    data = OrderedRKTTargets(flat, "validation", repo)
    batch = data[list(range(len(data)))]
    assert batch.delta_hours is None
    np.testing.assert_array_equal(batch.target_event_id, data.events)
    assert batch.history_mask[0].sum().item() == 6
    model = PaperFaithfulRKT(RKTConfig(10, 5, relation_only=True)).eval()
    with torch.no_grad():
        prediction, details = model(batch, return_attention=True)
    assert model.rho is None
    expected = batch.phi.masked_fill(~batch.history_mask, -torch.inf).softmax(dim=1)
    torch.testing.assert_close(details["relation"], expected)
    path = tmp_path / "rkt.pt"
    torch.save(model.checkpoint(), path)
    loaded = PaperFaithfulRKT.from_checkpoint(path).eval()
    assert loaded.config.relation_only
    with torch.no_grad():
        torch.testing.assert_close(loaded(batch), prediction, rtol=0, atol=0)
