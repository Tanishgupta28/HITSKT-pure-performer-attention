import numpy as np
import pandas as pd
import pytest

from ktbench.data.assist2009 import Assist2009Store, preprocess
from ktbench.data.fast_rolling import PackedRollingDataset


def test_deduplication_targets_and_past_only_histories(tmp_path):
    records = [(i, 11, 31, 101, 501, i % 2, 7) for i in range(1, 11)]
    records.extend([(6, 11, 31, 101, 501, 0, 2), (11, 22, 32, 201, 502, 0, None)])
    columns = ("order_id", "user_id", "problem_id", "assignment_id", "sequence_id",
               "correct", "skill_id")
    raw = pd.DataFrame(records[::-1], columns=columns)
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    path = tmp_path / "data.csv"
    raw.to_csv(path, index=False)
    output = tmp_path / "processed"
    summary = preprocess(path, output)
    assert (summary["raw_rows"], summary["interactions"], summary["exact_duplicate_rows"]) == (13, 11, 1)
    assert (summary["multi_skill_interactions"], summary["untagged_interactions"]) == (1, 1)
    assert summary["target_counts"] == {"train": 7, "validation": 2, "test": 2}
    assert not (output / "timestamp_ms.npy").exists()
    assert not (output / "session_offsets.npy").exists()

    store = Assist2009Store(output)
    for history in (2, 99, 199):
        test = store.targets("test", history)
        assert store.order_id[test.events].tolist() == [9, 10]
        batch = PackedRollingDataset(test)[[[0, 1]]].to_batches("cpu")[0]
        assert batch.target_event_id.tolist() == [8, 9]
        for index, target in enumerate((8, 9)):
            first = max(0, target - history)
            assert batch.history_correct[index, batch.history_mask[index]].tolist() == list(store.correct[first:target])
            assert batch.history_mask[index].sum().item() == target - first
    train = store.targets("train", 199)
    batch = train.vectorized_batch(np.array([0, len(train) - 1]))
    assert batch.student_id.tolist() == [1, 2]
    assert batch.target_skill[1] > 0
    assert not batch.history_mask.any()
    assert store.skill[5] != store.skill[4]

    raw.loc[raw.index[-1], "correct"] = 1
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        preprocess(path, tmp_path / "conflict")


def test_cuda_assist2009_resume_preserves_exact_dkt_results(tmp_path, monkeypatch):
    import torch
    import scripts.train_baseline as runner

    if not torch.cuda.is_available():
        pytest.skip("ASSIST2009 full-run path requires CUDA")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    records = [
        (100 * student + i, student, i, 1, 1, i % 2, i % 3 + 1)
        for student in range(1, 6) for i in range(1, 11)
    ]
    columns = ("order_id", "user_id", "problem_id", "assignment_id", "sequence_id",
               "correct", "skill_id")
    source = tmp_path / "source.csv"
    pd.DataFrame(records, columns=columns).to_csv(source, index=False)
    preprocess(source, tmp_path / "processed")
    store = Assist2009Store(tmp_path / "processed")
    training = dict(model_name="dkt", dataset_name="assist2009", store=store,
                    transport="packed64", epoch_ceiling_override=2)
    expected = runner.train_baseline(output_root=tmp_path / "continuous", **training)
    out = tmp_path / "interrupted"

    class StopAtBoundary(Exception):
        pass

    write_json = runner._write_json

    def interrupt(path, value):
        write_json(path, value)
        if path == out / "config.json" and value["epochs_completed"] == 1:
            raise StopAtBoundary

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_write_json", interrupt)
        with pytest.raises(StopAtBoundary):
            runner.train_baseline(output_root=out, **training)
    assert not (out / "_SUCCESS").exists()
    actual = runner.train_baseline(output_root=out, resume=True, **training)
    assert actual["test"] == expected["test"]
    assert actual["best_epoch"] == expected["best_epoch"]
    assert actual["resume_count"] == 1
