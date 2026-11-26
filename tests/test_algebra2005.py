"""Regressions for simultaneous Algebra transaction targets."""
import json

import numpy as np
import torch

from ktbench.batching import collate_hitskt
from ktbench.data.algebra2005 import AlgebraSessionTargets
from ktbench.data.fast_rolling import vectorized_batch
from ktbench.data.rolling_targets import RollingTargetDataset
from ktbench.data.session_store import SessionStore
from ktbench.rkt.data import RKTTargetDataset
from ktbench.rkt.phi import _student_pair_outcomes


def test_tied_outcomes_are_visible_only_after_the_clock_advances(tmp_path):
    arrays = {
        "question": [1, 2, 3, 4, 5, 6],
        "skill": [1] * 6,
        "correct": [1, 0, 1, 1, 0, 0],
        "timestamp_ms": [1000, 1000, 2000, 2000, 3000, 3000],
        "session_offsets": [0, 5, 6],
        "session_student": [1, 2],
        "session_number": [1, 1],
        "session_split": [0, 0],
        "student_offsets": [0, 1, 2],
    }
    for name, values in arrays.items():
        np.save(tmp_path / f"{name}.npy", np.asarray(values))
    (tmp_path / "_SUCCESS").touch()
    (tmp_path / "metadata.json").write_text(json.dumps({"strict_timestamp_history": True}))
    store = SessionStore(tmp_path)
    np.testing.assert_array_equal(store.history_stop(np.arange(6)), [0, 0, 2, 2, 4, 5])
    rolling = RollingTargetDataset(store, "train", history_length=2)
    packed = vectorized_batch(rolling, [1, 3, 4, 5])
    assert packed.history_question.tolist() == [[0, 0], [1, 2], [3, 4], [0, 0]]
    temporal = RKTTargetDataset(store, "train")
    assert temporal[3].history_timestamp_ms.tolist() == [1000, 1000]

    hierarchy = AlgebraSessionTargets(store, "train")
    batch = collate_hitskt(list(hierarchy[i] for i in range(len(hierarchy))), num_questions=6, num_skills=1)
    assert sorted(batch.target_event_ids[batch.target_metric_mask].tolist()) == list(range(6))
    assert batch.history_count.tolist() == [0] * len(hierarchy)
    target = torch.where(batch.target_event_ids.eq(3) & batch.target_metric_mask)
    assert batch.target_question[target[0].item(), :3].tolist() == [1, 2, 4]

    pair_codes = _student_pair_outcomes(store.question[:5], store.correct[:5], 6, 2,
                                        store.history_stop(np.arange(5)))
    pairs = {(int(code // 4 // 7), int(code // 4 % 7)) for code in pair_codes}
    assert pairs == {(3, 1), (3, 2), (4, 1), (4, 2), (5, 3), (5, 4)}
