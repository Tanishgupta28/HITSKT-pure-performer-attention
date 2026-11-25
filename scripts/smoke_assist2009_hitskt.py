"""CUDA smoke test for assignment prefixes and zero-history HiTSKT batches."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ktbench.batching import Session, collate_hitskt
from ktbench.config import seed_everything
from ktbench.data.assist2009 import Assist2009Store
from ktbench.data.assist2009_sessions import AssignmentRunStore, AssignmentTargets
from ktbench.hitskt_registry import HITSKT_TRAINING_CONFIGS
from ktbench.metrics import masked_bce_loss
from ktbench.models import HiTSKT, HiTSKTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("assignment_root", type=Path)
    args = parser.parse_args()
    seed_everything()
    store = AssignmentRunStore.open(Assist2009Store(args.store), args.assignment_root)
    shape = {k: store.metadata[k] for k in ("num_questions", "num_skills")}
    cfg = HITSKT_TRAINING_CONFIGS["assist2009"]
    model = HiTSKT(HiTSKTConfig(**shape, width=cfg["width"], heads=cfg["heads"],
                                feedforward_width=cfg["feedforward_width"])).cuda()
    train = AssignmentTargets(store, "train")
    valid = AssignmentTargets(store, "validation")
    first = int(np.flatnonzero(store.session_number[train.targets] == 1)[0])
    prefix = int(np.flatnonzero((valid.metric_start > 0) & (valid.lengths > 1))[0])
    example = valid[prefix]
    empty = collate_hitskt([train[first]], **shape).to("cuda")
    batch = collate_hitskt([example], **shape).to("cuda")
    assert empty.history_count.item() == 0 and empty.history_question.numel() == 0
    assert not batch.target_metric_mask[0, :example.metric_start].any()
    assert batch.target_metric_mask.sum().item() == valid.lengths[prefix]
    expected = valid.events[np.isin(valid.events, np.arange(example.event_start,
                                                            example.event_start + len(example.target)))]
    np.testing.assert_array_equal(batch.target_event_ids[batch.target_metric_mask].cpu(), expected)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss = masked_bce_loss(model(empty), empty.target_labels, empty.target_metric_mask)
    loss.backward()
    optimizer.step()
    model.eval()
    flip = list(example.target.correct)
    flip[example.metric_start] = 1 - flip[example.metric_start]
    changed = replace(example, target=Session(example.target.question_ids, example.target.skill_ids, flip))
    with torch.no_grad():
        a = model(batch)
        b = model(collate_hitskt([changed], **shape).to("cuda"))
    torch.testing.assert_close(a[:, :example.metric_start + 1], b[:, :example.metric_start + 1], rtol=0, atol=0)
    assert torch.isfinite(loss) and torch.isfinite(a).all()
    print(json.dumps({"smoke_complete": True, "model": "HiTSKT", "gpu": torch.cuda.get_device_name(),
                      "zero_history_train_loss": float(loss.detach()), "prefix_targets": len(expected),
                      "current_response_causality_verified": True,
                      "max_cuda_bytes": torch.cuda.max_memory_allocated()}))


if __name__ == "__main__":
    main()
