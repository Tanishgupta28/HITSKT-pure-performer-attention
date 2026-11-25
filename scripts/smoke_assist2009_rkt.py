"""Check the CUDA Phi-only RKT batch, relation math, and training path."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from ktbench.config import seed_everything
from ktbench.data.assist2009 import Assist2009Store
from ktbench.models import PaperFaithfulRKT, RKTConfig
from ktbench.rkt.data import OrderedRKTTargets
from ktbench.rkt.phi import CrossFittedPhiRepository
from ktbench.rkt.settings import RKT_BATCH_SIZE, RKT_LEARNING_RATE, RKT_WEIGHT_DECAY, RKT_GRADIENT_CLIP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("phi_root", type=Path)
    args = parser.parse_args()
    seed_everything()
    store = Assist2009Store(args.store)
    repo = CrossFittedPhiRepository(args.phi_root, store.metadata["num_questions"])
    assert repo.metadata["dataset_provenance"] == store.provenance
    data = OrderedRKTTargets(store, "train", repo)
    indices = np.linspace(0, len(data) - 1, RKT_BATCH_SIZE, dtype=np.int64)
    batch = data[indices].to("cuda")
    assert batch.delta_hours is None
    events = data.events[indices]
    for row, event in enumerate(events):
        start = max(int(store.student_offsets[store.student[event] - 1]), event - data.history_length)
        n = event - start
        assert batch.history_mask[row].sum().item() == n
        torch.testing.assert_close(batch.history_question[row, batch.history_mask[row]].cpu(),
                                   torch.tensor(store.question[start:event], dtype=torch.long))
        if n:
            expected = repo.lookup(int(store.student[event]), 0, np.full(n, store.question[event]),
                                   np.asarray(store.question[start:event]))
            np.testing.assert_array_equal(batch.phi[row, batch.history_mask[row]].cpu(), expected)
    config = RKTConfig(store.metadata["num_questions"], store.metadata["students"], relation_only=True)
    model = PaperFaithfulRKT(config).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=RKT_LEARNING_RATE, weight_decay=RKT_WEIGHT_DECAY)
    logits, details = model(batch, return_attention=True)
    valid = batch.history_mask.any(dim=1)
    expected = batch.phi[valid].masked_fill(~batch.history_mask[valid], -torch.inf).softmax(dim=1)
    torch.testing.assert_close(details["relation"][valid], expected)
    assert "temporal" not in details and model.rho is None
    loss = binary_cross_entropy_with_logits(logits, batch.target_labels.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), RKT_GRADIENT_CLIP)
    optimizer.step()
    assert torch.isfinite(loss)
    print(json.dumps({"smoke_complete": True, "model": "RKT (Phi-only)",
                      "gpu": torch.cuda.get_device_name(), "train_loss": float(loss.detach()),
                      "targets": len(events), "temporal_tensors": 0,
                      "max_cuda_bytes": torch.cuda.max_memory_allocated()}))


if __name__ == "__main__":
    main()
