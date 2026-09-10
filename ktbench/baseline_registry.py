"""Central repository-derived configurations for the three legacy baselines."""

from __future__ import annotations

from typing import Any

from torch import nn

from ktbench.models import (
    DKT,
    DKVMN,
    SAKT,
    DKTConfig,
    DKVMNConfig,
    SAKTConfig,
)


BASELINE_TRAINING_CONFIGS: dict[str, dict[str, Any]] = {
    "dkt": {
        "model": "DKT",
        "context_length": 200,
        "history_length": 199,
        "batch_size": 20,
        "optimizer": "Adam",
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "epoch_ceiling": 200,
        "gradient_clip": None,
        "dropout": 0.1,
        "embedding_size": 64,
        "hidden_size": 64,
        "layers": 1,
        "effective_recurrent_dropout": 0.0,
    },
    "dkvmn": {
        "model": "DKVMN",
        "context_length": 200,
        "history_length": 199,
        "batch_size": 32,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "epoch_ceiling": 100,
        "gradient_clip": 50.0,
        "dropout": 0.0,
        "question_embedding_size": 50,
        "interaction_embedding_size": 100,
        "memory_size": 20,
        "key_state_size": 50,
        "value_state_size": 100,
        "final_size": 50,
    },
    "sakt": {
        "model": "SAKT",
        "context_length": 100,
        "history_length": 99,
        "batch_size": 10,
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "epoch_ceiling": 300,
        "gradient_clip": 10.0,
        "dropout": 0.2,
        "width": 200,
        "layers": 1,
        "heads": 5,
        "maximum_relative_position": 10,
    },
}


def make_baseline_model(
    name: str, *, num_questions: int, num_skills: int
) -> nn.Module:
    if name == "dkt":
        return DKT(DKTConfig(num_skills=num_skills))
    if name == "dkvmn":
        return DKVMN(DKVMNConfig(num_skills=num_skills))
    if name == "sakt":
        return SAKT(SAKTConfig(num_questions=num_questions, num_skills=num_skills))
    raise ValueError(f"unknown baseline: {name}")
