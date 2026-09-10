"""Established dataset-specific HiTSKT experiment ceilings and hyperparameters."""

from __future__ import annotations

from typing import Any


HITSKT_TRAINING_CONFIGS: dict[str, dict[str, Any]] = {
    "assist2017": {
        "width": 256,
        "heads": 4,
        "feedforward_width": 2048,
        "learning_rate": 5e-5,
        "epoch_ceiling": 100,
    },
    "junyi": {
        "width": 128,
        "heads": 2,
        "feedforward_width": 1024,
        "learning_rate": 5e-5,
        "epoch_ceiling": 50,
    },
    "ednet_kt1": {
        "width": 128,
        "heads": 2,
        "feedforward_width": 1024,
        "learning_rate": 8e-5,
        "epoch_ceiling": 40,
    },
}

HITSKT_COMMON_CONFIG: dict[str, Any] = {
    "model": "HiTSKT",
    "optimizer": "Adam",
    "weight_decay": 0.0,
    "dropout": 0.1,
    "action_layers": 1,
    "session_layers": 1,
    "correct_layers": 1,
    "decoder_layers": 1,
    "history_sessions": 15,
    "context_sessions_including_target": 16,
    "token_budget": 32_768,
    "maximum_batch_size": 64,
    "bucket_size": 512,
}
