"""Variable-length hierarchical HiTSKT with pure Performer attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ktbench.batching import HiTSKTBatch
from ktbench.models.performer import PerformerBlock, PerformerDecoderBlock, sinusoidal_positions


@dataclass(frozen=True)
class HiTSKTConfig:
    num_questions: int
    num_skills: int
    width: int = 128
    heads: int = 2
    feedforward_width: int = 256
    action_layers: int = 1
    session_layers: int = 1
    correct_layers: int = 1
    decoder_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_questions <= 0 or self.num_skills <= 0:
            raise ValueError("question and skill vocabularies must be positive")
        if self.width <= 0 or self.width % self.heads:
            raise ValueError("width must be positive and divisible by heads")
        if min(self.action_layers, self.session_layers, self.correct_layers, self.decoder_layers) <= 0:
            raise ValueError("each research architecture stage needs at least one layer")


def _stack(width: int, heads: int, ff: int, dropout: float, count: int) -> nn.ModuleList:
    return nn.ModuleList(PerformerBlock(width, heads, ff, dropout) for _ in range(count))


class ActionEncoder(nn.Module):
    def __init__(self, config: HiTSKTConfig) -> None:
        super().__init__()
        self.question = nn.Embedding(config.num_questions + 2, config.width, padding_idx=0)
        self.skill = nn.Embedding(config.num_skills + 2, config.width, padding_idx=0)
        self.correct = nn.Embedding(5, config.width, padding_idx=2)
        self.input_norm = nn.LayerNorm(config.width)
        self.layers = _stack(
            config.width, config.heads, config.feedforward_width, config.dropout, config.action_layers
        )

    def forward(self, batch: HiTSKTBatch) -> torch.Tensor:
        values = (
            self.question(batch.history_question)
            + self.skill(batch.history_skill)
            + self.correct(batch.history_correct)
        )
        values = self.input_norm(
            values + sinusoidal_positions(values.shape[1], values.shape[2], values.device, values.dtype)
        )
        for layer in self.layers:
            values = layer(values, batch.history_mask)
        eos_index = batch.history_mask.long().sum(dim=1) - 1
        return values[torch.arange(values.shape[0], device=values.device), eos_index]


class SessionEncoder(nn.Module):
    def __init__(self, config: HiTSKTConfig) -> None:
        super().__init__()
        self.session_eos = nn.Parameter(torch.empty(config.width))
        nn.init.normal_(self.session_eos, std=0.02)
        self.layers = _stack(
            config.width, config.heads, config.feedforward_width, config.dropout, config.session_layers
        )

    def forward(self, session_vectors: torch.Tensor, batch: HiTSKTBatch) -> torch.Tensor:
        batch_size = batch.target_question.shape[0]
        maximum = int(batch.history_count.max().item()) + 1
        packed = session_vectors.new_zeros(batch_size, maximum, session_vectors.shape[-1])
        packed[batch.history_owner, batch.history_order] = session_vectors
        eos_positions = batch.history_count
        packed[torch.arange(batch_size, device=packed.device), eos_positions] = self.session_eos
        positions = torch.arange(maximum, device=packed.device).unsqueeze(0)
        mask = positions <= eos_positions.unsqueeze(1)
        packed = packed + sinusoidal_positions(maximum, packed.shape[-1], packed.device, packed.dtype)
        for layer in self.layers:
            packed = layer(packed, mask)
        return packed[torch.arange(batch_size, device=packed.device), eos_positions]


class CorrectPaddingEncoder(nn.Module):
    def __init__(self, config: HiTSKTConfig) -> None:
        super().__init__()
        self.correct = nn.Embedding(5, config.width, padding_idx=2)
        self.layers = _stack(
            config.width, config.heads, config.feedforward_width, config.dropout, config.correct_layers
        )

    def forward(self, history: torch.Tensor, batch: HiTSKTBatch) -> torch.Tensor:
        values = history.unsqueeze(1) + self.correct(batch.target_correct_input)
        values = values + sinusoidal_positions(values.shape[1], values.shape[2], values.device, values.dtype)
        for layer in self.layers:
            values = layer(values, batch.target_attention_mask)
        return values


class Decoder(nn.Module):
    def __init__(self, config: HiTSKTConfig) -> None:
        super().__init__()
        self.question = nn.Embedding(config.num_questions + 2, config.width, padding_idx=0)
        self.skill = nn.Embedding(config.num_skills + 2, config.width, padding_idx=0)
        self.layers = nn.ModuleList(
            PerformerDecoderBlock(
                config.width, config.heads, config.feedforward_width, config.dropout
            )
            for _ in range(config.decoder_layers)
        )

    def forward(self, memory: torch.Tensor, batch: HiTSKTBatch) -> torch.Tensor:
        values = self.question(batch.target_question) + self.skill(batch.target_skill)
        values = values + sinusoidal_positions(values.shape[1], values.shape[2], values.device, values.dtype)
        for layer in self.layers:
            values = layer(values, memory, batch.target_attention_mask, batch.target_attention_mask)
        return values


class HiTSKT(nn.Module):
    """Action Encoder -> Session Encoder -> Correct/Padding Encoder -> Decoder."""

    def __init__(self, config: HiTSKTConfig) -> None:
        super().__init__()
        self.config = config
        self.action_encoder = ActionEncoder(config)
        self.session_encoder = SessionEncoder(config)
        self.correct_padding_encoder = CorrectPaddingEncoder(config)
        self.decoder = Decoder(config)
        self.prediction = nn.Linear(config.width, 1)

    def forward(self, batch: HiTSKTBatch) -> torch.Tensor:
        actions = self.action_encoder(batch)
        history = self.session_encoder(actions, batch)
        correct_memory = self.correct_padding_encoder(history, batch)
        decoded = self.decoder(correct_memory, batch)
        return self.prediction(decoded).squeeze(-1)

    def checkpoint(self) -> dict[str, Any]:
        return {"format_version": 1, "config": asdict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "HiTSKT":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported HiTSKT checkpoint format")
        model = cls(HiTSKTConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model
