"""Paper-faithful performance-only Phi-relation RKT variant."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ktbench.rkt.data import RKTBatch
from ktbench.rkt.phi import HISTORY_LENGTH


RKT_LABEL = "RKT paper-faithful performance-only Phi-relation variant with 5-fold student-level cross-fitting."


@dataclass(frozen=True)
class RKTConfig:
    num_questions: int
    num_students: int
    width: int = 64
    heads: int = 1
    history_length: int = HISTORY_LENGTH
    dropout: float = 0.1
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.width != 64 or self.heads != 1 or self.dropout != 0.1:
            raise ValueError("RKT paper-faithful configuration is width=64, heads=1, dropout=0.1")
        if self.history_length != 49:
            raise ValueError("RKT uses 49 prior interactions plus current target context")


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus for strictly positive initialization."""

    return value + torch.log(-torch.expm1(-value))


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask, -torch.inf)
    maximum = masked.max(dim=-1, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponent = torch.exp(masked - maximum) * mask.to(values.dtype)
    return exponent / exponent.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class PaperFaithfulRKT(nn.Module):
    """RKT attention with raw Phi and learned positive student decay scale."""

    def __init__(self, config: RKTConfig, initial_s_hours: np.ndarray | torch.Tensor) -> None:
        super().__init__()
        self.config = config
        initial = torch.as_tensor(initial_s_hours, dtype=torch.float32)
        if len(initial) != config.num_students + 1 or torch.any(initial <= config.epsilon):
            raise ValueError("initial S_u must contain one positive value per student index")
        self.question_embedding = nn.Embedding(config.num_questions + 1, config.width, padding_idx=0)
        # Paper P has the same 2d width as the correctness-gated interaction.
        self.position_embedding = nn.Embedding(config.history_length + 1, 2 * config.width, padding_idx=0)
        self.input_projection = nn.Linear(2 * config.width, config.width)
        self.query_projection = nn.Linear(config.width, config.width)
        self.key_projection = nn.Linear(config.width, config.width)
        self.value_projection = nn.Linear(config.width, config.width)
        self.attention_norm = nn.LayerNorm(config.width)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, config.width),
            nn.ReLU(),
            nn.Linear(config.width, config.width),
        )
        self.feedforward_norm = nn.LayerNorm(config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.width, 1)
        self.rho = nn.Embedding(config.num_students + 1, 1)
        rho_initial = _softplus_inverse((initial - config.epsilon).clamp_min(config.epsilon))
        with torch.no_grad():
            self.rho.weight.copy_(rho_initial[:, None])
        # sigmoid(0) = 0.5 exactly.
        self.eta = nn.Parameter(torch.zeros(()))
        self._initialize_paper_parameters()
        # Restore data-derived S_u after normal initialization of model tensors.
        with torch.no_grad():
            self.rho.weight.copy_(rho_initial[:, None])

    def _initialize_paper_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def memory_strength(self, student_id: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.rho(student_id).squeeze(-1)) + self.config.epsilon

    def fusion_lambda(self) -> torch.Tensor:
        return torch.sigmoid(self.eta)

    def set_relation_parameters_trainable(self, trainable: bool) -> None:
        """Learn S_u/lambda in training and explicitly freeze for val/test."""

        self.rho.weight.requires_grad_(trainable)
        self.eta.requires_grad_(trainable)

    def forward(
        self, batch: RKTBatch, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        item = self.question_embedding(batch.history_question)
        correct = batch.history_correct.unsqueeze(-1).to(item.dtype)
        interaction = torch.cat((item * correct, item * (1.0 - correct)), dim=-1)
        position_ids = batch.history_position * batch.history_mask.long()
        interaction = interaction + self.position_embedding(position_ids)
        interaction = F.relu(self.input_projection(interaction))

        query = self.query_projection(self.question_embedding(batch.target_question))
        key = self.key_projection(interaction)
        value = self.value_projection(interaction)
        scores = torch.einsum("bd,bld->bl", query, key) / self.config.width**0.5
        alpha = _masked_softmax(scores, batch.history_mask)

        strength = self.memory_strength(batch.student_id).unsqueeze(-1)
        temporal = torch.exp(-batch.delta_hours / strength)
        relation = _masked_softmax(batch.phi + temporal, batch.history_mask)
        fusion = self.fusion_lambda()
        beta = fusion * alpha + (1.0 - fusion) * relation
        attended = torch.einsum("bl,bld->bd", beta, value)

        hidden = self.attention_norm(query + self.dropout(attended))
        hidden = self.feedforward_norm(hidden + self.dropout(self.feedforward(hidden)))
        logits = self.output(hidden).squeeze(-1)
        if return_attention:
            return logits, {
                "alpha": alpha,
                "temporal": temporal * batch.history_mask,
                "relation": relation,
                "beta": beta,
                "memory_strength_hours": strength.squeeze(-1),
                "lambda": fusion,
            }
        return logits

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "label": RKT_LABEL,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "PaperFaithfulRKT":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("format_version") != 1 or payload.get("label") != RKT_LABEL:
            raise ValueError("unsupported RKT checkpoint")
        config = RKTConfig(**payload["config"])
        # State loading overwrites this constructor placeholder, including rho.
        placeholder = np.ones(config.num_students + 1, dtype=np.float32)
        model = cls(config, placeholder)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model
