"""Clean adapters of the repository DKT, DKVMN, and SAKT architectures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from ktbench.data.rolling_targets import RollingTargetBatch


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask, -torch.inf)
    maximum = masked.max(dim=-1, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponent = torch.exp(masked - maximum) * mask.to(values.dtype)
    return exponent / exponent.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass(frozen=True)
class DKTConfig:
    num_skills: int
    context_length: int = 200
    history_length: int = 199
    embedding_size: int = 64
    hidden_size: int = 64
    layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.context_length != 200 or self.history_length != 199:
            raise ValueError("repository DKT context is 199 prior interactions plus target")
        if (
            self.embedding_size,
            self.hidden_size,
            self.layers,
            self.dropout,
        ) != (64, 64, 1, 0.1):
            raise ValueError("repository DKT hyperparameters changed")


class DKT(nn.Module):
    """Repository DKT: interaction embedding, one-layer LSTM, skill output."""

    model_name = "DKT"

    def __init__(self, config: DKTConfig) -> None:
        super().__init__()
        self.config = config
        vocabulary = config.num_skills + 1
        self.interaction_embedding = nn.Embedding(
            vocabulary * 3, config.embedding_size, padding_idx=0
        )
        # The repository records dropout=0.1 but does not apply recurrent
        # dropout in its single-layer LSTM. Preserve that effective behavior.
        self.rnn = nn.LSTM(
            config.embedding_size,
            config.hidden_size,
            num_layers=config.layers,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, vocabulary)

    def forward(self, batch: RollingTargetBatch) -> torch.Tensor:
        vocabulary = self.config.num_skills + 1
        interaction_ids = batch.history_skill + batch.history_correct * vocabulary
        embedded = self.interaction_embedding(interaction_ids)
        lengths = batch.history_mask.sum(dim=1)
        summaries = embedded.new_zeros((len(lengths), self.config.hidden_size))
        nonempty = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if len(nonempty):
            sequences = [
                embedded[index, batch.history_mask[index]] for index in nonempty.tolist()
            ]
            right_padded = pad_sequence(sequences, batch_first=True)
            packed = pack_padded_sequence(
                right_padded,
                lengths[nonempty].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.rnn(packed)
            summaries = summaries.index_copy(0, nonempty, hidden[-1])
        all_skill_logits = self.output(summaries)
        return all_skill_logits.gather(1, batch.target_skill[:, None]).squeeze(1)

    def checkpoint(self) -> dict[str, Any]:
        return _checkpoint(self.model_name, self.config, self.state_dict())


@dataclass(frozen=True)
class DKVMNConfig:
    num_skills: int
    context_length: int = 200
    history_length: int = 199
    question_embedding_size: int = 50
    interaction_embedding_size: int = 100
    memory_size: int = 20
    key_state_size: int = 50
    value_state_size: int = 100
    final_size: int = 50
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.context_length != 200 or self.history_length != 199:
            raise ValueError("repository DKVMN context is 199 prior interactions plus target")
        actual = (
            self.question_embedding_size,
            self.interaction_embedding_size,
            self.memory_size,
            self.key_state_size,
            self.value_state_size,
            self.final_size,
            self.dropout,
        )
        if actual != (50, 100, 20, 50, 100, 50, 0.0):
            raise ValueError("repository DKVMN hyperparameters changed")


class DKVMN(nn.Module):
    """Repository dynamic key-value memory network with differentiable writes."""

    model_name = "DKVMN"

    def __init__(self, config: DKVMNConfig) -> None:
        super().__init__()
        self.config = config
        vocabulary = config.num_skills + 1
        self.skill_embedding = nn.Embedding(
            vocabulary, config.question_embedding_size, padding_idx=0
        )
        self.interaction_embedding = nn.Embedding(
            2 * vocabulary, config.interaction_embedding_size, padding_idx=0
        )
        self.memory_key = nn.Parameter(
            torch.empty(config.memory_size, config.key_state_size)
        )
        self.initial_memory_value = nn.Parameter(
            torch.empty(config.memory_size, config.value_state_size)
        )
        self.erase = nn.Linear(config.interaction_embedding_size, config.value_state_size)
        self.add = nn.Linear(config.interaction_embedding_size, config.value_state_size)
        self.read_projection = nn.Linear(
            config.value_state_size + config.question_embedding_size,
            config.final_size,
        )
        self.output = nn.Linear(config.final_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.memory_key)
        nn.init.kaiming_normal_(self.initial_memory_value)
        for layer in (self.erase, self.add, self.read_projection, self.output):
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _weights(self, skill_embedding: torch.Tensor) -> torch.Tensor:
        return torch.softmax(skill_embedding @ self.memory_key.T, dim=-1)

    def _write_terms(
        self, batch: RollingTargetBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the affine ``A`` and ``B`` terms for every memory write."""

        vocabulary = self.config.num_skills + 1
        skills = self.skill_embedding(batch.history_skill)
        interaction_ids = batch.history_skill + batch.history_correct * vocabulary
        interactions = self.interaction_embedding(interaction_ids)
        weight = self._weights(skills)
        erase = torch.sigmoid(self.erase(interactions))
        add = torch.tanh(self.add(interactions))
        a = 1.0 - weight.unsqueeze(-1) * erase.unsqueeze(2)
        b = weight.unsqueeze(-1) * add.unsqueeze(2)
        valid = batch.history_mask[:, :, None, None]
        return torch.where(valid, a, torch.ones_like(a)), torch.where(
            valid, b, torch.zeros_like(b)
        )

    def _final_memory_sequential(self, batch: RollingTargetBatch) -> torch.Tensor:
        """Reference recurrence retained for equivalence tests."""

        batch_size, history_width = batch.history_skill.shape
        memory = self.initial_memory_value.unsqueeze(0).expand(batch_size, -1, -1)
        a, b = self._write_terms(batch)
        for position in range(history_width):
            memory = a[:, position] * memory + b[:, position]
        return memory

    def _final_memory_balanced(self, batch: RollingTargetBatch) -> torch.Tensor:
        """Compose the identical affine writes with a logarithmic-depth scan."""

        batch_size, history_width = batch.history_skill.shape
        if history_width == 0:
            return self.initial_memory_value.unsqueeze(0).expand(batch_size, -1, -1)
        a, b = self._write_terms(batch)
        while a.shape[1] > 1:
            pair_count = a.shape[1] // 2
            stop = pair_count * 2
            early_a, late_a = a[:, :stop:2], a[:, 1:stop:2]
            early_b, late_b = b[:, :stop:2], b[:, 1:stop:2]
            combined_a = late_a * early_a
            combined_b = late_a * early_b + late_b
            if stop < a.shape[1]:
                combined_a = torch.cat((combined_a, a[:, -1:]), dim=1)
                combined_b = torch.cat((combined_b, b[:, -1:]), dim=1)
            a, b = combined_a, combined_b
        initial = self.initial_memory_value.unsqueeze(0).expand(batch_size, -1, -1)
        return a[:, 0] * initial + b[:, 0]

    def forward(self, batch: RollingTargetBatch) -> torch.Tensor:
        memory = self._final_memory_balanced(batch)

        target = self.skill_embedding(batch.target_skill)
        target_weight = self._weights(target)
        read = torch.sum(memory * target_weight.unsqueeze(-1), dim=1)
        hidden = torch.tanh(self.read_projection(torch.cat((read, target), dim=-1)))
        return self.output(hidden).squeeze(-1)

    def checkpoint(self) -> dict[str, Any]:
        return _checkpoint(self.model_name, self.config, self.state_dict())


@dataclass(frozen=True)
class SAKTConfig:
    num_questions: int
    num_skills: int
    context_length: int = 100
    history_length: int = 99
    width: int = 200
    layers: int = 1
    heads: int = 5
    maximum_relative_position: int = 10
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.context_length != 100 or self.history_length != 99:
            raise ValueError("repository SAKT context is 99 prior interactions plus target")
        actual = (
            self.width,
            self.layers,
            self.heads,
            self.maximum_relative_position,
            self.dropout,
        )
        if actual != (200, 1, 5, 10, 0.2):
            raise ValueError("repository SAKT hyperparameters changed")


class SAKT(nn.Module):
    """Repository SAKT item/skill self-attention reduced to one target query."""

    model_name = "SAKT"

    def __init__(self, config: SAKTConfig) -> None:
        super().__init__()
        self.config = config
        half = config.width // 2
        head_size = config.width // config.heads
        self.question_embedding = nn.Embedding(
            config.num_questions + 1, half, padding_idx=0
        )
        self.skill_embedding = nn.Embedding(
            config.num_skills + 1, half, padding_idx=0
        )
        self.input_projection = nn.Linear(2 * config.width, config.width)
        self.query_projection = nn.Linear(config.width, config.width)
        self.key_projection = nn.Linear(config.width, config.width)
        self.value_projection = nn.Linear(config.width, config.width)
        self.position_key = nn.Embedding(config.maximum_relative_position, head_size)
        self.position_value = nn.Embedding(config.maximum_relative_position, head_size)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.width, 1)

    def forward(
        self, batch: RollingTargetBatch, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        question = self.question_embedding(batch.history_question)
        skill = self.skill_embedding(batch.history_skill)
        item = torch.cat((question, skill), dim=-1)
        correct = batch.history_correct.unsqueeze(-1).to(item.dtype)
        interaction = torch.cat((item * correct, item * (1.0 - correct)), dim=-1)
        interaction = F.relu(self.input_projection(interaction))
        target = torch.cat(
            (
                self.question_embedding(batch.target_question),
                self.skill_embedding(batch.target_skill),
            ),
            dim=-1,
        )
        batch_size, width, _ = interaction.shape
        head_size = self.config.width // self.config.heads
        query = self.query_projection(target).view(batch_size, self.config.heads, head_size)
        key = self.key_projection(interaction).view(
            batch_size, width, self.config.heads, head_size
        ).transpose(1, 2)
        value = self.value_projection(interaction).view(
            batch_size, width, self.config.heads, head_size
        ).transpose(1, 2)

        ordinal = batch.history_mask.long().cumsum(dim=-1) - 1
        latest = batch.history_mask.sum(dim=-1, keepdim=True) - 1
        distance = (latest - ordinal).clamp(
            min=0, max=self.config.maximum_relative_position - 1
        )
        positional_key = self.position_key(distance).unsqueeze(1)
        positional_value = self.position_value(distance).unsqueeze(1)
        scores = torch.einsum("bhd,bhld->bhl", query, key + positional_key)
        scores = scores / head_size**0.5
        attention = _masked_softmax(scores, batch.history_mask.unsqueeze(1))
        attended = torch.sum(attention.unsqueeze(-1) * (value + positional_value), dim=2)
        logits = self.output(self.dropout(attended.reshape(batch_size, -1))).squeeze(-1)
        if return_attention:
            return logits, attention
        return logits

    def checkpoint(self) -> dict[str, Any]:
        return _checkpoint(self.model_name, self.config, self.state_dict())


def _checkpoint(model: str, config: object, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": model,
        "config": asdict(config),
        "state_dict": state_dict,
    }


def load_baseline_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> DKT | DKVMN | SAKT:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported baseline checkpoint")
    classes: dict[str, tuple[type, type[nn.Module]]] = {
        "DKT": (DKTConfig, DKT),
        "DKVMN": (DKVMNConfig, DKVMN),
        "SAKT": (SAKTConfig, SAKT),
    }
    if payload.get("model") not in classes:
        raise ValueError("unknown baseline checkpoint model")
    config_class, model_class = classes[payload["model"]]
    model = model_class(config_class(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model
