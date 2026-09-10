"""Pure ELU+1 Performer layers with linear-memory causal attention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_positions(length: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build positional encodings at runtime, with no fixed sequence cap."""

    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if width > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class CausalLinearAttention(nn.Module):
    """Multi-head Performer attention using phi(x)=ELU(x)+1.

    This implementation never materializes an L-by-L attention matrix. Causal
    attention is calculated with prefix sums of K and outer(K, V).
    """

    def __init__(self, width: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("model width must be divisible by attention heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, self.head_width).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.shape != value.shape:
            raise ValueError("attention expects [batch, length, width] tensors")
        if query.shape[0] != key.shape[0] or query.shape[2] != self.width:
            raise ValueError("attention batch/width mismatch")
        if causal and query.shape[1] != key.shape[1]:
            raise ValueError("aligned causal cross-attention requires equal sequence lengths")

        q = F.elu(self._heads(self.query(query))) + 1.0
        k = F.elu(self._heads(self.key(key))) + 1.0
        v = self._heads(self.value(value))
        valid_keys = key_mask[:, None, :, None].to(dtype=k.dtype)
        k = k * valid_keys
        v = v * valid_keys

        if causal:
            # [B,H,L,D,D] is linear in L; no quadratic softmax score matrix.
            kv = torch.einsum("bhld,bhlv->bhldv", k, v).cumsum(dim=2)
            k_prefix = k.cumsum(dim=2)
            numerator = torch.einsum("bhld,bhldv->bhlv", q, kv)
            denominator = torch.einsum("bhld,bhld->bhl", q, k_prefix).clamp_min(1e-6)
        else:
            kv = torch.einsum("bhld,bhlv->bhdv", k, v)
            k_sum = k.sum(dim=2)
            numerator = torch.einsum("bhld,bhdv->bhlv", q, kv)
            denominator = torch.einsum("bhld,bhd->bhl", q, k_sum).clamp_min(1e-6)

        attended = numerator / denominator.unsqueeze(-1)
        attended = attended.transpose(1, 2).contiguous().view(query.shape)
        attended = self.output(attended)
        attended = self.dropout(attended)
        return attended * query_mask.unsqueeze(-1).to(dtype=attended.dtype)


class PerformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, feedforward_width: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalLinearAttention(width, heads, dropout)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_width, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(values)
        values = values + self.attention(
            normalized,
            normalized,
            normalized,
            query_mask=mask,
            key_mask=mask,
            causal=True,
        )
        values = values + self.feedforward(self.feedforward_norm(values))
        return values * mask.unsqueeze(-1).to(dtype=values.dtype)


class PerformerDecoderBlock(nn.Module):
    """Causal linear cross-attention decoder block."""

    def __init__(self, width: int, heads: int, feedforward_width: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = CausalLinearAttention(width, heads, dropout)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_width, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = query + self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            query_mask=query_mask,
            key_mask=memory_mask,
            causal=True,
        )
        query = query + self.feedforward(self.feedforward_norm(query))
        return query * query_mask.unsqueeze(-1).to(dtype=query.dtype)
