"""Softmax FAVOR+ with Gaussian-marginal positive orthogonal features.

Independent implementation of Choromanski et al., arXiv:2009.14794, sections
2.3/2.4 and Appendix B. This is separate from the historical ELU+1 module.
The normalized prefix is represented by per-feature maxima, sums and weighted
values. Rescaling changes a numerical gauge, never the relative key weights.
No sequence-wide/future maximum is used, and no denominator epsilon changes
the kernel. Chunk checkpointing bounds retained outer-product activations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class FavorPlusConfig:
    features: int = 128
    projection_seed: int = 42
    chunk_size: int = 32
    accumulation_dtype: str = "float32"
    sampling: str = "gaussian_orthogonal_blocks_chi"
    redraw: str = "fixed"
    stabilization: str = "causal_per_feature_max_v1"

    def __post_init__(self) -> None:
        if self.features <= 0 or self.chunk_size <= 0:
            raise ValueError("features and chunk_size must be positive")
        if not 0 <= self.projection_seed < 2**63 - 1:
            raise ValueError("projection seed must be in [0, 2**63-1)")
        if self.accumulation_dtype not in ("float32", "float64"):
            raise ValueError("only float32/float64 accumulation is supported")
        if (self.sampling, self.redraw, self.stabilization) != (
            "gaussian_orthogonal_blocks_chi", "fixed", "causal_per_feature_max_v1"
        ):
            raise ValueError("unsupported FAVOR+ sampling/redraw/stabilization policy")


def orthogonal_gaussian_projection(heads: int, features: int, dimension: int,
                                   seed: int) -> torch.Tensor:
    """CPU float64 Haar-QR blocks times independent chi_d row radii.

    Rows are orthogonal *within each d-row block*. For m>d the independent
    blocks cannot be mutually orthogonal. Chi radii preserve N(0,I) marginals;
    constant sqrt(d) radii would instead estimate regularized softmax.
    Uses a local generator, leaving the model/dropout RNG stream untouched.
    """
    if min(heads, features, dimension) <= 0:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for _ in range(heads):
        blocks = []
        for start in range(0, features, dimension):
            q, r = torch.linalg.qr(torch.randn(dimension, dimension,
                                             generator=generator, dtype=torch.float64))
            signs = torch.where(r.diag() < 0, -1.0, 1.0)
            rows = (q * signs).T
            radii = torch.randn(dimension, dimension, generator=generator,
                                dtype=torch.float64).norm(dim=1)
            blocks.append((rows * radii[:, None])[:min(dimension, features-start)])
        result.append(torch.cat(blocks))
    return torch.stack(result)


def softmax_log_features(values: torch.Tensor, projection: torch.Tensor,
                         *, query_gauge: bool = False) -> torch.Tensor:
    """Log phi(x) for x=values/d**.25; phi=exp(Omega x-||x||²/2)/sqrt(m).

    query_gauge omits only query-wise constants, which cancel in normalized
    attention. In particular it avoids subtracting a large query norm.
    """
    x = values * values.shape[-1] ** -0.25
    projected = torch.einsum("bhld,hmd->bhlm", x, projection)
    if query_gauge:
        return projected
    return projected - 0.5 * x.square().sum(-1, keepdim=True) - 0.5 * math.log(projection.shape[1])


def _prefix_chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  valid: torch.Tensor, maximum: torch.Tensor, total: torch.Tensor,
                  weighted: torch.Tensor, seen: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = []
    for i in range(q.shape[2]):
        use = valid[:, None, i, None]
        # The maxima are a gauge only. Detaching them removes non-smooth max
        # derivatives that cancel algebraically from the normalized answer.
        candidate = torch.where(seen, torch.maximum(maximum, k[:, :, i]), k[:, :, i])
        updated = torch.where(use, candidate, maximum).detach()
        old_weight = torch.where(seen, (maximum-updated).clamp_max(0).exp(), 0.0)
        key_weight = torch.where(use, (k[:, :, i]-updated).clamp_max(0).exp(), 0.0)
        total = total * old_weight + key_weight
        weighted = weighted * old_weight.unsqueeze(-1) + key_weight.unsqueeze(-1) * v[:, :, i, None, :]
        maximum = updated
        seen = seen | use
        safe_total = torch.where(seen, total, 1.0)
        weights = torch.softmax(q[:, :, i] + maximum + safe_total.log(), dim=-1)
        attended = (weights.unsqueeze(-1) * (weighted / safe_total.unsqueeze(-1))).sum(-2)
        outputs.append(torch.where(seen, attended, 0.0))
    return torch.stack(outputs, dim=2), maximum, total, weighted, seen


class _RecomputedCausalScan(torch.autograd.Function):
    """Store only chunk boundaries; recreate a single chunk's graph in backward.

    Non-reentrant torch.utils.checkpoint also retains an autograd graph node
    for every token operation. A custom first-order VJP avoids that substantial
    O(L) Python/C++ metadata overhead without truncating state gradients.
    Higher-order derivatives are deliberately unsupported in this v1 kernel.
    """

    @staticmethod
    def forward(ctx, q, k, v, valid, chunk_size):
        b,h,length,m = q.shape
        maximum, total = q.new_zeros(b,h,m), q.new_zeros(b,h,m)
        weighted = q.new_zeros(b,h,m,v.shape[-1])
        seen = torch.zeros(b,1,1,device=q.device,dtype=torch.bool)
        boundaries, outputs = [], []
        for start in range(0,length,chunk_size):
            boundaries.extend((maximum,total,weighted,seen))
            out,maximum,total,weighted,seen = _prefix_chunk(
                q[:,:,start:start+chunk_size],k[:,:,start:start+chunk_size],
                v[:,:,start:start+chunk_size],valid[:,start:start+chunk_size],
                maximum,total,weighted,seen)
            outputs.append(out)
        ctx.save_for_backward(q,k,v,valid,*boundaries)
        ctx.chunk_size = chunk_size
        return torch.cat(outputs,dim=2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        q,k,v,valid,*boundaries = ctx.saved_tensors
        dq,dk,dv = torch.zeros_like(q),torch.zeros_like(k),torch.zeros_like(v)
        carry_total = torch.zeros_like(boundaries[1])
        carry_weighted = torch.zeros_like(boundaries[2])
        starts = range(0,q.shape[2],ctx.chunk_size)
        for n in reversed(range(len(starts))):
            start = starts[n]; stop = min(q.shape[2],start+ctx.chunk_size)
            maximum,total,weighted,seen = boundaries[n*4:n*4+4]
            with torch.enable_grad():
                qc = q[:,:,start:stop].detach().requires_grad_(True)
                kc = k[:,:,start:stop].detach().requires_grad_(True)
                vc = v[:,:,start:stop].detach().requires_grad_(True)
                z = total.detach().requires_grad_(True)
                s = weighted.detach().requires_grad_(True)
                out,_,end_total,end_weighted,_ = _prefix_chunk(
                    qc,kc,vc,valid[:,start:stop],maximum,z,s,seen)
                gradients = torch.autograd.grad((out,end_total,end_weighted),(qc,kc,vc,z,s),
                    grad_outputs=(grad_out[:,:,start:stop],carry_total,carry_weighted))
            dq[:,:,start:stop],dk[:,:,start:stop],dv[:,:,start:stop] = gradients[:3]
            carry_total,carry_weighted = gradients[3:]
        return dq,dk,dv,None,None


def causal_favor_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           projection: torch.Tensor, *, query_mask: torch.Tensor,
                           key_mask: torch.Tensor, chunk_size: int = 32,
                           accumulation_dtype: str = "float32") -> torch.Tensor:
    """Aligned causal attention on [B,H,L,D] tensors; diagonal included.

    Arithmetic O(BHLmDv); no L² or full [B,H,L,m,Dv] allocation. Retained
    states O(BH ceil(L/C) mDv), recomputation workspace O(BHCmDv), plus linear
    input/log-feature/output storage. First derivatives are supported with
    checkpoint recomputation; there is no truncated gradient at chunk edges.
    """
    if query.ndim != 4 or query.shape != key.shape or value.shape[:3] != query.shape[:3]:
        raise ValueError("expected aligned [B,H,L,D] queries/keys and [B,H,L,Dv] values")
    b, h, length, d = query.shape
    if length == 0 or projection.ndim != 3 or projection.shape[0] != h or projection.shape[2] != d:
        raise ValueError("empty sequence or projection shape mismatch")
    if query_mask.shape != (b, length) or key_mask.shape != (b, length):
        raise ValueError("mask shape mismatch")
    if query_mask.dtype != torch.bool or key_mask.dtype != torch.bool or chunk_size <= 0:
        raise ValueError("boolean masks and positive chunk_size required")
    if accumulation_dtype not in ("float32", "float64"):
        raise ValueError("unsupported accumulation dtype")
    dtype = torch.float64 if query.dtype == torch.float64 or accumulation_dtype == "float64" else torch.float32
    # Explicit precision island: AMP must not downcast random-feature matmuls.
    with torch.autocast(device_type=query.device.type, enabled=False):
        q_input = torch.where(query_mask[:, None, :, None], query, 0.0).to(dtype)
        k_input = torch.where(key_mask[:, None, :, None], key, 0.0).to(dtype)
        v = torch.where(key_mask[:, None, :, None], value, 0.0).to(dtype)
        omega = projection.to(device=query.device, dtype=dtype)
        q = softmax_log_features(q_input, omega, query_gauge=True)
        k = softmax_log_features(k_input, omega)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v)):
            answer = _RecomputedCausalScan.apply(q,k,v,key_mask,chunk_size)
        else:
            maximum = q.new_zeros(b,h,omega.shape[1])
            total = torch.zeros_like(maximum)
            weighted = q.new_zeros(b,h,omega.shape[1],v.shape[-1])
            seen = torch.zeros(b,1,1,device=query.device,dtype=torch.bool)
            pieces = []
            for start in range(0,length,chunk_size):
                stop = min(length,start+chunk_size)
                out,maximum,total,weighted,seen = _prefix_chunk(q[:,:,start:stop],k[:,:,start:stop],
                    v[:,:,start:stop],key_mask[:,start:stop],maximum,total,weighted,seen)
                pieces.append(out)
            answer = torch.cat(pieces,dim=2)
        return torch.where(query_mask[:, None, :, None], answer, 0.0).to(query.dtype)


class FavorPlusAttention(nn.Module):
    """Attention-only replacement, reusing every learned legacy projection."""

    def __init__(self, original: nn.Module, config: FavorPlusConfig, *, seed: int) -> None:
        super().__init__()
        self.width, self.heads, self.head_width = original.width, original.heads, original.head_width
        self.query, self.key, self.value = original.query, original.key, original.value
        self.output, self.dropout = original.output, original.dropout
        self.config = config
        self.register_buffer("projection", orthogonal_gaussian_projection(
            self.heads, config.features, self.head_width, seed).float())

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_width).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *,
                query_mask: torch.Tensor, key_mask: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if not causal:
            raise ValueError("hitskt_favor_plus_v1 implements aligned causal attention only")
        # Mask before linear layers too, so even nonfinite PAD values are inert.
        q = self._heads(self.query(torch.where(query_mask[..., None], query, 0.0)))
        k = self._heads(self.key(torch.where(key_mask[..., None], key, 0.0)))
        v = self._heads(self.value(torch.where(key_mask[..., None], value, 0.0)))
        attended = causal_favor_attention(q, k, v, self.projection,
            query_mask=query_mask, key_mask=key_mask, chunk_size=self.config.chunk_size,
            accumulation_dtype=self.config.accumulation_dtype)
        attended = attended.transpose(1, 2).contiguous().reshape(query.shape)
        return self.dropout(self.output(attended)) * query_mask[..., None].to(attended.dtype)
