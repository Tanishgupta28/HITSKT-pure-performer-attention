"""Bounded research diagnostics; never change attention or training gradients."""
from __future__ import annotations

import math
import numpy as np
import torch

from ktbench.models.favor_plus import FavorPlusAttention, causal_favor_attention
from scripts.favor_plus_quality_common import describe, independent_attention, output_errors

PREFIX_LIMIT = 32
FEATURE_KERNEL_ATOL = 5e-5  # inherited CUDA output gate, unchanged
# Prospective review triggers, not acceptance tolerances or full-training approval.
SOFTMAX_REVIEW_RELATIVE_L2 = .20
FEATURE_ESS_REVIEW_MEDIAN = 2.
NORM_GROWTH_REVIEW_RATIO = 2.


def prefix_diagnostics(q, k, v, projection, query_mask, key_mask, actual) -> dict:
    """Independent float64 kernel/softmax comparison on an already bounded prefix.

    actual is production raw attention before output projection/dropout. Results
    are measurements only when the caller actually supplies computed arrays.
    """
    q, k, v, projection, actual = [np.asarray(t) for t in (q, k, v, projection, actual)]
    qm, km = np.asarray(query_mask, bool), np.asarray(key_mask, bool)
    if (q.ndim != 4 or q.shape != k.shape or v.shape[:3] != q.shape[:3]
            or actual.shape != v.shape or q.shape[0] != 1 or not 1 <= q.shape[2] <= PREFIX_LIMIT
            or qm.shape != (1, q.shape[2]) or km.shape != qm.shape
            or projection.ndim != 3 or projection.shape[0] != q.shape[1]
            or projection.shape[2] != q.shape[3] or not qm.any() or not km.any()):
        raise ValueError("diagnostic requires one nonempty bounded aligned prefix")
    q = np.where(qm[:, None, :, None], q, 0.)
    k = np.where(km[:, None, :, None], k, 0.)
    v = np.where(km[:, None, :, None], v, 0.)
    if not all(np.isfinite(a).all() for a in (q, k, v, projection, actual)):
        raise FloatingPointError("nonfinite active diagnostic input or attention output")
    feature, exact, stats = independent_attention(q, k, v, projection, qm, km)
    implementation = output_errors(actual, feature, qm)
    approximation = output_errors(feature, exact, qm)
    observed = output_errors(actual, exact, qm)
    if not math.isfinite(implementation["max_abs"]) or implementation["max_abs"] > FEATURE_KERNEL_ATOL:
        raise FloatingPointError("production output fails independent feature-kernel tolerance")
    flags = []
    if approximation["relative_l2"] > SOFTMAX_REVIEW_RELATIVE_L2:
        flags.append("softmax_relative_l2_above_review_trigger")
    if stats["feature_ess"]["median"] < FEATURE_ESS_REVIEW_MEDIAN:
        flags.append("concentrated_features")
    return {"feature_kernel_agreement": "PASS", "implementation_error": implementation,
            "feature_versus_exact_softmax": approximation, "production_versus_exact_softmax": observed,
            "kernel_statistics": stats, "review_flags": flags,
            "prefix_positions_including_eos": q.shape[2], "heads": q.shape[1],
            "features": projection.shape[1], "oracle": "independent NumPy/SciPy float64"}


def fixed_example_diagnostics(model, batch, *, baseline: dict | None = None,
                              save_prefix=None) -> dict:
    """Same train example at initialization/after updates, eval mode, no RNG draws.

    Norm summaries cover all valid tokens/heads of that example. Exact attention
    is restricted to one row and <=32 positions per stage. The fixed input is
    not a fixed QKV tensor: QKV changes as weights learn. No hooks remain during
    optimization. save_prefix, if supplied, persists small CPU arrays locally.
    """
    entries, handles = {}, []

    def collect(name):
        def hook(module, args, kwargs):
            with torch.no_grad():
                q0, k0, v0 = args
                qm, km = kwargs["query_mask"], kwargs["key_mask"]
                q = module._heads(module.query(torch.where(qm[..., None], q0, 0.)))
                k = module._heads(module.key(torch.where(km[..., None], k0, 0.)))
                v = module._heads(module.value(torch.where(km[..., None], v0, 0.)))
                norms = {}
                for key, tensor, mask in (("query", q, qm), ("key", k, km)):
                    values = tensor.double().norm(dim=-1).masked_select(mask[:, None].expand(tensor.shape[:3]))
                    if not torch.isfinite(values).all():
                        raise FloatingPointError("nonfinite query/key norms")
                    norms[key] = describe(values.cpu().numpy())
                row = int(km.sum(-1).argmax().item())
                length = min(q.shape[2], PREFIX_LIMIT)
                qs, ks, vs = [a[row:row+1, :, :length] for a in (q, k, v)]
                qms, kms = qm[row:row+1, :length], km[row:row+1, :length]
                actual = causal_favor_attention(qs, ks, vs, module.projection,
                    query_mask=qms, key_mask=kms, chunk_size=module.config.chunk_size,
                    accumulation_dtype=module.config.accumulation_dtype)
                arrays = {key: tensor.detach().cpu().numpy() for key, tensor in (
                    ("q", qs), ("k", ks), ("v", vs), ("projection", module.projection),
                    ("query_mask", qms), ("key_mask", kms), ("actual", actual))}
                result = prefix_diagnostics(**arrays)
                result["norms_all_valid_tokens"] = norms
                if baseline is not None:
                    for key in ("query", "key"):
                        denominator = max(baseline[name]["norms_all_valid_tokens"][key]["p95"], 1e-30)
                        ratio = norms[key]["p95"] / denominator
                        result.setdefault("p95_ratio_to_initial", {})[key] = ratio
                        if ratio > NORM_GROWTH_REVIEW_RATIO:
                            result["review_flags"].append(f"{key}_p95_more_than_2x_initial")
                if save_prefix is not None:
                    save_prefix(name, arrays)
                entries[name] = result
        return hook

    was_training = model.training
    try:
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, FavorPlusAttention):
                handles.append(module.register_forward_pre_hook(collect(name), with_kwargs=True))
        with torch.no_grad():
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite fixed-example model logits")
        if not entries:
            raise ValueError("no FAVOR+ stages observed")
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return entries


def finite_tree(value) -> bool:
    """Includes Adam buffers; finite weights alone do not certify optimizer state."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def equal_tree(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right
