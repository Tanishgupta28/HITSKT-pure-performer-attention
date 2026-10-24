"""Frozen-array controls, independent of training and production attention.

None of these controls edits checkpoints or selects parameters from outcomes.
Key translation preserves exact normalized softmax; Q/K scaling does not.
"""
from __future__ import annotations

import numpy as np
from ktbench.favor_plus_post50_diagnostics import attention_diagnostics, attention_weights, describe


def first_key_center(key, key_mask):
    key = np.asarray(key, dtype=np.float64)
    mask = np.asarray(key_mask, dtype=bool)
    if key.ndim != 3 or mask.shape != (key.shape[1],):
        raise ValueError('expected key[H,L,D], mask[L]')
    clean = np.where(mask[None, :, None], key, 0.)
    if not np.isfinite(clean).all():
        raise ValueError('nonfinite active key')
    first = np.flatnonzero(mask)
    center = clean[:, first[0]:first[0]+1] if len(first) else np.zeros_like(clean[:, :1])
    return np.where(mask[None, :, None], clean-center, 0.)


def projection_arm(pool, features, arm):
    pool = np.asarray(pool)
    if type(features) is not int or not 0 < features <= pool.shape[1]:
        raise ValueError('positive feature count within frozen pool required')
    if arm == 'antithetic_ORF':
        if features % 2:
            raise ValueError('antithetic total count must be even')
        half = pool[:, :features//2]
        return np.concatenate((half, -half), axis=1)
    if arm not in ('original_ORF', 'first_valid_key_centering_ORF'):
        raise ValueError('unknown diagnostic arm')
    return pool[:, :features].copy()


def evaluate_arm(q, k, v, projection, qm, km, arm='original_ORF'):
    original_key = np.asarray(k, dtype=np.float64)
    transformed = first_key_center(original_key, km) if arm == 'first_valid_key_centering_ORF' else original_key
    result = attention_diagnostics(q, transformed, v, projection, qm, km)
    if arm == 'first_valid_key_centering_ORF':
        # Independent explicit matrices prove target preservation on these inputs.
        base = attention_weights(q, original_key, projection, qm, km)[1]
        centered = attention_weights(q, transformed, projection, qm, km)[1]
        result['exact_softmax_translation_max_abs'] = float(np.max(np.abs(base-centered)))
        if result['exact_softmax_translation_max_abs'] > 1e-12:
            raise FloatingPointError('key translation changed exact softmax target')
    return result


def pair_geometry(q, k, qm, km):
    q, k = np.asarray(q, float), np.asarray(k, float)
    n, d = q.shape[1:]
    cosine, sums = [], []
    for h in range(q.shape[0]):
        for i in np.unique(np.linspace(0, n-1, min(n, 8)).astype(int)):
            if not qm[i]:
                continue
            js = np.flatnonzero(km[:i+1])
            for j in js[np.unique(np.linspace(0, len(js)-1, min(len(js), 12)).astype(int))] if len(js) else []:
                x, y = q[h, i]/d**.25, k[h, j]/d**.25
                cosine.append(float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30)))
                sums.append(float(np.sum((x+y)**2)))
    return dict(cosine=describe(cosine), scaled_pair_sum_norm_squared=describe(sums))


def value_offset_control(q, k, v, w, qm, km):
    valid = np.asarray(km, bool)
    offset = np.asarray(v, float)[:, valid].mean(axis=1, keepdims=True)
    shifted = np.asarray(v, float)-offset
    feature, exact, _ = attention_weights(q, k, w, qm, km)
    base_error = (feature-exact)@v
    shifted_error = (feature-exact)@shifted
    eligible = np.broadcast_to(np.asarray(qm)[None, :, None], base_error.shape)
    return dict(original=attention_diagnostics(q, k, v, w, qm, km),
                centered_values=attention_diagnostics(q, k, shifted, w, qm, km),
                absolute_error_invariance_max_abs=float(np.max(np.abs(base_error-shifted_error)[eligible])),
                interpretation='algebraic output-denominator control only; not a proposed model change')
