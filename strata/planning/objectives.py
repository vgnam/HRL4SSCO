"""L3 objectives: scalarise a vector of values indexed by hypotheses.

The belief of L3 is a finite distribution ``b`` over the hypothesis set
:math:`\\mathcal{H}`.  A planner estimates one value per hypothesis (the value
of the same macro under every law) and must turn that vector into a single
number.  This module provides the four objectives of the STRATA plan:

``mean``        weighted mean ``sum_i b_i v_i`` (the Bayes-optimal objective);
``cvar``        weighted lower-tail CVaR at level ``alpha`` (exact for a
                discrete weighted distribution, including interpolation of the
                boundary atom);
``maxmin``      minimum value over hypotheses with positive weight (robust);
``optimistic``  maximum value over hypotheses with positive weight.

Zero-weight hypotheses do not enter ``maxmin``/``optimistic``.  If every weight
is zero the objectives fall back to uniform weights, which keeps them total
functions instead of raising on a degenerate belief.  ``values`` may carry
leading dimensions, e.g. ``(n_macros, m)``; the aggregation is always over the
last axis and the result has shape ``values.shape[:-1]``.
"""

from __future__ import annotations

import numpy as np

MODES = ("mean", "cvar", "maxmin", "optimistic")


def _normalise_weights(weights, m):
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != m:
        raise ValueError("values has %d hypotheses but weights has %d" % (m, w.size))
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(m, 1.0 / max(m, 1), dtype=np.float64)
    return w / total


def _cvar_rows(values, weights, alpha):
    """CVaR of every row of ``values`` (shape ``(-1, m)``)."""
    m = values.shape[1]
    alpha = float(np.clip(alpha, 1e-12, 1.0))
    out = np.empty(values.shape[0], dtype=np.float64)
    for row in range(values.shape[0]):
        v = values[row]
        order = np.argsort(v, kind="stable")
        vs = v[order]
        ws = weights[order]
        cum = np.cumsum(ws)
        if alpha >= 1.0 - 1e-12 or cum[-1] <= 0.0:
            out[row] = float(np.sum(vs * ws) / max(cum[-1], 1e-12))
            continue
        idx = int(np.searchsorted(cum, alpha, side="left"))
        idx = min(idx, m - 1)
        before = float(cum[idx - 1]) if idx > 0 else 0.0
        boundary = max(alpha - before, 0.0)
        mass = np.sum(ws[:idx] * vs[:idx]) if idx > 0 else 0.0
        out[row] = float((mass + boundary * vs[idx]) / alpha)
    return out


def aggregate(values, weights, mode="mean", alpha=0.2):
    """Aggregate per-hypothesis values into one scalar per leading index.

    Parameters
    ----------
    values
        ``(m,)`` or ``(..., m)`` array; the hypothesis axis is the last one.
    weights
        ``(m,)`` belief weights (non-negative; they are renormalised).
    mode
        ``'mean'``, ``'cvar'``, ``'maxmin'`` or ``'optimistic'``.
    alpha
        Tail level of ``cvar`` (``0 < alpha <= 1``; e.g. ``0.2`` is the worst
        20% of the belief mass).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 0:
        raise ValueError("values must have at least one dimension")
    m = v.shape[-1]
    w = _normalise_weights(weights, m)
    mode = str(mode).lower()

    if mode == "mean":
        out = np.tensordot(v, w, axes=([-1], [0]))
    elif mode == "cvar":
        flat = v.reshape(-1, m)
        out = _cvar_rows(flat, w, alpha).reshape(v.shape[:-1])
    elif mode in ("maxmin", "optimistic"):
        positive = w > 0.0
        if not positive.any():
            positive = np.ones(m, dtype=bool)
        selected = v[..., positive]
        out = (
            np.min(selected, axis=-1) if mode == "maxmin" else np.max(selected, axis=-1)
        )
    else:
        raise ValueError("unknown aggregation mode %r; choose from %s" % (mode, MODES))
    out = np.asarray(out)
    return float(out) if out.ndim == 0 else out
