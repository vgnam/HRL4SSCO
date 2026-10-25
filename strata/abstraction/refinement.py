"""Decision-gap refinement of the L2 abstraction.

A bucket of the abstract state is *decision-sufficient* when the concrete
states it lumps together agree on which operator to play. ``decision_gap``
measures disagreement: mean pairwise Kendall tau between the operator rankings
of the states, and the spread of the best operator's value. When the bucket
is not sufficient, ``propose_split`` returns the abstract feature and cut that
best separate states by their best operator (information gain), to be applied
with :meth:`strata.abstraction.abstract_state.Abstraction.split`.

Buckets are only split for decisions, never for prediction error alone.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, Sequence

import numpy as np
from scipy import stats


def decision_gap(values: np.ndarray, tau0: float = 0.5, delta: float = 1.0) -> Dict[str, float]:
    """``values[s, o]``: value of operator o in concrete state s of one bucket."""
    values = np.asarray(values, dtype=np.float64)
    taus = []
    for i, j in combinations(range(values.shape[0]), 2):
        t, _ = stats.kendalltau(values[i], values[j])
        taus.append(1.0 if np.isnan(t) else float(t))
    tau = float(np.mean(taus)) if taus else 1.0
    best = values.argmax(axis=1)
    # loss of playing one operator for the whole bucket instead of each state's best
    shared = values[:, np.bincount(best).argmax()]
    gap = float(np.mean(values.max(axis=1) - shared))
    return {"tau": tau, "gap": gap, "split": bool(tau < tau0 and gap > delta),
            "n_best_ops": int(np.unique(best).size)}


def _entropy(labels):
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def propose_split(features: Dict[str, np.ndarray], values: np.ndarray):
    """Best (feature, cut) separating states by their best operator, or None."""
    best_ops = np.asarray(values).argmax(axis=1)
    h0 = _entropy(best_ops)
    best = None
    for name, x in features.items():
        x = np.asarray(x, dtype=np.float64)
        for cut in np.unique(x)[:-1]:
            left, right = best_ops[x <= cut], best_ops[x > cut]
            h = (left.size * _entropy(left) + right.size * _entropy(right)) / best_ops.size
            gain = h0 - h
            if best is None or gain > best[2] + 1e-12:
                nxt = x[x > cut].min()
                best = (name, float((cut + nxt) / 2.0), float(gain))
    if best is None or best[2] <= 1e-12:
        return None
    return {"feature": best[0], "cut": best[1], "info_gain": best[2]}


def refine_bucket(states: Sequence, operators: Sequence[str], value_fn: Callable,
                  feature_fn: Callable, tau0: float = 0.5, delta: float = 1.0):
    """Evaluate operators on the states of one bucket; propose a split if needed."""
    values = np.asarray([[value_fn(s, o) for o in operators] for s in states], dtype=np.float64)
    gap = decision_gap(values, tau0, delta)
    gap["proposal"] = propose_split(feature_fn(states), values) if gap["split"] else None
    gap["values"] = values
    return gap
