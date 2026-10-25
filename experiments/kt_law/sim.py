"""Vectorised simulator of the IM environment for an arbitrary law.

Semantics copied from IM/Env.py (one propagation round per step):
  * seeds are set active (a seed that was removed becomes active again, and can
    propagate again -- IM/Env allows this);
  * every node still inactive and not seeded activates with the law's probability,
    driven by the active nodes of the previous round plus this round's seeds;
  * nodes that were active become "removed"; the round's reward is the number of
    nodes that left the inactive state (inactive seeds + new activations).

A rollout is vectorised over M independent Monte-Carlo paths. `transition` accepts
either one shared seed set (1-D) or per-path seed sets ((M,k) with -1 padding).
"""

import numpy as np

from laws import STATUS_INACTIVE, STATUS_ACTIVE, STATUS_REMOVED


def transition(law, op, indeg, status, E, seeds, rng, return_info=False):
    """One round. status (M,n) int8, E (M,n) float32. Returns status, E, reward (M,)."""
    M, n = status.shape
    act = status == STATUS_ACTIVE
    reward = np.zeros(M, dtype=np.float64)
    seeded = None

    if seeds is not None and np.size(seeds):
        s = np.asarray(seeds)
        sm = np.repeat(s[None, :], M, axis=0) if s.ndim == 1 else s
        valid = sm >= 0
        rows = np.broadcast_to(np.arange(M)[:, None], sm.shape)
        seeded = np.zeros((M, n), dtype=bool)
        seeded[rows[valid], sm[valid]] = True
        # count inactive seeds per path (duplicates collapse: a node is activated once)
        was_inactive = status == STATUS_INACTIVE
        reward += (was_inactive & seeded).sum(axis=1)
        status[seeded] = STATUS_ACTIVE
        act = act | seeded

    a = act.astype(np.float32) @ op
    eligible = status == STATUS_INACTIVE
    E = law.accumulate(E, a)
    p = np.where(eligible, law.prob(E, a, indeg), 0.0)

    new = (rng.random((M, n)) < p) & eligible
    reward += new.sum(axis=1)

    E = np.where(new, 0.0, E)
    status[status == STATUS_ACTIVE] = STATUS_REMOVED
    status[new] = STATUS_ACTIVE
    E[status != STATUS_INACTIVE] = 0.0
    if return_info:
        info = {'a': a, 'eligible': eligible, 'new': new, 'p': p,
                'seeded': seeded if seeded is not None else np.zeros((M, n), dtype=bool)}
        return status, E, reward, info
    return status, E, reward


def rollout(law, adj, indeg, seed_sets, M, seed, op=None):
    """Evaluate fixed per-round seed sets under `law` on M Monte-Carlo paths.

    Returns rewards (M, T) per round. Reusing `seed` across plans/laws gives
    common random numbers.
    """
    op = law.op(adj) if op is None else op
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    status = np.zeros((M, n), dtype=np.int8)
    E = np.zeros((M, n), dtype=np.float32)
    R = np.zeros((M, len(seed_sets)), dtype=np.float64)
    for t, seeds in enumerate(seed_sets):
        status, E, R[:, t] = transition(law, op, indeg, status, E, seeds, rng)
    return R


def value(R, discount=None):
    """Mean episode value from a (M,T) reward matrix; discounted if `discount` given."""
    if discount is None:
        per_path = R.sum(axis=1)
    else:
        per_path = R @ (discount ** np.arange(R.shape[1]))
    return float(per_path.mean()), float(per_path.std(ddof=1) / np.sqrt(R.shape[0]))


def evaluate_plan(law, adj, indeg, seed_sets, M=400, seed=0, discount=0.99, op=None):
    R = rollout(law, adj, indeg, seed_sets, M, seed, op=op)
    total, se_total = value(R)
    disc, se_disc = value(R, discount)
    return {'total': total, 'total_se': se_total, 'disc': disc, 'disc_se': se_disc, 'R': R}


def realized_path(law, adj, indeg, seed_sets, seed, op=None):
    """One realized path: used while planning so the planner sees its own outcomes."""
    return rollout(law, adj, indeg, seed_sets, M=1, seed=seed, op=op)
