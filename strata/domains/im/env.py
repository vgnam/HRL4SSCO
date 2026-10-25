"""Vectorised IM environment (L0 + one stochastic propagation step).

Semantics are an exact port of ``experiments/kt_law/sim.transition`` (itself
copied from ``IM/Env.py``):

* seeds become active; a seed that was removed becomes active again and can
  propagate again;
* every node that is still inactive and was not seeded activates with the
  law's probability, driven by the active nodes of the previous round plus the
  seeds just placed;
* nodes that were active become ``removed``; the round reward is the number of
  nodes that left the inactive state (inactive seeds + new activations).

The deterministic layer (L0) is split into small helpers -- :func:`apply_seeds`,
:func:`eligible_mask`, :func:`retire_active`, :func:`reward_count`,
:func:`budget_ok` -- so later abstractions can reuse them.  The only stochastic
operation is ``Bernoulli(law.prob(features))``.
"""

import numpy as np

from strata.core.types import (
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    STATUS_REMOVED,
    Features,
    SimState,
)


def influence_operator(graph, law, op=None):
    """Influence operator ``act @ op`` gives each node's active-neighbour count."""
    if op is not None:
        return op
    if getattr(law, "two_hop", False):
        return (graph.adj + graph.adj @ graph.adj).tocsr()
    return graph.adj


def accumulate(law, E, a):
    """Exposure accumulation ``a + rho * E`` (``a`` when ``rho == 0``)."""
    rho = float(getattr(law, "rho", 0.0))
    return a + rho * E if rho else a


def make_state(graph, M=1, budget_left=0):
    """Fresh :class:`SimState` for ``M`` parallel paths."""
    n = graph.adj.shape[0]
    return SimState(
        status=np.zeros((M, n), dtype=np.int8),
        E=np.zeros((M, n), dtype=np.float32),
        t=0,
        budget_left=int(budget_left),
    )


def eligible_mask(status):
    """Boolean mask of nodes that may activate this round."""
    return status == STATUS_INACTIVE


def apply_seeds(status, seeds):
    """Activate ``seeds`` in place; return ``(seeded mask, seed reward)``.

    ``seeds`` may be one shared set ``(k,)`` or per-path sets ``(M, k)`` with
    ``-1`` padding.  Duplicate ids collapse (a node activates once), exactly as
    in the verified simulator.
    """
    M, n = status.shape
    if seeds is None or np.size(seeds) == 0:
        return (np.zeros((M, n), dtype=bool), np.zeros(M, dtype=np.float64))
    s = np.asarray(seeds)
    sm = np.repeat(s[None, :], M, axis=0) if s.ndim == 1 else s
    valid = sm >= 0
    rows = np.broadcast_to(np.arange(M)[:, None], sm.shape)
    seeded = np.zeros((M, n), dtype=bool)
    seeded[rows[valid], sm[valid]] = True
    was_inactive = status == STATUS_INACTIVE
    seed_reward = (was_inactive & seeded).sum(axis=1).astype(np.float64)
    status[seeded] = STATUS_ACTIVE
    return seeded, seed_reward


def retire_active(status):
    """Move every already-active node to ``removed`` (in place)."""
    status[status == STATUS_ACTIVE] = STATUS_REMOVED


def reward_count(before_status, after_status):
    """Number of nodes that left the inactive state, per path."""
    return ((before_status == STATUS_INACTIVE) & (after_status != STATUS_INACTIVE)).sum(
        axis=1
    )


def seed_count(seeds):
    """Number of valid seed entries (the budget consumed by a round)."""
    if seeds is None:
        return 0
    s = np.asarray(seeds)
    if s.size == 0:
        return 0
    valid = s >= 0
    return int(valid.sum()) if valid.ndim == 1 else int(valid.sum(axis=1).max())


def budget_ok(budget_left, seeds):
    """Whether placing ``seeds`` respects the remaining budget."""
    if isinstance(budget_left, SimState):
        budget_left = budget_left.budget_left
    return seed_count(seeds) <= int(budget_left)


def compute_features(graph, law, status, E, op=None):
    """Build the per-round :class:`Features` bundle without sampling."""
    op = influence_operator(graph, law, op)
    act = status == STATUS_ACTIVE
    a = act.astype(np.float32) @ op
    eligible = eligible_mask(status)
    E = accumulate(law, E, a)
    return Features(
        a=a,
        E=E,
        a2=None,
        eligible=eligible,
        indeg=graph.indeg,
        community=graph.community,
    )


def next_probabilities(graph, law, state, op=None):
    """Activation probability of every node in the next round, ``(M, n)``."""
    feats = compute_features(graph, law, state.status, state.E, op=op)
    p = law.prob(feats)
    return np.where(feats.eligible, p, 0.0)


def step(graph, law, state, seeds, rng, op=None, return_info=True):
    """One round for ``M`` paths.

    Mutates ``state`` in place (status/exposure/t/budget) and returns the
    contract tuple ``(state, reward (M,), info)``; pass ``return_info=False``
    for ``(state, reward)``.  The arithmetic mirrors
    ``kt_law/sim.transition`` operation for operation so that M0 can assert
    bit-for-bit equality.
    """
    status = state.status
    M, n = status.shape
    act = status == STATUS_ACTIVE
    reward = np.zeros(M, dtype=np.float64)

    seeded, seed_reward = apply_seeds(status, seeds)
    reward += seed_reward
    act = act | seeded

    op = influence_operator(graph, law, op)
    a = act.astype(np.float32) @ op
    eligible = eligible_mask(status)
    E = accumulate(law, state.E, a)
    feats = Features(
        a=a,
        E=E,
        a2=None,
        eligible=eligible,
        indeg=graph.indeg,
        community=graph.community,
    )
    p = np.where(eligible, law.prob(feats), 0.0)

    new = (rng.random((M, n)) < p) & eligible
    reward += new.sum(axis=1)

    E = np.where(new, 0.0, E)
    retire_active(status)
    status[new] = STATUS_ACTIVE
    E[status != STATUS_INACTIVE] = 0.0

    state.E = E
    state.t += 1
    state.budget_left = max(0, int(state.budget_left) - seed_count(seeds))

    if return_info:
        info = {"a": a, "eligible": eligible, "new": new, "p": p, "seeded": seeded}
        return state, reward, info
    return state, reward


def rollout(graph, law, seed_sets, M=400, seed=0, op=None):
    """Evaluate fixed per-round seed sets on ``M`` Monte-Carlo paths.

    Returns rewards ``(M, T)``.  Reusing ``seed`` across plans/laws gives common
    random numbers.
    """
    op = influence_operator(graph, law, op)
    n = graph.adj.shape[0]
    rng = np.random.default_rng(seed)
    state = SimState(
        status=np.zeros((M, n), dtype=np.int8),
        E=np.zeros((M, n), dtype=np.float32),
        t=0,
        budget_left=0,
    )
    R = np.zeros((M, len(seed_sets)), dtype=np.float64)
    for t, seeds in enumerate(seed_sets):
        _, R[:, t], _ = step(graph, law, state, seeds, rng, op=op)
    return R


def value(R, discount=None):
    """Mean episode value from an ``(M, T)`` reward matrix (with SE)."""
    if discount is None:
        per_path = R.sum(axis=1)
    else:
        per_path = R @ (discount ** np.arange(R.shape[1]))
    return float(per_path.mean()), float(per_path.std(ddof=1) / np.sqrt(R.shape[0]))


def evaluate_plan(graph, law, seed_sets, M=400, seed=0, discount=0.99, op=None):
    """Discounted and undiscounted value of a fixed open-loop plan."""
    R = rollout(graph, law, seed_sets, M=M, seed=seed, op=op)
    total, se_total = value(R)
    disc, se_disc = value(R, discount)
    return {
        "total": total,
        "total_se": se_total,
        "disc": disc,
        "disc_se": se_disc,
        "R": R,
    }


def realized_path(graph, law, seed_sets, seed, op=None):
    """One realised path, used by planners that react to their own outcomes."""
    return rollout(graph, law, seed_sets, M=1, seed=seed, op=op)
