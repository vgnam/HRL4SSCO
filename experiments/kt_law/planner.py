"""Planners for the sequential seeding problem under a given law.

A plan is a list of T seed sets (one per round), i.e. an open-loop schedule that
can be replayed under another law. Three decisions make a planner:

  allocation  how much budget to spend this round
              'even'     spend ceil(remaining / rounds_left), dump the rest last
              'normal'   spend everything in round 1
              'static'   spend every `period` rounds
              'lookahead' pick k in 0..kmax by Monte-Carlo evaluation of the whole
                          remaining episode (continuation = 'even' policy)
  selection   which nodes to seed
              'greedy'   CELF-style greedy with Monte-Carlo one-round gains (law-aware,
                         considers one-hop spread and re-seeding of removed nodes)
              'prob'     top-k by immediate activation probability under the law
              'score'    top-k by the repo's score: sum of edge weights to inactive nodes
              'indeg'    top-k by in-degree (the "followers" heuristic)
              'outdeg'   top-k by out-degree
              'random'
"""

import numpy as np

from laws import STATUS_INACTIVE, STATUS_ACTIVE, STATUS_REMOVED
from sim import transition


# ---------------------------------------------------------------- selection rules

def immediate_prob(law, op, indeg, status, E):
    """Activation probability of every node in the next round, shape (M,n)."""
    act = status == STATUS_ACTIVE
    a = act.astype(np.float32) @ op
    return law.prob(law.accumulate(E, a), a, indeg)


def _topk(values, status, k):
    """Top-k node ids per path, excluding nodes that are already active. (M,k) with -1 pad."""
    M, n = status.shape
    k = int(min(k, n))
    v = np.where(status == STATUS_ACTIVE, -np.inf, values)
    idx = np.argpartition(-v, k - 1, axis=1)[:, :k]
    return idx.astype(np.int64)


def select_heuristic(rule, law, adj, op, indeg, status, E, k, rng=None):
    """Per-path top-k selection by a cheap rule. Returns (M,k) ids."""
    M, n = status.shape
    if k <= 0:
        return np.zeros((M, 0), dtype=np.int64)
    if rule == 'prob':
        vals = immediate_prob(law, op, indeg, status, E)
    elif rule == 'score':
        # repo heuristic: sum of edge weights towards currently inactive targets
        w = law.edge_weight(indeg)
        inactive = (status == STATUS_INACTIVE).astype(np.float64)
        vals = (inactive * w[None, :]) @ adj.T
    elif rule == 'indeg':
        vals = np.broadcast_to(indeg.astype(np.float64), (M, n))
    elif rule == 'outdeg':
        outdeg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float64)
        vals = np.broadcast_to(outdeg, (M, n))
    elif rule == 'random':
        rng = np.random.default_rng(0) if rng is None else rng
        vals = rng.random((M, n))
    else:
        raise KeyError(f'unknown selection rule {rule!r}')
    return _topk(np.asarray(vals, dtype=np.float64), status, k)


def greedy_sequence(law, adj, op, indeg, status_row, E_row, k, M, seed, max_cand=200):
    """CELF-style greedy on one-round Monte-Carlo gains for a single realised state.

    Returns (seeds, gains) with len <= k. Evaluating every candidate in one batched
    transition keeps this exact in the one-round objective (the state is realised, so
    the only randomness is this round's propagation).
    """
    n = status_row.shape[0]
    seeded = np.zeros(n, dtype=bool)
    seeds, gains = [], []
    for step in range(int(k)):
        cand = np.flatnonzero((status_row != STATUS_ACTIVE) & ~seeded)
        if cand.size == 0:
            break
        if cand.size > max_cand:
            p = immediate_prob(law, op, indeg, status_row[None, :], E_row[None, :])[0]
            p = np.where(seeded[cand], -np.inf, p[cand])
            cand = cand[np.argsort(-p)[:max_cand]]
            cand.sort()
        C = cand.size
        st = np.repeat(status_row[None, :], C * M, axis=0).astype(np.int8)
        Es = np.repeat(E_row[None, :], C * M, axis=0).astype(np.float32)
        if seeds:
            st[:, np.asarray(seeds)] = STATUS_ACTIVE
        # block c occupies rows [c*M, (c+1)*M) and seeds candidate cand[c]
        st[np.arange(C * M), np.repeat(cand, M)] = STATUS_ACTIVE
        rng = np.random.default_rng(seed + 12345 * (step + 1))
        _, _, r = transition(law, op, indeg, st, Es, None, rng)
        # a candidate that is still inactive also scores its own activation
        gains_c = r.reshape(C, M).mean(axis=1) + (status_row[cand] == STATUS_INACTIVE)
        best = int(np.argmax(gains_c))
        seeds.append(int(cand[best]))
        gains.append(float(gains_c[best]))
        seeded[seeds[-1]] = True
    return np.asarray(seeds, dtype=np.int64), np.asarray(gains, dtype=np.float64)


# ---------------------------------------------------------------- continuation value

def rollout_value_from(law, adj, op, indeg, status_row, E_row, first_seeds,
                       rounds_left, remaining_after, M, seed, alloc='even',
                       max_per_round=None):
    """MC value of the remaining episode: `first_seeds` now, then an 'even' continuation."""
    n = status_row.shape[0]
    status = np.repeat(status_row[None, :], M, axis=0).astype(np.int8)
    E = np.repeat(E_row[None, :], M, axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    total = np.zeros(M, dtype=np.float64)

    status, E, r = transition(law, op, indeg, status, E, first_seeds, rng)
    total += r
    rem = remaining_after
    for t in range(1, rounds_left):
        left = rounds_left - t
        if rem <= 0:
            break
        if alloc == 'even':
            k = rem if left == 0 else int(np.ceil(rem / left))
        elif alloc == 'normal':
            k = 0
        else:
            raise KeyError(alloc)
        if max_per_round is not None:
            k = min(k, max_per_round)
        k = int(min(k, rem))
        if k > 0:
            vals = immediate_prob(law, op, indeg, status, E)
            seeds = _topk(vals, status, k)
            status, E, r = transition(law, op, indeg, status, E, seeds, rng)
            total += r
            rem -= k
    return float(total.mean())


# ---------------------------------------------------------------- episode planner

def _allocation_k(alloc, t, rounds_left, remaining, period):
    if alloc == 'even':
        return remaining if rounds_left == 1 else min(remaining, int(np.ceil(remaining / rounds_left)))
    if alloc == 'normal':
        return remaining if t == 0 else 0
    if alloc == 'static':
        if remaining == 0 or t % period != 0:
            return 0
        n_spends = int(np.ceil(rounds_left / period))  # rounds t, t+period, ... still available
        return min(remaining, int(np.ceil(remaining / max(1, n_spends))))
    raise KeyError(alloc)


def plan_episode(law, adj, indeg, T, budget, alloc='lookahead', select='greedy',
                 M_greedy=96, M_look=64, seed=0, kmax=8, max_cand=200, period=2,
                 op=None, keep_state=False):
    """Run one episode under `law`, planning online; returns the seed sets per round.

    The planner sees the realised state (one sampled path under `law`), so the plan
    is closed-loop with respect to its own outcomes, exactly like a deployed policy.
    """
    op = law.op(adj) if op is None else op
    n = adj.shape[0]
    rng = np.random.default_rng(seed + 555)
    status = np.zeros((1, n), dtype=np.int8)
    E = np.zeros((1, n), dtype=np.float32)
    seed_sets = []
    remaining = int(budget)

    for t in range(T):
        rounds_left = T - t
        if remaining <= 0:
            S = np.zeros(0, dtype=np.int64)
        elif alloc == 'lookahead':
            kmax_t = int(min(remaining, kmax))
            seq, gains = greedy_sequence(law, adj, op, indeg, status[0], E[0],
                                         kmax_t, M_greedy, seed + 7919 * t, max_cand)
            best_k, best_v = 0, -np.inf
            for k in range(len(seq) + 1):
                S_k = seq[:k]
                v = rollout_value_from(law, adj, op, indeg, status[0], E[0], S_k,
                                       rounds_left, remaining - k, M_look,
                                       seed + 104729 * (t + 1) + 31 * k)
                if v > best_v:
                    best_k, best_v = k, v
            S = seq[:best_k]
        else:
            k = _allocation_k(alloc, t, rounds_left, remaining, period)
            k = int(min(k, remaining))
            if k > 0 and select == 'greedy':
                S, _ = greedy_sequence(law, adj, op, indeg, status[0], E[0], k,
                                       M_greedy, seed + 7919 * t, max_cand)
            elif k > 0:
                S = select_heuristic(select, law, adj, op, indeg, status, E, k)[0]
            else:
                S = np.zeros(0, dtype=np.int64)

        S = np.asarray(S, dtype=np.int64)
        seed_sets.append(S)
        status, E, _ = transition(law, op, indeg, status, E, S, rng)
        remaining -= int(S.size)

    if keep_state:
        return seed_sets, status[0].copy(), E[0].copy()
    return seed_sets


def plan_episode_heuristic(law, adj, indeg, T, budget, rule, alloc='even', period=2,
                           seed=0, op=None):
    """Closed-loop heuristic planner: cheap rule, no Monte-Carlo search."""
    if rule == 'greedy':
        raise ValueError("use plan_episode(..., select='greedy') for the greedy rule")
    return plan_episode(law, adj, indeg, T, budget, alloc=alloc, select=rule,
                        seed=seed, op=op)
