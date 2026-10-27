"""Classic adaptive influence-maximisation baselines on the STRATA IM env.

All three re-plan every round on the *current* state (the residual graph of
still-inactive nodes), i.e. they are the adaptive versions of their static
algorithms, and spend the budget with a fixed schedule (``even``, ``normal``
or ``static``, the allocations of ``strata.baselines.flat``):

* :class:`DegreeDiscountPolicy` -- DegreeDiscountIC (Chen, Wang & Yang, KDD
  2009) on the residual graph, edge probability ``1 / indeg`` (weighted
  cascade): pick by ``dd_v = d_v - 2 t_v - (d_v - t_v) t_v p_v``.
* :class:`RRSetPolicy` -- reverse-reachable-set coverage as in IMM / AdaIMM
  (Tang et al. 2015; Han et al. 2018): sample ``n_rr`` RR sets on the residual
  graph under the weighted cascade and pick the ``k`` nodes covering the most
  (greedy max coverage).  A fixed number of RR sets replaces IMM's adaptive
  sample-size bound.
* :class:`AdaptiveGreedyPolicy` -- adaptive greedy (Golovin & Krause 2011):
  each seed of the round maximises the Monte-Carlo estimate of the expected
  number of activations over the *rest of the episode* (no later seeds), with
  CELF lazy evaluation and common random numbers across candidates.  It
  simulates under a given law -- the instance's true law by default, which
  makes it a privileged baseline like ``oracle``.

The ``*`` variants (:func:`best_schedule`) choose the schedule with the
policy's own model: DDIC and RR sets under the weighted cascade (their model
assumption), adaptive greedy under the law it simulates.
"""

from __future__ import annotations

import heapq
from typing import Optional

import numpy as np

from strata.baselines.flat import ALLOCS, STATIC_PERIOD, allocation_k
from strata.core.types import STATUS_ACTIVE, STATUS_INACTIVE, Features
from strata.domains.im import env as im_env
from strata.domains.im import hidden_laws as im_laws


class _ScheduledIM:
    """Shared policy plumbing: a budget schedule plus a per-round seed selector."""

    method = "im-classic"

    def __init__(self, alloc="even", period=STATIC_PERIOD, seed=0):
        if alloc not in ALLOCS:
            raise KeyError("unknown allocation %r; choose from %s" % (alloc, ALLOCS))
        self.alloc, self.period, self.seed = str(alloc), int(period), int(seed)
        self.instance = None

    def reset(self, instance, rng) -> None:
        self.instance = instance
        self.graph = instance.graph
        self.rng = rng if rng is not None else np.random.default_rng(self.seed)

    def act(self, state, t: int) -> np.ndarray:
        remaining = int(state.budget_left)
        k = allocation_k(self.alloc, t, self.instance.T, remaining, self.period)
        status = np.asarray(state.status)[0]
        eligible = status == STATUS_INACTIVE
        k = min(int(k), int(eligible.sum()))
        if k <= 0:
            return np.zeros(0, dtype=np.int64)
        return np.asarray(self.select(state, t, k), dtype=np.int64)

    def observe(self, state_before, action, state_after, info) -> None:
        return None


# --------------------------------------------------------------------- DDIC


class DegreeDiscountPolicy(_ScheduledIM):
    """DegreeDiscountIC on the residual graph (weighted-cascade edge probabilities)."""

    method = "ddic"

    def select(self, state, t, k):
        g = self.graph
        status = np.asarray(state.status)[0]
        alive = status == STATUS_INACTIVE
        adj = g.adj
        w = 1.0 / np.maximum(np.asarray(g.indeg, dtype=np.float64), 1.0)  # p of an edge into v
        out_alive = np.asarray(adj @ alive.astype(np.float64)).ravel()  # d_v on the residual graph
        # mean probability of v's residual out-edges (DDIC assumes one p per edge)
        p = np.asarray(adj @ (alive * w)).ravel() / np.maximum(out_alive, 1.0)
        tcount = np.zeros_like(out_alive)
        dd = out_alive.copy()
        chosen = []
        cand = alive.copy()
        for _ in range(k):
            score = np.where(cand, dd, -np.inf)
            u = int(np.argmax(score))
            if not np.isfinite(score[u]):
                break
            chosen.append(u)
            cand[u] = False
            # in-neighbours v of u (edge v -> u) lose the overlap with u
            for v in g.adjT[u].indices:
                if cand[v]:
                    tcount[v] += 1
                    dd[v] = out_alive[v] - 2 * tcount[v] - (out_alive[v] - tcount[v]) * tcount[v] * p[v]
        return chosen


# ------------------------------------------------------------------ RR sets


class RRSetPolicy(_ScheduledIM):
    """Reverse-reachable-set max coverage (IMM / AdaIMM style) on the residual graph."""

    method = "imm-rr"

    def __init__(self, alloc="even", period=STATIC_PERIOD, seed=0, n_rr=2000):
        super().__init__(alloc, period, seed)
        self.n_rr = int(n_rr)

    def select(self, state, t, k):
        g = self.graph
        status = np.asarray(state.status)[0]
        alive = status == STATUS_INACTIVE
        roots_pool = np.flatnonzero(alive)
        w = 1.0 / np.maximum(np.asarray(g.indeg, dtype=np.float64), 1.0)
        indptr, indices = g.adjT.indptr, g.adjT.indices  # in-neighbours of v: adjT[v]
        rng = self.rng
        sets = []
        for root in rng.choice(roots_pool, size=self.n_rr):
            seen = {int(root)}
            frontier = [int(root)]
            while frontier:
                v = frontier.pop()
                nbrs = indices[indptr[v]:indptr[v + 1]]
                if nbrs.size == 0:
                    continue
                live = nbrs[rng.random(nbrs.size) < w[v]]
                for u in live:
                    u = int(u)
                    if alive[u] and u not in seen:
                        seen.add(u)
                        frontier.append(u)
            sets.append(seen)
        # greedy max coverage
        n = alive.size
        covers = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for u in s:
                covers[u].append(i)
        gain = np.asarray([len(c) for c in covers], dtype=np.float64)
        gain[~alive] = -np.inf
        covered = np.zeros(len(sets), dtype=bool)
        chosen = []
        for _ in range(k):
            u = int(np.argmax(gain))
            if not np.isfinite(gain[u]):
                break
            chosen.append(u)
            gain[u] = -np.inf
            for i in covers[u]:
                if not covered[i]:
                    covered[i] = True
                    for x in sets[i]:
                        if np.isfinite(gain[x]):
                            gain[x] -= 1.0
        return chosen


# ---------------------------------------------------------- adaptive greedy


def _rollout_crn(graph, law, state, first_seeds, U):
    """Total activations of each path: ``first_seeds`` (P, k) now, no later seeds.

    ``U`` (H, P, n) are the uniforms of the H remaining rounds, shared across
    candidates by the caller (common random numbers).
    """
    P = first_seeds.shape[0]
    status = np.repeat(np.asarray(state.status)[:1], P, axis=0).astype(np.int8)
    E = np.repeat(np.asarray(state.E)[:1], P, axis=0).astype(np.float32)
    op = im_env.influence_operator(graph, law)
    total = np.zeros(P, dtype=np.float64)
    for h in range(U.shape[0]):
        act = status == STATUS_ACTIVE
        if h == 0:
            seeded, seed_reward = im_env.apply_seeds(status, first_seeds)
            total += seed_reward
            act = act | seeded
        a = act.astype(np.float32) @ op
        eligible = status == STATUS_INACTIVE
        E = im_env.accumulate(law, E, a)
        feats = Features(a=a, E=E, a2=None, eligible=eligible, indeg=graph.indeg,
                         community=graph.community)
        p = np.where(eligible, law.prob(feats), 0.0)
        new = (U[h] < p) & eligible
        total += new.sum(axis=1)
        E = np.where(new, 0.0, E)
        im_env.retire_active(status)
        status[new] = STATUS_ACTIVE
        E[status != STATUS_INACTIVE] = 0.0
    return total


class AdaptiveGreedyPolicy(_ScheduledIM):
    """Adaptive greedy with CELF and Monte-Carlo marginal gains (simulates ``law``)."""

    method = "ada-greedy"

    def __init__(self, alloc="even", period=STATIC_PERIOD, seed=0, n_paths=16, law=None):
        super().__init__(alloc, period, seed)
        self.n_paths = int(n_paths)
        self.law = law

    def reset(self, instance, rng) -> None:
        super().reset(instance, rng)
        self.sim_law = self.law if self.law is not None else instance.law

    def select(self, state, t, k):
        g, law, R = self.graph, self.sim_law, self.n_paths
        H = max(1, int(self.instance.T) - int(t))
        n = np.asarray(state.status).shape[1]
        alive = np.asarray(state.status)[0] == STATUS_INACTIVE
        U = self.rng.random((H, R, n))

        def value(sets):
            """Mean value of each seed set in ``sets`` (list of lists), CRN across sets."""
            k_max = max(1, max(len(s) for s in sets))
            seeds = -np.ones((len(sets) * R, k_max), dtype=np.int64)
            for i, s in enumerate(sets):
                if s:
                    seeds[i * R:(i + 1) * R, :len(s)] = s
            tot = _rollout_crn(g, law, state, seeds, np.tile(U, (1, len(sets), 1)))
            return tot.reshape(len(sets), R).mean(axis=1)

        cands = [int(v) for v in np.flatnonzero(alive)]
        chosen = []
        base = float(value([[]])[0])
        gains = value([[v] for v in cands]) - base
        heap = [(-float(gv), v, 0) for gv, v in zip(gains, cands)]
        heapq.heapify(heap)
        while heap and len(chosen) < k:
            neg, v, stamp = heapq.heappop(heap)
            if stamp == len(chosen):  # gain is fresh for the current seed set
                chosen.append(v)
                base = base - neg
                continue
            fresh = float(value([chosen + [v]])[0]) - base
            heapq.heappush(heap, (-fresh, v, len(chosen)))
        return chosen


# ------------------------------------------------------------- schedules


def best_schedule(make, instance, model_law, episodes=2, seed=0):
    """``make(alloc)`` with the schedule that scores best when simulated under ``model_law``."""
    from strata.eval import protocol

    plan = instance.replace(law=model_law)
    best, best_v = None, -np.inf
    for alloc in ALLOCS:
        v = float(np.mean(protocol.evaluate(make(alloc), plan, episodes=episodes, seed=seed)))
        if v > best_v:
            best, best_v = alloc, v
    policy = make(best)
    policy.selected_alloc, policy.selection_value = best, best_v
    return policy


def make_policy(method, instance, seed=0, selection_episodes=2):
    """``ddic*``, ``imm-rr*`` or ``ada-greedy*`` with its self-selected schedule."""
    wc = im_laws.make_law("WC", instance.graph)
    if method == "ddic*":
        return best_schedule(lambda a: DegreeDiscountPolicy(a, seed=seed), instance, wc,
                             selection_episodes, seed)
    if method == "imm-rr*":
        return best_schedule(lambda a: RRSetPolicy(a, seed=seed), instance, wc,
                             selection_episodes, seed)
    if method == "ada-greedy*":
        return best_schedule(lambda a: AdaptiveGreedyPolicy(a, seed=seed), instance,
                             instance.law, selection_episodes, seed)
    raise KeyError("unknown classic IM method %r" % (method,))
