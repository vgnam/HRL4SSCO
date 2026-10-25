"""Vectorised RPP environment (L0 + hidden profit drift).

Semantics are a port of ``RPP/Env.py``:

* a day's action is an **ordered sequence** of POIs;
* reward is ``sum(profit[action]) - eta * (k - 1)^2``;
* distance is normalised by the largest pairwise distance; exceeding
  ``max_daily_distance`` costs ``distance_penalty * excess``;
* on the last day the route must return to the start node;
* visited POIs are removed from the set of visitable POIs and the remaining
  profits drift (the hidden law).

The deterministic layer (budget, unvisited set, distance, penalty, submodular
cost, current position) is exposed as module-level helpers.  The only
stochastic part is the drift law, which receives the realised state and returns
per-node profit changes.

``M`` independent paths are supported.  RPP state is domain-specific (each path
carries a profit vector, a visited mask, a current node and a budget), so it
uses :class:`RPPState` rather than the IM :class:`SimState`.
"""

from dataclasses import dataclass

import numpy as np

from strata.domains.rpp.points import distance_matrix


@dataclass
class RPPState:
    """State of ``M`` parallel RPP paths."""

    profit: np.ndarray  # (M, n) current profit of every POI
    visited: np.ndarray  # (M, n) bool, already visited this episode
    current: np.ndarray  # (M,) current position
    budget_left: np.ndarray  # (M,) remaining POI budget
    t: int  # 1-based day index (RPP/Env.py convention)

    def clone(self):
        return RPPState(
            self.profit.copy(),
            self.visited.copy(),
            self.current.copy(),
            self.budget_left.copy(),
            self.t,
        )


class RPPEnv:
    """Parameters and immutable geometry of one RPP instance."""

    def __init__(
        self,
        loc,
        initial_profit=None,
        budget=20,
        T=10,
        max_daily_distance=1.5,
        distance_penalty=5.0,
        start_node=0,
        profit_factor=0.05,
        profit_min=0.3,
        eta=0.1,
        seed=0,
        empty_day_advances=False,
    ):
        self.loc = np.asarray(loc, dtype=np.float64)
        if self.loc.ndim != 2 or self.loc.shape[1] != 2:
            raise ValueError("loc must have shape (n, 2)")
        self.n = self.loc.shape[0]
        self.T = int(T)
        self.budget = int(budget)
        self.max_daily_distance = float(max_daily_distance)
        self.distance_penalty = float(distance_penalty)
        self.start_node = int(start_node)
        self.profit_factor = float(profit_factor)
        self.profit_min = float(profit_min)
        self.eta = float(eta)
        # RPP/Env.py ignores an empty day entirely (no drift, the day counter
        # does not move), so "wait for profits to rise" is impossible there.
        # False keeps that behaviour for parity; the STRATA benchmark sets True
        # so that an empty day drifts the profits and consumes the day.
        self.empty_day_advances = bool(empty_day_advances)

        if initial_profit is None:
            rng = np.random.default_rng(seed)
            initial_profit = rng.uniform(0.5, 1.0, self.n)
        self.initial_profit = np.asarray(initial_profit, dtype=np.float64).copy()
        if self.initial_profit.shape != (self.n,):
            raise ValueError("initial_profit must have shape (n,)")

        self.dist = distance_matrix(self.loc)
        self.max_dist = float(np.max(self.dist))

    # ------------------------------------------------------------- lifecycle
    def reset(self, M=1):
        """Fresh state for ``M`` paths (day 1, full budget, start position)."""
        profit = np.tile(self.initial_profit, (int(M), 1))
        profit[:, self.start_node] = 0.0
        visited = profit <= 0.0
        return RPPState(
            profit=profit,
            visited=visited,
            current=np.full(int(M), self.start_node, dtype=np.int64),
            budget_left=np.full(int(M), self.budget, dtype=np.int64),
            t=1,
        )

    def step(self, state, action, law=None, rng=None, return_info=False):
        return step(self, state, action, law=law, rng=rng, return_info=return_info)

    # --------------------------------------------------------- L0 conveniences
    def get_not_visited(self, state, path=0):
        return not_visited(state, path=path)

    def cal_total_distance(self, state, action):
        return total_distance(self, state, action)

    def cal_reward(self, state, action):
        A = as_action_matrix(action, state.profit.shape[0])
        lengths = action_lengths(A)
        return profit_reward(self, state.profit, A, lengths)


# --------------------------------------------------------------------- L0 layer


def not_visited(state, path=0):
    """Indices of POIs still visitable by ``path`` (profit strictly positive)."""
    return np.flatnonzero(state.profit[path] > 0)


def as_action_matrix(action, M):
    """Broadcast a shared action to ``(M, k)`` (``-1`` pads shorter paths)."""
    a = np.asarray(action, dtype=np.int64)
    if a.size == 0:
        return np.zeros((int(M), 0), dtype=np.int64)
    if a.ndim == 1:
        return np.repeat(a[None, :], int(M), axis=0)
    if a.shape[0] != int(M):
        raise ValueError("action has %d rows, expected %d" % (a.shape[0], M))
    return a


def action_lengths(A):
    """Number of valid POIs per path (trailing ``-1`` padding is ignored)."""
    return (np.asarray(A) >= 0).sum(axis=1)


def total_distance(env, state, action, lengths=None):
    """Normalised route length per path (return-to-start on the last day).

    Vectorised over paths and sequence positions.  ``cumsum`` accumulates in
    the same left-to-right order as the reference environment, so ``M = 1``
    remains bit-identical to ``RPP/Env.py``.
    """
    A = as_action_matrix(action, state.profit.shape[0])
    lengths = action_lengths(A) if lengths is None else np.asarray(lengths)
    M = A.shape[0]
    out = np.zeros(M, dtype=np.float64)
    if A.shape[1] == 0:
        if state.t == env.T:
            out[:] = env.dist[state.current, env.start_node] / env.max_dist
        return out

    valid = A >= 0
    idx = np.where(valid, A, 0)
    seg = np.zeros((M, A.shape[1]), dtype=np.float64)
    seg[:, 0] = env.dist[state.current, idx[:, 0]]
    if A.shape[1] > 1:
        seg[:, 1:] = env.dist[idx[:, :-1], idx[:, 1:]]
    seg = np.where(valid, seg, 0.0)

    rows = np.arange(M)
    has = lengths > 0
    cumulative = np.cumsum(seg, axis=1)
    last = np.where(has, lengths - 1, 0)
    out[has] = cumulative[rows[has], last[has]]
    if state.t == env.T:
        out[has] += env.dist[idx[has, last[has]], env.start_node]
    out /= env.max_dist
    return out


def profit_reward(env, profit, A, lengths=None):
    """Submodular profit minus ``eta * (k - 1)^2`` per path (vectorised)."""
    A = np.asarray(A)
    lengths = action_lengths(A) if lengths is None else np.asarray(lengths)
    M = A.shape[0]
    out = np.zeros(M, dtype=np.float64)
    if A.shape[1] > 0:
        valid = A >= 0
        idx = np.where(valid, A, 0)
        rows = np.arange(M)
        vals = np.where(valid, profit[rows[:, None], idx], 0.0)
        cumulative = np.cumsum(vals, axis=1)
        has = lengths > 0
        last = np.where(has, lengths - 1, 0)
        out[has] = cumulative[rows[has], last[has]]
    out -= env.eta * (np.maximum(lengths - 1, 0) ** 2)
    return out


def distance_penalty_reward(env, reward, dist):
    """Subtract the over-distance penalty where the daily limit is exceeded."""
    excess = dist - env.max_daily_distance
    return np.where(
        dist > env.max_daily_distance, reward - env.distance_penalty * excess, reward
    )


def budget_ok(state, lengths):
    """Whether every path can pay for its action."""
    lengths = np.asarray(lengths)
    return bool(np.all(lengths <= state.budget_left))


def apply_visits(env, state, A, lengths=None):
    """Mark action POIs visited, spend budget and move the current position."""
    A = np.asarray(A)
    lengths = action_lengths(A) if lengths is None else np.asarray(lengths)
    if not budget_ok(state, lengths):
        raise AssertionError("Budget exceeded")
    M, n = state.profit.shape
    visited_now = np.zeros((M, n), dtype=bool)
    rows = np.broadcast_to(np.arange(M)[:, None], A.shape)
    valid = A >= 0
    visited_now[rows[valid], A[valid]] = True
    state.profit[visited_now] = 0.0
    state.visited |= visited_now
    state.budget_left -= lengths.astype(np.int64)
    has = lengths > 0
    if np.any(has):
        idx = np.where(A >= 0, A, 0)
        state.current[has] = idx[has, lengths[has] - 1]
    return visited_now


def apply_drift(env, state, visited_now, law, rng):
    """Draw a drift from ``law`` and update unvisited profits (floor at min)."""
    if law is None:
        return np.zeros_like(state.profit)
    delta = law.drift(env, state, visited_now, rng)
    delta = np.asarray(delta, dtype=np.float64)
    unvisited = state.profit > 0.0
    updated = np.maximum(state.profit + delta, env.profit_min)
    state.profit = np.where(unvisited, updated, state.profit)
    return delta


# ------------------------------------------------------------------- dynamics


def step(env, state, action, law=None, rng=None, return_info=False):
    """One day for ``M`` paths.

    Mutates ``state`` in place and returns ``(state, reward (M,), done)``; with
    ``return_info`` also returns a dict with distance, reward terms and the
    visited mask.  An all-empty action reproduces the reference environment:
    no state change, zero reward, ``done = (t == T)``.
    """
    A = as_action_matrix(action, state.profit.shape[0])
    lengths = action_lengths(A)
    M = A.shape[0]

    if np.all(lengths == 0):
        reward = np.zeros(M, dtype=np.float64)
        delta = np.zeros_like(state.profit)
        if env.empty_day_advances:
            delta = apply_drift(env, state, np.zeros_like(state.visited), law, rng)
            state.t += 1
            done = bool(state.t > env.T)
        else:
            done = bool(state.t == env.T)
        if return_info:
            dist = np.zeros(M, dtype=np.float64)
            info = {
                "distance": dist,
                "base_reward": reward.copy(),
                "penalty": np.zeros(M, dtype=np.float64),
                "k": lengths,
                "visited_now": np.zeros_like(state.visited),
                "delta": delta,
            }
            return state, reward, done, info
        return state, reward, done

    dist = total_distance(env, state, A, lengths)
    base = profit_reward(env, state.profit, A, lengths)
    reward = distance_penalty_reward(env, base, dist)
    reward = np.where(lengths > 0, reward, 0.0)

    visited_now = apply_visits(env, state, A, lengths)
    delta = apply_drift(env, state, visited_now, law, rng)

    state.t += 1
    done = bool(state.t > env.T)
    if return_info:
        info = {
            "distance": dist,
            "base_reward": base,
            "penalty": np.where(lengths > 0, reward - base, 0.0),
            "k": lengths,
            "visited_now": visited_now,
            "delta": delta,
        }
        return state, reward, done, info
    return state, reward, done
