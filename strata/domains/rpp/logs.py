"""Behaviour-log collection for the RPP benchmark.

Policies operate on one realised path and choose an ordered list of POIs for
the day:

``greedy``   profit-greedy tour (port of ``RPP/Baselines.Greedy``): repeatedly
             take the most profitable reachable unvisited POI;
``nearest``  nearest-neighbour tour under the daily distance limit;
``random``   a uniformly random subset of unvisited POIs.

The budget follows the same ``even`` schedule as the IM logs.  Each round is
stored as an :class:`RPPRound` with the profit/visited snapshots before and
after the day, so M1 can fit drift distributions from observed changes.
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np

from strata.domains.rpp import env as rpp_env


@dataclass
class RPPRound:
    """One logged day of an RPP episode (``M = 1``)."""

    profit_before: np.ndarray
    profit_after: np.ndarray
    visited_before: np.ndarray
    visited_after: np.ndarray
    action: np.ndarray
    reward: float
    distance: float
    t: int
    current_before: int
    current_after: int
    policy: str


RPPEpisode = List[RPPRound]

BEHAVIOUR = ["greedy", "nearest", "random"]


def even_allocation(remaining, t, T):
    """Even budget schedule: ceil(remaining / rounds_left), dump the rest last."""
    rounds_left = int(T) - int(t)
    if remaining <= 0 or rounds_left <= 0:
        return 0
    k = remaining if rounds_left == 1 else int(np.ceil(remaining / rounds_left))
    return int(min(k, remaining))


def _reachable(env, current, total, node):
    return (total + env.dist[current, node]) / env.max_dist <= env.max_daily_distance


def select_action(rule, env, state, k, rng):
    """Choose up to ``k`` POIs for one path; returns an int64 action array."""
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    profit = state.profit[0]
    available = [int(v) for v in np.flatnonzero(profit > 0)]
    if not available:
        return np.zeros(0, dtype=np.int64)
    current = int(state.current[0])

    if rule in ("greedy", "nearest"):
        chosen = []
        total = 0.0
        pool = list(available)
        while len(chosen) < int(k) and pool:
            best = None
            best_value = -np.inf
            for node in pool:
                if not _reachable(env, current, total, node):
                    continue
                value = profit[node] if rule == "greedy" else -env.dist[current, node]
                if value > best_value:
                    best_value = value
                    best = node
            if best is None:
                break
            chosen.append(best)
            total += env.dist[current, best]
            current = best
            pool.remove(best)
        return np.asarray(chosen, dtype=np.int64)

    if rule == "random":
        k = int(min(k, len(available)))
        picks = rng.choice(len(available), size=k, replace=False)
        return np.asarray([available[i] for i in picks], dtype=np.int64)

    if rule == "profit":
        order = sorted(available, key=lambda v: (-profit[v], v))
        return np.asarray(order[: int(k)], dtype=np.int64)

    raise KeyError("unknown RPP behaviour rule %r" % (rule,))


def collect_logs(env, law, T=None, budget=None, n_episodes=100, policies=None, seed=0):
    """Collect behaviour episodes under a hidden drift law.

    Returns ``List[RPPEpisode]`` (one :class:`RPPRound` per played day).
    """
    T = env.T if T is None else int(T)
    budget = env.budget if budget is None else int(budget)
    if policies is None:
        policies = BEHAVIOUR
    if isinstance(policies, str):
        policies = [policies]
    policies = [str(p) for p in policies]
    if not policies:
        raise ValueError("at least one behaviour policy is required")

    rng = np.random.default_rng(seed)
    episodes = []
    for ep in range(int(n_episodes)):
        policy = policies[ep % len(policies)]
        state = env.reset(M=1)
        remaining = int(budget)
        rounds = []
        for t in range(T):
            k = even_allocation(remaining, t, T)
            action = (
                select_action(policy, env, state, k, rng)
                if k > 0
                else np.zeros(0, dtype=np.int64)
            )
            before_profit = state.profit[0].copy()
            before_visited = state.visited[0].copy()
            before_current = int(state.current[0])
            t_before = int(state.t)
            _, reward, done, info = rpp_env.step(
                env, state, action, law=law, rng=rng, return_info=True
            )
            rounds.append(
                RPPRound(
                    profit_before=before_profit,
                    profit_after=state.profit[0].copy(),
                    visited_before=before_visited,
                    visited_after=state.visited[0].copy(),
                    action=np.asarray(action, dtype=np.int64).copy(),
                    reward=float(reward[0]),
                    distance=float(info["distance"][0]),
                    t=t_before,
                    current_before=before_current,
                    current_after=int(state.current[0]),
                    policy=policy,
                )
            )
            remaining -= int(len(action))
            if done:
                break
        episodes.append(rounds)
    return episodes


# ------------------------------------------------------------------ npz format

_FORMAT_VERSION = 1


def _npz_path(path):
    path = os.fspath(path)
    return path if path.endswith(".npz") else path + ".npz"


def save_logs(path, episodes):
    """Save RPP episodes to a compressed ``.npz`` file."""
    path = _npz_path(path)
    flat = []
    ep_starts = [0]
    for ep in episodes:
        flat.extend(ep)
        ep_starts.append(len(flat))

    n = int(flat[0].profit_before.size) if flat else 0
    kmax = max((r.action.size for r in flat), default=0)
    R = len(flat)

    profit_before = (
        np.stack([r.profit_before for r in flat]).astype(np.float64)
        if flat
        else np.zeros((0, n))
    )
    profit_after = (
        np.stack([r.profit_after for r in flat]).astype(np.float64)
        if flat
        else np.zeros((0, n))
    )
    visited_before = (
        np.stack([r.visited_before for r in flat]).astype(bool)
        if flat
        else np.zeros((0, n), dtype=bool)
    )
    visited_after = (
        np.stack([r.visited_after for r in flat]).astype(bool)
        if flat
        else np.zeros((0, n), dtype=bool)
    )
    action = np.full((R, kmax), -1, dtype=np.int64)
    for i, r in enumerate(flat):
        action[i, : r.action.size] = r.action
    action_len = np.asarray([r.action.size for r in flat], dtype=np.int64)
    reward = np.asarray([r.reward for r in flat], dtype=np.float64)
    distance = np.asarray([r.distance for r in flat], dtype=np.float64)
    t = np.asarray([r.t for r in flat], dtype=np.int64)
    current_before = np.asarray([r.current_before for r in flat], dtype=np.int64)
    current_after = np.asarray([r.current_after for r in flat], dtype=np.int64)
    policy = np.asarray([r.policy for r in flat], dtype="U64")

    np.savez_compressed(
        path,
        version=np.asarray([_FORMAT_VERSION], dtype=np.int64),
        n=np.asarray([n], dtype=np.int64),
        ep_starts=np.asarray(ep_starts, dtype=np.int64),
        profit_before=profit_before,
        profit_after=profit_after,
        visited_before=visited_before,
        visited_after=visited_after,
        action=action,
        action_len=action_len,
        reward=reward,
        distance=distance,
        t=t,
        current_before=current_before,
        current_after=current_after,
        policy=policy,
    )
    return path


def load_logs(path):
    """Load episodes written by :func:`save_logs`."""
    path = os.fspath(path)
    if not os.path.exists(path):
        alt = _npz_path(path)
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        version = int(data["version"][0])
        if version != _FORMAT_VERSION:
            raise ValueError("unsupported RPP log format version %d" % version)
        ep_starts = data["ep_starts"].astype(np.int64)
        profit_before = data["profit_before"].copy()
        profit_after = data["profit_after"].copy()
        visited_before = data["visited_before"].copy()
        visited_after = data["visited_after"].copy()
        action = data["action"].copy()
        action_len = data["action_len"].copy()
        reward = data["reward"].copy()
        distance = data["distance"].copy()
        t = data["t"].copy()
        current_before = data["current_before"].copy()
        current_after = data["current_after"].copy()
        policy = data["policy"].copy()

    episodes = []
    for e in range(len(ep_starts) - 1):
        start, stop = int(ep_starts[e]), int(ep_starts[e + 1])
        rounds = []
        for j in range(start, stop):
            k = int(action_len[j])
            rounds.append(
                RPPRound(
                    profit_before=profit_before[j].copy(),
                    profit_after=profit_after[j].copy(),
                    visited_before=visited_before[j].copy(),
                    visited_after=visited_after[j].copy(),
                    action=action[j, :k].copy(),
                    reward=float(reward[j]),
                    distance=float(distance[j]),
                    t=int(t[j]),
                    current_before=int(current_before[j]),
                    current_after=int(current_after[j]),
                    policy=str(policy[j]),
                )
            )
        episodes.append(rounds)
    return episodes
