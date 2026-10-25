"""Behaviour-log collection for the IM benchmark.

The behaviour policies and the ``even`` budget schedule are ports of
``experiments/kt_law/planner.select_heuristic`` and
``experiments/kt_law/t2_fit.collect_logs``.  Keeping the RNG consumption
identical is essential: the M0 acceptance test replays the CC2/dblp log and
compares the total number of activations with ``t2_fit.collect_logs``.

A log is a list of :class:`strata.core.types.Round` records (one per round,
``M = 1``).  ``save_logs``/``load_logs`` persist it as a compressed ``.npz``
file with a flat round table plus episode offsets.
"""

import os

import numpy as np

from strata.core.types import Episode, Round, STATUS_ACTIVE, STATUS_INACTIVE
from strata.domains.im import env as im_env

BEHAVIOUR = ["random", "score", "indeg", "outdeg"]


def law_edge_weight(law, indeg):
    """Per-target edge weight for the ``score`` heuristic."""
    fn = getattr(law, "edge_weight", None)
    if fn is None:
        return 1.0 / np.maximum(indeg, 1)
    return fn(indeg)


def _topk(values, status, k):
    """Top-k ids per path, excluding active nodes; ``(M, k)`` with -1 padding."""
    M, n = status.shape
    k = int(min(k, n))
    v = np.where(status == STATUS_ACTIVE, -np.inf, values)
    idx = np.argpartition(-v, k - 1, axis=1)[:, :k]
    return idx.astype(np.int64)


def immediate_prob(graph, law, state, op=None):
    """Activation probability of every node in the next round, ``(M, n)``."""
    feats = im_env.compute_features(graph, law, state.status, state.E, op=op)
    return law.prob(feats)


def select_heuristic(rule, graph, law, state, k, rng=None, op=None):
    """Per-path top-k selection by a cheap behaviour rule.

    Rules: ``prob`` (law probability), ``score`` (repo heuristic: sum of edge
    weights towards inactive targets), ``indeg``, ``outdeg``, ``random``.
    Returns ``(M, k)`` int64 ids.
    """
    M, n = state.status.shape
    if k <= 0:
        return np.zeros((M, 0), dtype=np.int64)
    if rule == "prob":
        vals = immediate_prob(graph, law, state, op=op)
    elif rule == "score":
        w = law_edge_weight(law, graph.indeg)
        inactive = (state.status == STATUS_INACTIVE).astype(np.float64)
        # Use adj.T (not the cached csr adjT) to match kt_law bit for bit.
        vals = (inactive * w[None, :]) @ graph.adj.T
    elif rule == "indeg":
        vals = np.broadcast_to(graph.indeg.astype(np.float64), (M, n))
    elif rule == "outdeg":
        vals = np.broadcast_to(graph.outdeg.astype(np.float64), (M, n))
    elif rule == "random":
        rng = np.random.default_rng(0) if rng is None else rng
        vals = rng.random((M, n))
    else:
        raise KeyError("unknown selection rule %r" % (rule,))
    return _topk(np.asarray(vals, dtype=np.float64), state.status, k)


def even_allocation(t, T, remaining):
    """Even schedule: ``ceil(remaining / rounds_left)``, dump the rest last."""
    rounds_left = T - t
    if remaining <= 0 or rounds_left <= 0:
        return 0
    k = remaining if rounds_left == 1 else int(np.ceil(remaining / rounds_left))
    return int(min(k, remaining))


def collect_logs(graph, law, T=10, B=20, n_episodes=200, policies=None, seed=0):
    """Collect behaviour episodes under ``law``.

    Behaviour policies cycle over ``policies`` (default ``random, score,
    indeg, outdeg``), one policy per episode, with the even budget schedule.
    The returned structure is ``List[Episode]`` with one :class:`Round` per
    simulated round.
    """
    if policies is None:
        policies = BEHAVIOUR
    if isinstance(policies, str):
        policies = [policies]
    policies = [str(p) for p in policies]
    if not policies:
        raise ValueError("at least one behaviour policy is required")

    n = graph.adj.shape[0]
    op = im_env.influence_operator(graph, law)
    rng = np.random.default_rng(seed)
    episodes = []

    for ep in range(int(n_episodes)):
        policy = policies[ep % len(policies)]
        state = im_env.make_state(graph, M=1, budget_left=int(B))
        remaining = int(B)
        rounds = []
        for t in range(int(T)):
            k = even_allocation(t, int(T), remaining)
            if k > 0:
                seeds = select_heuristic(policy, graph, law, state, k, rng=rng, op=op)[
                    0
                ]
            else:
                seeds = np.zeros(0, dtype=np.int64)
            before = state.status[0].copy()
            _, _, info = im_env.step(
                graph, law, state, seeds, rng, op=op, return_info=True
            )
            rounds.append(
                Round(
                    a=info["a"][0].copy(),
                    eligible=info["eligible"][0].copy(),
                    y=info["new"][0].copy(),
                    seeded=info["seeded"][0].copy(),
                    before=before,
                    after=state.status[0].copy(),
                    policy=policy,
                )
            )
            remaining -= int(seeds.size)
        episodes.append(rounds)
    return episodes


# ------------------------------------------------------------------ npz format

_FORMAT_VERSION = 1


def _npz_path(path):
    path = os.fspath(path)
    return path if path.endswith(".npz") else path + ".npz"


def save_logs(path, episodes):
    """Save a list of IM episodes to a compressed ``.npz`` file."""
    path = _npz_path(path)
    flat = []
    ep_starts = [0]
    for ep in episodes:
        flat.extend(ep)
        ep_starts.append(len(flat))

    n = int(flat[0].before.size) if flat else 0
    if flat:
        a = np.stack([r.a for r in flat]).astype(np.float32)
        eligible = np.stack([r.eligible for r in flat]).astype(bool)
        y = np.stack([r.y for r in flat]).astype(bool)
        seeded = np.stack([r.seeded for r in flat]).astype(bool)
        before = np.stack([r.before for r in flat]).astype(np.int8)
        after = np.stack([r.after for r in flat]).astype(np.int8)
        policy = np.asarray([r.policy for r in flat], dtype="U64")
    else:
        a = np.zeros((0, n), dtype=np.float32)
        eligible = np.zeros((0, n), dtype=bool)
        y = np.zeros((0, n), dtype=bool)
        seeded = np.zeros((0, n), dtype=bool)
        before = np.zeros((0, n), dtype=np.int8)
        after = np.zeros((0, n), dtype=np.int8)
        policy = np.zeros(0, dtype="U64")

    np.savez_compressed(
        path,
        version=np.asarray([_FORMAT_VERSION], dtype=np.int64),
        n=np.asarray([n], dtype=np.int64),
        ep_starts=np.asarray(ep_starts, dtype=np.int64),
        a=a,
        eligible=eligible,
        y=y,
        seeded=seeded,
        before=before,
        after=after,
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
            raise ValueError("unsupported IM log format version %d" % version)
        ep_starts = data["ep_starts"].astype(np.int64)
        a = data["a"].copy()
        eligible = data["eligible"].copy()
        y = data["y"].copy()
        seeded = data["seeded"].copy()
        before = data["before"].copy()
        after = data["after"].copy()
        policy = data["policy"].copy()

    episodes = []
    for i in range(len(ep_starts) - 1):
        start, stop = int(ep_starts[i]), int(ep_starts[i + 1])
        rounds = [
            Round(
                a=a[j].copy(),
                eligible=eligible[j].copy(),
                y=y[j].copy(),
                seeded=seeded[j].copy(),
                before=before[j].copy(),
                after=after[j].copy(),
                policy=str(policy[j]),
            )
            for j in range(start, stop)
        ]
        episodes.append(rounds)
    return episodes
