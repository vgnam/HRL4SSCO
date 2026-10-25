"""Macro-operators of the L2 abstraction (IM and RPP).

IM operators
------------
:class:`Hub`, :class:`Gate`, :class:`Hedge` are direct ports of
``experiments/kt_law/ktb_hier.ground``: the same greedy, round-by-round
selection of the remaining seeds, so ``refine`` reproduces the verified kill
test exactly on the same input.  :class:`Prime` is new and maximises the
exposure written to inactive nodes (only meaningful when the law has
``rho > 0``); :class:`Wait` places no seeds.

RPP operators
-------------
:class:`Harvest` selects the ``k`` most profitable POIs of a region and orders
the route with nearest-neighbour + 2-opt; :class:`Sweep` follows the same
ordering but selects the spatially nearest POIs instead of the most profitable
ones; :class:`Reposition` moves the current position into a region by visiting
its nearest POI; :class:`Wait` takes an empty day (the drift law moves profits).

All ``refine`` methods are vectorised over the ``M`` Monte-Carlo paths and
return either an IM seed matrix ``(M, k)`` or an RPP action matrix ``(M, k)``,
padded with ``-1`` when a path has fewer entries.  ``refine`` never mutates the
state.  ``checker`` receives the states before/after one round and returns a
boolean ``(M,)`` array; the quantities a checker needs are recorded by the
preceding ``refine`` call on the operator instance.
"""

import numpy as np

from strata.core.types import (
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    Features,
)
from strata.domains.rpp import env as rpp_env

# --------------------------------------------------------------------------- IM


def _im_prob(law, E, a, graph, eligible):
    """``law.prob`` for an explicit exposure/active-neighbour pair (ktb port)."""
    feats = Features(
        a=a,
        E=E,
        a2=None,
        eligible=eligible,
        indeg=graph.indeg,
        community=graph.community,
    )
    return law.prob(feats)


def _parse_im_op(op):
    parts = str(op).split("-")
    kind = parts[0]
    gate = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return kind, gate


def ground_im(kind, gate, region_mask, k, law, graph, status, E):
    """Greedy seed sets for every path, the port of ``ktb_hier.ground``.

    Returns ``(seeds (M, k), aux)`` where ``aux`` records the predicted gain
    (sum of selected per-step scores), the hub-only predicted gain and the
    number of region nodes that reached the gate after all seeds.
    """
    M, n = status.shape
    k = max(0, int(k))
    seeds = -np.ones((M, k), dtype=np.int64)
    predicted = np.zeros(M, dtype=np.float64)
    hub_predicted = np.zeros(M, dtype=np.float64)
    reached = np.zeros(M, dtype=np.int64)
    aux = {
        "predicted": predicted,
        "hub_predicted": hub_predicted,
        "reached": reached,
        "gate": int(gate),
    }
    if k == 0:
        return seeds, aux

    gate = int(gate)
    if "-" in str(kind):
        kind, parsed_gate = _parse_im_op(kind)
        if gate <= 0:
            gate = parsed_gate
    aux["gate"] = int(gate)
    adj, adjT = graph.adj, graph.adjT
    act = status == STATUS_ACTIVE
    seeded = np.zeros((M, n), dtype=bool)
    rows = np.arange(M)

    for j in range(k):
        a = (act | seeded).astype(np.float32) @ adj
        eligible = (status == STATUS_INACTIVE) & ~seeded
        use = kind
        if kind == "hedge":
            use = "hub" if j % 2 == 0 else "gate"
        if use == "hub":
            P0 = _im_prob(law, _accumulate(law, E, a), a, graph, eligible)
            P1 = _im_prob(law, _accumulate(law, E, a + 1), a + 1, graph, eligible)
            D = np.where(eligible, P1 - P0, 0.0)
            gain = np.asarray(D @ adjT) + eligible * (1.0 - P0)
        else:
            # Completion counts most, progress towards the threshold a little.
            comp = eligible & (a == gate - 1)
            prog = eligible & (a < gate - 1)
            W = comp + 0.3 * prog
            gain = np.asarray(W.astype(np.float32) @ adjT) + 1e-3 * eligible

        cand = (status != STATUS_ACTIVE) & ~seeded & region_mask[None, :]
        gain = np.where(cand, gain, -np.inf)
        best = np.argmax(gain, axis=1)
        ok = np.isfinite(gain[rows, best])
        score = np.where(ok, gain[rows, best], 0.0)
        predicted += score
        if use == "hub":
            hub_predicted += score
        seeds[ok, j] = best[ok]
        seeded[rows[ok], best[ok]] = True

    if gate > 0:
        a_after = (act | seeded).astype(np.float32) @ adj
        target = (status == STATUS_INACTIVE) & ~seeded & region_mask[None, :]
        reached = (target & (a_after >= float(gate))).sum(axis=1).astype(np.int64)
    aux["reached"] = reached
    return seeds, aux


def _accumulate(law, E, a):
    """Use the law's own accumulation when available (bit-compatible port)."""
    fn = getattr(law, "accumulate", None)
    if fn is not None:
        return fn(E, a)
    rho = float(getattr(law, "rho", 0.0))
    return a + rho * E if rho else a


def _new_activations(before, after):
    return ((before.status == STATUS_INACTIVE) & (after.status != STATUS_INACTIVE)).sum(
        axis=1
    )


class Hub:
    """Seed the nodes with the largest expected one-round activation gain."""

    name = "hub"
    kind = "hub"

    def __init__(self):
        self.last = {}

    def precondition(self, z, k, region):
        return int(k) > 0

    def refine(self, graph, law, state, k, region_mask):
        seeds, aux = ground_im(
            "hub", 0, region_mask, k, law, graph, state.status, state.E
        )
        self.last = aux
        return seeds

    def checker(self, before, after, region_mask):
        predicted = self.last.get("predicted")
        M = after.status.shape[0]
        if predicted is None or np.asarray(predicted).size != M:
            return np.ones(M, dtype=bool)
        actual = _new_activations(before, after)
        ok = actual >= 0.5 * np.asarray(predicted) - 1e-9
        return np.asarray(ok, dtype=bool)


class Gate:
    """Threshold coverage: push inactive nodes in a region towards ``gate``."""

    kind = "gate"

    def __init__(self, gate=2, min_nodes=1, g=None):
        self.gate = int(g if g is not None else gate)
        self.min_nodes = max(1, int(min_nodes))
        self.name = "gate-%d" % self.gate
        self.params = {"gate": self.gate, "min_nodes": self.min_nodes}
        self.last = {}

    def precondition(self, z, k, region):
        return int(k) > 0

    def refine(self, graph, law, state, k, region_mask):
        seeds, aux = ground_im(
            "gate", self.gate, region_mask, k, law, graph, state.status, state.E
        )
        self.last = aux
        return seeds

    def checker(self, before, after, region_mask):
        reached = self.last.get("reached")
        M = after.status.shape[0]
        if reached is None or np.asarray(reached).size != M:
            return _new_activations(before, after) > 0
        return np.asarray(reached >= self.min_nodes, dtype=bool)


class Hedge:
    """Alternate hub and gate picks (hub on even steps, gate on odd ones)."""

    kind = "hedge"

    def __init__(self, gate=2, min_nodes=1, g=None):
        self.gate = int(g if g is not None else gate)
        self.min_nodes = max(1, int(min_nodes))
        self.name = "hedge-%d" % self.gate
        self.params = {"gate": self.gate, "min_nodes": self.min_nodes}
        self.last = {}

    def precondition(self, z, k, region):
        return int(k) > 0

    def refine(self, graph, law, state, k, region_mask):
        seeds, aux = ground_im(
            "hedge", self.gate, region_mask, k, law, graph, state.status, state.E
        )
        self.last = aux
        return seeds

    def checker(self, before, after, region_mask):
        M = after.status.shape[0]
        reached = self.last.get("reached")
        predicted = self.last.get("hub_predicted")
        actual = _new_activations(before, after)
        gate_ok = (
            np.asarray(reached) >= self.min_nodes
            if reached is not None and np.asarray(reached).size == M
            else np.zeros(M, dtype=bool)
        )
        hub_ok = (
            (np.asarray(predicted) > 0) & (actual >= 0.5 * np.asarray(predicted) - 1e-9)
            if predicted is not None and np.asarray(predicted).size == M
            else np.zeros(M, dtype=bool)
        )
        return np.asarray(gate_ok | hub_ok, dtype=bool)


class Prime:
    """Write exposure to inactive nodes, maximising ``sum_v delta E_v``.

    Seeding ``u`` raises the active-neighbour count of its out-neighbours, so
    the immediate exposure gain of an inactive target ``v`` is one unit; the
    operator scales the objective by ``rho`` because accumulated exposure only
    matters to laws that reuse it.  With ``rho == 0`` it places nothing.
    """

    name = "prime"
    kind = "prime"

    def __init__(self):
        self.last = {}

    def precondition(self, z, k, region):
        return int(k) > 0

    def refine(self, graph, law, state, k, region_mask):
        M, n = state.status.shape
        k = max(0, int(k))
        seeds = -np.ones((M, k), dtype=np.int64)
        predicted = np.zeros(M, dtype=np.float64)
        rho = float(getattr(law, "rho", 0.0))
        self.last = {"predicted": predicted, "rho": rho}
        if k == 0 or rho <= 0.0:
            return seeds

        seeded = np.zeros((M, n), dtype=bool)
        rows = np.arange(M)
        for j in range(k):
            targets = (state.status == STATUS_INACTIVE) & ~seeded
            gain = rho * np.asarray(targets.astype(np.float32) @ graph.adjT)
            cand = (state.status == STATUS_INACTIVE) & ~seeded & region_mask[None, :]
            gain = np.where(cand, gain, -np.inf)
            best = np.argmax(gain, axis=1)
            ok = np.isfinite(gain[rows, best]) & (gain[rows, best] > 0)
            predicted += np.where(ok, gain[rows, best], 0.0)
            seeds[ok, j] = best[ok]
            seeded[rows[ok], best[ok]] = True
        return seeds

    def checker(self, before, after, region_mask):
        predicted = np.asarray(self.last.get("predicted", []), dtype=np.float64)
        M = after.status.shape[0]
        if predicted.size != M or self.last.get("rho", 0.0) <= 0.0:
            return np.zeros(M, dtype=bool)
        diff = np.maximum(after.E - before.E, 0.0)
        mask = after.status == STATUS_INACTIVE
        actual = np.where(mask, diff, 0.0).sum(axis=1)
        return np.asarray(actual >= 0.5 * predicted - 1e-9, dtype=bool)


class Wait:
    """Empty action: place no seeds (IM) or take an empty day (RPP)."""

    name = "wait"
    kind = "wait"

    def __init__(self):
        self.last = {}

    def precondition(self, z, k, region):
        return True

    def refine(self, graph, law, state, k, region_mask):
        M = state.status.shape[0] if hasattr(state, "status") else state.profit.shape[0]
        self.last = {"action": np.zeros((M, 0), dtype=np.int64)}
        return np.zeros((M, 0), dtype=np.int64)

    def checker(self, before, after, region_mask):
        b = np.asarray(before.budget_left)
        a = np.asarray(after.budget_left)
        M = after.status.shape[0] if hasattr(after, "status") else after.profit.shape[0]
        if b.ndim == 0:
            return np.full(M, bool(a == b), dtype=bool)
        return np.asarray(a == b, dtype=bool).reshape(-1)[:M]


# -------------------------------------------------------------------------- RPP


def _rpp_available(state, path, region_mask):
    """Unvisited, still-profitable POIs of ``path`` inside the region."""
    return np.flatnonzero((state.profit[path] > 0.0) & np.asarray(region_mask))


def _order_distance(env, start, order, closed):
    """Raw distance of an ordered route; ``closed`` adds the return to start."""
    if not order:
        return env.dist[start, env.start_node] if closed else 0.0
    d = env.dist[start, order[0]]
    for u, v in zip(order[:-1], order[1:]):
        d += env.dist[u, v]
    if closed:
        d += env.dist[order[-1], env.start_node]
    return float(d)


def _nearest_order(env, start, nodes):
    """Greedy nearest-neighbour ordering of ``nodes`` from ``start``."""
    pool = [int(v) for v in nodes]
    order = []
    cur = int(start)
    while pool:
        best = min(pool, key=lambda v: (env.dist[cur, v], v))
        order.append(best)
        pool.remove(best)
        cur = best
    return order


def _two_opt(env, start, order, closed=False, max_rounds=50):
    """2-opt improvement of an open (or returning) route with a fixed start."""
    best = [int(v) for v in order]
    if len(best) < 2:
        return best
    d = env.dist
    for _ in range(int(max_rounds)):
        improved = False
        n = len(best)
        for i in range(n - 1):
            prev = start if i == 0 else best[i - 1]
            for j in range(i + 1, n):
                first, last = best[i], best[j]
                if j + 1 < n:
                    nxt = best[j + 1]
                    old = d[prev, first] + d[last, nxt]
                    new = d[prev, last] + d[first, nxt]
                else:
                    old = d[prev, first]
                    new = d[prev, last]
                    if closed:
                        old += d[last, env.start_node]
                        new += d[first, env.start_node]
                if new < old - 1e-12:
                    best[i : j + 1] = best[i : j + 1][::-1]
                    improved = True
        if not improved:
            break
    return best


def _trim_route(env, start, order, closed, max_daily_distance):
    """Drop POIs until the normalised route respects the daily distance limit."""
    order = [int(v) for v in order]
    limit = float(max_daily_distance) * env.max_dist
    while order and _order_distance(env, start, order, closed) > limit + 1e-9:
        best = None
        for pos in range(len(order)):
            cand = order[:pos] + order[pos + 1 :]
            d = _order_distance(env, start, cand, closed)
            if best is None or d < best[0]:
                best = (d, cand)
        order = best[1]
    return order


class _RPPRoute:
    """Shared refine/checker for routing operators (Harvest and Sweep)."""

    name = "route"
    kind = "route"

    def __init__(self):
        self.last = {}

    def precondition(self, z, k, region):
        return int(k) > 0

    def _select(self, env, state, path, region_mask, k):
        raise NotImplementedError

    def refine(self, env, law, state, k, region_mask):
        M, n = state.profit.shape
        width = int(max(0, min(int(k), n)))
        action = np.full((M, width), -1, dtype=np.int64)
        region_mask = np.asarray(region_mask, dtype=bool)
        closed = bool(state.t == env.T)
        for m in range(M):
            start = int(state.current[m])
            chosen = self._select(env, state, m, region_mask, width)
            order = _nearest_order(env, start, chosen)
            order = _two_opt(env, start, order, closed=closed)
            order = _trim_route(env, start, order, closed, env.max_daily_distance)
            action[m, : len(order)] = order
        lengths = rpp_env.action_lengths(action)
        dist = rpp_env.total_distance(env, state, action)
        # The environment treats an all-empty day as zero distance even on the
        # last day, where a non-empty route would have to return to the start.
        dist = np.where(lengths > 0, dist, 0.0)
        self.last = {
            "action": action,
            "distance": dist,
            "k": width,
            "region_mask": region_mask,
            "env": env,
            "closed": closed,
        }
        return action

    def checker(self, before, after, region_mask):
        action = self.last.get("action")
        env = self.last.get("env")
        M = after.profit.shape[0]
        if action is None or env is None or action.shape[0] != M:
            return np.ones(M, dtype=bool)
        region_mask = np.asarray(region_mask, dtype=bool)
        dist = np.asarray(self.last["distance"], dtype=np.float64)
        ok = dist <= float(env.max_daily_distance) + 1e-9
        for m in range(M):
            nodes = action[m][action[m] >= 0]
            if nodes.size == 0:
                ok[m] = False
                continue
            if len(set(int(v) for v in nodes)) != nodes.size:
                ok[m] = False
            if not np.all(region_mask[nodes]):
                ok[m] = False
            if not np.all(before.profit[m, nodes] > 0.0):
                ok[m] = False
            if nodes.size > self.last.get("k", nodes.size):
                ok[m] = False
        return np.asarray(ok, dtype=bool)


class Harvest(_RPPRoute):
    """Top-``k`` profitable POIs of a region, routed nearest-neighbour + 2-opt."""

    name = "harvest"
    kind = "harvest"

    def _select(self, env, state, path, region_mask, k):
        avail = _rpp_available(state, path, region_mask)
        if avail.size == 0 or k <= 0:
            return []
        profit = state.profit[path, avail]
        order = np.lexsort((avail, -profit))[: int(k)]
        return avail[order]


class Sweep(_RPPRoute):
    """Spatially nearest ``k`` unvisited POIs of a region, then route-optimised."""

    name = "sweep"
    kind = "sweep"

    def _select(self, env, state, path, region_mask, k):
        pool = [int(v) for v in _rpp_available(state, path, region_mask)]
        chosen = []
        cur = int(state.current[path])
        while pool and len(chosen) < int(k):
            best = min(pool, key=lambda v: (env.dist[cur, v], v))
            chosen.append(best)
            pool.remove(best)
            cur = best
        return chosen


class Reposition:
    """Move the current position into a region through its nearest POI.

    The RPP environment only changes the position by visiting a POI, so a
    reposition is modelled as a one-POI day; it is cheap (no submodular cost)
    and leaves the position inside the target region.
    """

    name = "reposition"
    kind = "reposition"

    def __init__(self):
        self.last = {}

    def precondition(self, z, k, region):
        return int(k) > 0

    def refine(self, env, law, state, k, region_mask):
        M, n = state.profit.shape
        width = 1 if int(k) > 0 else 0
        action = np.full((M, width), -1, dtype=np.int64)
        region_mask = np.asarray(region_mask, dtype=bool)
        for m in range(M):
            avail = _rpp_available(state, m, region_mask)
            if avail.size == 0 or width == 0:
                continue
            start = int(state.current[m])
            action[m, 0] = min(avail, key=lambda v: (env.dist[start, v], v))
        lengths = rpp_env.action_lengths(action)
        dist = rpp_env.total_distance(env, state, action)
        dist = np.where(lengths > 0, dist, 0.0)
        # A reposition that violates the daily limit is not executable.
        if dist.size and np.any(dist > env.max_daily_distance + 1e-9):
            action = np.where(
                (dist > env.max_daily_distance + 1e-9)[:, None], -1, action
            )
            lengths = rpp_env.action_lengths(action)
            dist = np.where(lengths > 0, dist, 0.0)
        self.last = {
            "action": action,
            "distance": dist,
            "k": width,
            "region_mask": region_mask,
            "env": env,
        }
        return action

    def checker(self, before, after, region_mask):
        action = self.last.get("action")
        env = self.last.get("env")
        M = after.profit.shape[0]
        if action is None or env is None or action.shape[0] != M:
            return np.ones(M, dtype=bool)
        region_mask = np.asarray(region_mask, dtype=bool)
        dist = np.asarray(self.last["distance"], dtype=np.float64)
        ok = dist <= float(env.max_daily_distance) + 1e-9
        for m in range(M):
            nodes = action[m][action[m] >= 0]
            if nodes.size != 1:
                ok[m] = False
                continue
            node = int(nodes[0])
            if not region_mask[node]:
                ok[m] = False
            if before.profit[m, node] <= 0.0:
                ok[m] = False
            if not region_mask[int(after.current[m])]:
                ok[m] = False
        return np.asarray(ok, dtype=bool)
