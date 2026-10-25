"""L3 planner of STRATA: MCTS over macros with root sampling over the hypothesis set.

The tree is *open-loop*: a node is the sequence of macros played from the
current round, and every simulation re-samples the stochastic outcomes, so
the tree needs no state discretisation. Following BAMCP, each simulation
draws one hypothesis law from the belief at the root and keeps it for the
whole simulation; children of the root store visit counts and returns *per
hypothesis*, so the root decision can scalarise the per-law values with
:func:`strata.planning.objectives.aggregate` (mean, CVaR, max-min or
optimistic). Inside the tree, selection is UCT on the overall mean return
with progressive widening; macros are tried in the order "base policy's macro
first, then the rest". Leaves are valued by the flat base policy (IM: hub with
the best fixed schedule; RPP: harvest on "all", even schedule) on ``M_leaf``
paths.

All simulation machinery (macro enumeration, grounding through the operators,
environment steps, flat continuation) is inherited from
:class:`strata.planning.rollout_planner.RolloutPlanner`, so the one-step
rollout planner is exactly MCTS with depth 1 and one visit per macro.
"""

from __future__ import annotations

import math

import numpy as np

from strata.abstraction import regions as regions_mod
from strata.domains.im import env as im_env
from strata.domains.rpp import env as rpp_env
from strata.planning.objectives import aggregate
from strata.planning.rollout_planner import (
    RolloutPlanner,
    _clone_state,
    _expand_im,
    _expand_rpp,
    _remaining,
    _used,
)


class _Node:
    __slots__ = ("macros", "children", "N", "W")

    def __init__(self):
        self.macros = None  # candidate macros, base macro first
        self.children = {}  # macro index -> _Node
        self.N = 0
        self.W = 0.0


class MCTSPlanner(RolloutPlanner):
    """Open-loop MCTS over macros (see module docstring).

    Parameters beyond :class:`RolloutPlanner`
    -----------------------------------------
    n_sims
        Simulations per decision.
    c_uct
        Exploration constant, scaled by the spread of observed returns.
    widening
        ``(k0, alpha)``: a node with ``N`` visits may expand at most
        ``ceil(k0 * (N + 1) ** alpha)`` children.
    M_leaf
        Paths used by the flat leaf evaluation.
    tree
        ``'root'`` (default): only the root branches; every simulation plays
        one root macro and continues with the flat base policy, with common
        random numbers across root macros. Each value is then an unbiased
        estimate of "this macro, then the base policy", so replanning every
        round is a rollout improvement of the base policy. ``'full'``: the
        original open-loop tree; its mean backup averages over exploratory
        macros later in the episode, which undervalues macros that keep budget
        and makes the planner front-load the budget as the tree widens.
    min_visits
        With ``tree='root'``, visits a root macro needs to compete in the
        final decision.
    """

    def __init__(self, library, base=None, n_sims=200, c_uct=1.0, widening=(2.0, 0.5),
                 M_leaf=4, M_prior=4, mode="mean", seed=0, T=None, alpha=0.2, regions=None,
                 tree="root", min_visits=3, **kwargs):
        super().__init__(library, base=base, M=M_leaf, mode=mode, seed=seed, T=T,
                         alpha=alpha, regions=regions, **kwargs)
        self.M_prior = max(1, int(M_prior))
        self.n_sims = max(1, int(n_sims))
        self.c_uct = float(c_uct)
        self.k0, self.k_alpha = float(widening[0]), float(widening[1])
        self.M_leaf = max(1, int(M_leaf))
        self.tree = str(tree).lower()
        if self.tree not in ("root", "full"):
            raise ValueError("tree must be 'root' or 'full'")
        self.min_visits = max(1, int(min_visits))
        self.last_info = {}

    # ------------------------------------------------------------- helpers
    def _terminal(self, state, domain, T):
        return int(state.t) >= T if domain == "im" else int(state.t) > T

    def _step(self, obj, domain, state, macro, law, rng):
        """Ground ``macro`` under ``law`` on a one-path state and step it."""
        mask = regions_mod.region_mask(self.regions, macro.region)
        k = max(0, min(int(macro.k), _remaining(state, domain)))
        action = self._operator(macro.op).refine(obj, law, state, k, mask)
        if domain == "im":
            _, reward, _ = im_env.step(obj, law, state, action, rng)
        else:
            _, reward, _, _ = rpp_env.step(obj, state, action, law=law, rng=rng,
                                           return_info=True)
        return float(np.asarray(reward, dtype=np.float64).reshape(-1)[0])

    def _leaf(self, obj, domain, state, law, T, spec, rng):
        if self._terminal(state, domain, T):
            return 0.0
        st = _expand_im(state, self.M_leaf) if domain == "im" else _expand_rpp(state, self.M_leaf)
        rem = _remaining(state, domain)
        return float(self._continue_flat(obj, domain, st, law, rem, T, spec, rng).mean())

    def _allowed(self, node):
        return min(len(node.macros), int(math.ceil(self.k0 * (node.N + 1) ** self.k_alpha)))

    # ------------------------------------------------------------- Planner API
    def choose(self, env_or_graph, state, belief, t):
        domain = self._domain(state)
        obj = env_or_graph
        T = self._resolve_T(obj)
        self._ensure_regions(obj)
        spec = self._base_spec(obj, domain, state)
        if str(spec["alloc"]).lower() == "auto":
            spec["alloc"] = self._auto_alloc(obj, domain, state, belief)
        laws, weights = self._belief(belief)
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        m = len(laws)

        rng = np.random.default_rng(self.seed * 1000 + int(t))
        root = _Node()
        root.macros = self._root_prior(obj, domain, state, int(t), T, spec, laws, weights)
        n_root = len(root.macros)
        root_N = np.zeros((n_root, m))
        root_W = np.zeros((n_root, m))
        lo, hi = np.inf, -np.inf

        if self.tree == "root":
            return self._choose_root(obj, domain, state, int(t), T, spec, laws, weights,
                                     rng, root, root_N, root_W)

        for _ in range(self.n_sims):
            li = int(rng.choice(m, p=weights))
            law = laws[li]
            st = _clone_state(state)
            node, path, rewards = root, [], []
            leaf_value = 0.0
            while True:
                if self._terminal(st, domain, T):
                    break
                if node.macros is None:
                    node.macros = self._ordered_macros(st, int(st.t), T, domain, spec)
                if not node.macros:
                    break
                allowed = self._allowed(node)
                untried = [i for i in range(allowed) if i not in node.children]
                if untried:
                    ci = untried[0]
                    child = _Node()
                    node.children[ci] = child
                    rewards.append(self._step(obj, domain, st, node.macros[ci], law, rng))
                    path.append((node, ci))
                    leaf_value = self._leaf(obj, domain, st, law, T, spec, rng)
                    break
                ci = self._uct(node, allowed, lo, hi)
                rewards.append(self._step(obj, domain, st, node.macros[ci], law, rng))
                path.append((node, ci))
                node = node.children[ci]

            # backup: return from each visited node = rewards after it + leaf value
            G = leaf_value
            for depth in range(len(path) - 1, -1, -1):
                G += rewards[depth]
                parent, ci = path[depth]
                child = parent.children[ci]
                child.N += 1
                child.W += G
                parent.N += 1
                if parent is root:
                    root_N[ci, li] += 1
                    root_W[ci, li] += G
                lo, hi = min(lo, G), max(hi, G)

        return self._decide(root, root_N, root_W, weights)

    def _choose_root(self, obj, domain, state, t, T, spec, laws, weights, rng, root,
                     root_N, root_W):
        """UCT over root macros, each simulation = root macro + flat base continuation."""
        m = len(laws)
        lo, hi = np.inf, -np.inf
        for _ in range(self.n_sims):
            if not root.macros:
                break
            li = int(rng.choice(m, p=weights))
            law = laws[li]
            allowed = self._allowed(root)
            untried = [i for i in range(allowed) if i not in root.children]
            ci = untried[0] if untried else self._uct(root, allowed, lo, hi)
            child = root.children.setdefault(ci, _Node())
            # common random numbers: the j-th visit of every root macro shares one stream
            crn = np.random.default_rng([self.seed, t, child.N])
            st = _clone_state(state)
            G = self._step(obj, domain, st, root.macros[ci], law, crn)
            G += self._leaf(obj, domain, st, law, T, spec, crn)
            child.N += 1
            child.W += G
            root.N += 1
            root_N[ci, li] += 1
            root_W[ci, li] += G
            lo, hi = min(lo, G), max(hi, G)
        return self._decide(root, root_N, root_W, weights)

    def _root_prior(self, obj, domain, state, t, T, spec, laws, weights):
        """Root macros ordered by a cheap one-step value under the MAP law.

        Progressive widening only ever expands a prefix of the list, so the
        order decides which macros MCTS can see at all. The prior is the
        rollout planner's estimate with ``M_prior`` paths; the base policy's
        macro is kept first. The (op, region) ranking is reused to order the
        macros of inner nodes.
        """
        macros = self._macros(state, t, T, domain, spec=spec)
        if not macros:
            self._pair_rank = {}
            return macros
        base, rest = macros[-1], macros[:-1]
        law = laws[int(np.argmax(weights))]
        saved_M, self.M = self.M, self.M_prior
        try:
            values = [self._macro_value(obj, domain, state, mac, law, T, spec,
                                        self.seed * 7919 + t) for mac in rest]
        finally:
            self.M = saved_M
        order = np.argsort(-np.asarray(values), kind="stable")
        ranked = [rest[i] for i in order]
        self._pair_rank = {}
        for pos, mac in enumerate(ranked):
            self._pair_rank.setdefault((mac.op, int(mac.region)), pos)
        return [base] + ranked

    def _ordered_macros(self, state, t, T, domain, spec):
        macros = self._macros(state, t, T, domain, spec=spec)
        if not macros:
            return macros
        # _macros appends the base policy's macro last; try it first, then the
        # (op, region) pairs in the order the root prior ranked them
        base, rest = macros[-1], macros[:-1]
        rank = getattr(self, "_pair_rank", {})
        rest = sorted(rest, key=lambda m: (rank.get((m.op, int(m.region)), len(rank)), -int(m.k)))
        return [base] + rest

    def _uct(self, node, allowed, lo, hi):
        total = max(1, sum(node.children[i].N for i in range(allowed) if i in node.children))
        scale = (hi - lo) if np.isfinite(hi - lo) and hi > lo else 1.0
        best, best_u = 0, -np.inf
        for i in range(allowed):
            child = node.children[i]
            q = child.W / max(1, child.N)
            u = q + self.c_uct * scale * math.sqrt(math.log(total + 1) / max(1, child.N))
            if u > best_u:
                best, best_u = i, u
        return best

    def _decide(self, root, root_N, root_W, weights):
        visited = np.flatnonzero(root_N.sum(axis=1) > 0)
        if visited.size == 0:
            return root.macros[0]
        overall = root_W.sum(axis=1) / np.maximum(root_N.sum(axis=1), 1)
        # per-law means; a law never sampled at a child falls back to the child's mean
        per_law = np.where(root_N > 0, root_W / np.maximum(root_N, 1), overall[:, None])
        if self.tree == "root":
            # only macros with enough visits compete (guards against lucky one-offs)
            trusted = visited[root_N[visited].sum(axis=1) >= self.min_visits]
            if trusted.size:
                visited = trusted
        scores = np.asarray(aggregate(per_law[visited], weights, self.mode, self.alpha))
        best = int(visited[int(np.argmax(scores))])
        self.last_info = {
            "macros": [root.macros[i] for i in visited],
            "visits": root_N[visited].sum(axis=1).astype(int).tolist(),
            "per_law_values": per_law[visited],
            "scores": scores,
        }
        return root.macros[best]
