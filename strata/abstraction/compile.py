"""Lazily compiled stochastic macro model ``P_hat(z', r | z, macro, law)``.

``CompiledModel`` is the L2/L3 bridge used by MCTS: instead of simulating the
concrete environment for every macro at every node, it

1. **pools concrete states by abstract state** (:meth:`build_pool`).  Rollouts
   are driven by every hypothesis law, alternating a base flat policy (``hub``
   for IM, ``harvest`` for RPP, with the ``even`` budget schedule) and random
   macros drawn from the operator library.  Every visited concrete state is
   keyed by its :meth:`Abstraction.encode` tuple, with reservoir sampling
   capped at ``pool_cap`` states per ``z``.
2. **compiles a macro on demand** (:meth:`sample`).  For ``(z, macro, law)`` it
   takes one pooled concrete state, tiles it to ``n_samples`` Monte-Carlo
   paths, refines the macro under that law and steps the *real* domain
   environment.  The empirical joint distribution of ``(z', r)`` is cached, so
   later calls only resample from the cache.
3. exposes :meth:`expected_reward`, :meth:`outcome_distribution` and
   :meth:`stats` for planners, diagnostics and reporting.

If ``z`` is absent from the pool, :meth:`sample` raises ``KeyError`` so the
caller can fall back to concrete simulation (the M3b contract).  Everything is
deterministic given ``seed``; the only external randomness is the ``rng``
passed explicitly to :meth:`sample`/:meth:`simulate`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from strata.abstraction.library import even_allocation, k_options, make_operator
from strata.config import AbstractionConfig, IMConfig, RPPConfig
from strata.core.rng import make_rng
from strata.core.types import Macro, SimState
from strata.domains.im import env as im_env
from strata.domains.rpp import env as rpp_env


def _laws_from(hyps) -> List[Any]:
    """Accept a HypothesisSet, a list of hypotheses or a list of laws."""
    if hyps is None:
        return []
    if hasattr(hyps, "prob"):  # a single law
        return [hyps]
    items = hyps.hyps if hasattr(hyps, "hyps") else hyps
    return [getattr(h, "law", h) for h in items]


def _clone_state(state):
    clone = getattr(state, "clone", None)
    if clone is None:
        raise TypeError("domain state must provide clone()")
    return clone()


def _macro_parts(macro) -> Tuple[str, int, int]:
    """Normalise a :class:`Macro` (or a 3-tuple) to ``(op, region, k)``."""
    if hasattr(macro, "op"):
        return str(macro.op), int(macro.region), int(macro.k)
    op, region, k = macro
    return str(op), int(region), int(k)


class CompiledModel:
    """Compiled stochastic macro model for one instance and hypothesis set."""

    def __init__(
        self,
        domain: str,
        env_or_graph,
        hyps,
        abstraction,
        library,
        n_samples: int = None,
        seed: int = 0,
        T: int = None,
        budget: int = None,
        pool_cap: int = 8,
    ):
        self.domain = str(domain).lower()
        if self.domain not in ("im", "rpp"):
            raise ValueError("unknown domain %r" % (domain,))
        self.obj = env_or_graph
        self.hyps = hyps
        self.laws = _laws_from(hyps)
        if not self.laws:
            raise ValueError("CompiledModel needs at least one hypothesis/law")
        self.abstraction = abstraction
        self.library = library
        if n_samples is None:
            n_samples = AbstractionConfig().compile_samples
        self.n_samples = int(n_samples)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self.seed = int(seed)
        self.pool_cap = max(1, int(pool_cap))

        if self.domain == "im":
            cfg = IMConfig()
            self.T = int(cfg.T if T is None else T)
            self.budget = int(cfg.budget if budget is None else budget)
        else:
            cfg = RPPConfig()
            self.T = int(getattr(env_or_graph, "T", cfg.T) if T is None else T)
            self.budget = int(
                getattr(env_or_graph, "budget", cfg.budget)
                if budget is None
                else budget
            )

        # z -> list of concrete states (SimState / RPPState).
        self.pool: Dict[Tuple, List[Any]] = {}
        # (z, (op, region, k), law_idx) -> (z_next tuple, rewards array).
        self._cache: Dict[Tuple, Tuple[Tuple, np.ndarray]] = {}
        self._hits = 0
        self._misses = 0
        self._pool_seen: Dict[Tuple, int] = {}
        self._pool_rng = make_rng("strata-compile-record", self.seed, self.domain)

    # ------------------------------------------------------------- utilities
    @property
    def n_laws(self) -> int:
        return len(self.laws)

    def _operator(self, name):
        if self.library is not None:
            try:
                return self.library.get(name)
            except KeyError:
                pass
        return make_operator(name)

    def _base_operator(self):
        name = "hub" if self.domain == "im" else "harvest"
        return self._operator(name)

    def _record(self, state):
        """Add a concrete state to the pool (reservoir sampling per ``z``)."""
        z = tuple(self.abstraction.encode(self.obj, state)[0])
        seen = self._pool_seen.get(z, 0) + 1
        self._pool_seen[z] = seen
        states = self.pool.setdefault(z, [])
        if len(states) < self.pool_cap:
            states.append(_clone_state(state))
            return
        slot = int(self._pool_rng.integers(seen))
        if slot < self.pool_cap:
            states[slot] = _clone_state(state)

    def macros(self, z) -> List[Macro]:
        """Macro options for ``z`` under the abstraction's region layout.

        ``strata.abstraction.library.OperatorLibrary.macros`` infers the number
        of regions as ``len(z) - 2``, which assumes one bucket id per region;
        the L2 abstraction stores several bucketed features per region, so the
        region count is taken from the abstraction instead.  Everything else
        (the ``even`` schedule and the operator set) comes from the library.
        """
        z = tuple(z)
        t = int(z[0]) if len(z) > 0 else 0
        b_left = int(z[1]) if len(z) > 1 else 0
        t_eff = t if self.domain == "im" else t - 1
        opts = k_options(b_left, t_eff, self.T)
        if self.library is not None:
            operators = [op for op in self.library.operators if op.name != "wait"]
            has_wait = any(op.name == "wait" for op in self.library.operators)
        else:
            operators = [self._base_operator()]
            has_wait = False
        out: List[Macro] = []
        if 0 in opts and has_wait:
            out.append(Macro("wait", 0, 0))
        for k in opts:
            if k <= 0:
                continue
            for op in operators:
                for region in range(self.abstraction.n_regions):
                    out.append(Macro(op.name, region, int(k)))
        return out

    def _random_macro(self, state, rng):
        """Uniform random macro from the library for the current abstract state."""
        z = tuple(self.abstraction.encode(self.obj, state)[0])
        candidates = self.macros(z)
        if candidates:
            return candidates[int(rng.integers(len(candidates)))]
        op = self._base_operator()
        k = even_allocation(self._budget_left(state), int(self._day(state)), self.T)
        return Macro(op.name, 0, k)

    def _day(self, state) -> int:
        """Zero-based day index (RPP states are 1-based, IM states are 0-based)."""
        return int(state.t) if self.domain == "im" else int(state.t) - 1

    def _budget_left(self, state) -> int:
        if self.domain == "im":
            return int(state.budget_left)
        return int(np.asarray(state.budget_left).reshape(-1)[0])

    # ------------------------------------------------------------ pool build
    def build_pool(
        self, n_rollouts: int = 8, policies: Sequence[str] = ("base", "random")
    ):
        """Populate the state pool by rolling out under every hypothesis law.

        ``'base'`` uses the domain's flat policy (``hub``/``harvest`` with the
        ``even`` schedule) and ``'random'`` draws a macro uniformly from the
        operator library.  Rollouts cycle through the laws; the policy advances
        only after a full law cycle, so every law is exercised by every policy.
        The pool, cache and counters are reset first, making repeated calls
        deterministic.
        """
        policies = list(policies) if policies is not None else ["base"]
        if not policies:
            raise ValueError("at least one pool policy is required")
        n_rollouts = int(n_rollouts)
        self.pool = {}
        self._cache = {}
        self._hits = 0
        self._misses = 0
        self._pool_seen = {}
        self._pool_rng = make_rng("strata-compile-record", self.seed, self.domain)

        for i in range(n_rollouts):
            law_idx = i % self.n_laws
            law = self.laws[law_idx]
            # Decorrelate the policy cycle from the law cycle so that every
            # law is exercised by both the flat and the random policy.
            policy = str(policies[(i // self.n_laws) % len(policies)])
            rng = make_rng(
                "strata-compile-rollout", self.seed, self.domain, int(i), int(law_idx)
            )
            if self.domain == "im":
                self._rollout_im(law, policy, rng)
            else:
                self._rollout_rpp(law, policy, rng)
        return self

    def _rollout_im(self, law, policy, rng):
        state = im_env.make_state(self.obj, M=1, budget_left=self.budget)
        for _ in range(self.T):
            self._record(state)
            if policy == "base":
                op = self._base_operator()
                region = 0
                k = even_allocation(int(state.budget_left), int(state.t), self.T)
            else:
                op_name, region, k = _macro_parts(self._random_macro(state, rng))
                op = self._operator(op_name)
            mask = self.abstraction.region_mask(region)
            seeds = op.refine(self.obj, law, state, k, mask)
            im_env.step(self.obj, law, state, seeds, rng)
        self._record(state)

    def _rollout_rpp(self, law, policy, rng):
        state = self.obj.reset(M=1)
        for _ in range(self.T):
            self._record(state)
            if policy == "base":
                op = self._base_operator()
                region = 0
                k = even_allocation(self._budget_left(state), self._day(state), self.T)
            else:
                op_name, region, k = _macro_parts(self._random_macro(state, rng))
                op = self._operator(op_name)
            mask = self.abstraction.region_mask(region)
            action = op.refine(self.obj, law, state, k, mask)
            rpp_env.step(self.obj, state, action, law=law, rng=rng)
        self._record(state)

    # --------------------------------------------------------------- tiling
    def _tile(self, state, n_samples: int):
        """Replicate one concrete state into ``n_samples`` identical paths."""
        M = int(n_samples)
        if self.domain == "im":
            status = np.asarray(state.status)
            E = np.asarray(state.E)
            rows = np.zeros(M, dtype=np.int64)
            return SimState(
                status=status[rows].copy(),
                E=E[rows].copy(),
                t=int(state.t),
                budget_left=int(state.budget_left),
            )
        profit = np.asarray(state.profit)
        visited = np.asarray(state.visited)
        current = np.asarray(state.current).reshape(-1)
        budget = np.asarray(state.budget_left).reshape(-1)
        rows = np.zeros(M, dtype=np.int64)
        return rpp_env.RPPState(
            profit=profit[rows].copy(),
            visited=visited[rows].copy(),
            current=current[rows].copy(),
            budget_left=budget[rows].copy(),
            t=int(state.t),
        )

    # ------------------------------------------------------------ simulation
    def simulate(self, state, macro, law_idx: int, rng, n_samples: int = None):
        """Direct simulation of ``macro`` from one concrete ``state``.

        Returns ``(state_after, z_next tuple, rewards (M,))`` with ``M =
        n_samples``.  This is the concrete fallback and the reference the
        compiled cache is built from; it never touches the pool or the cache.
        """
        law_idx = int(law_idx)
        if law_idx < 0 or law_idx >= self.n_laws:
            raise IndexError("law_idx %d out of range" % (law_idx,))
        law = self.laws[law_idx]
        op_name, region, k = _macro_parts(macro)
        op = self._operator(op_name)
        M = self.n_samples if n_samples is None else int(n_samples)
        tiled = self._tile(state, M)
        mask = self.abstraction.region_mask(region)
        action = op.refine(self.obj, law, tiled, k, mask)
        if self.domain == "im":
            tiled, reward, _ = im_env.step(self.obj, law, tiled, action, rng)
        else:
            tiled, reward, _ = rpp_env.step(self.obj, tiled, action, law=law, rng=rng)
        z_next = tuple(self.abstraction.encode(self.obj, tiled))
        return tiled, z_next, np.asarray(reward, dtype=np.float64)

    # ---------------------------------------------------------------- sample
    def _cache_key(self, z, macro, law_idx):
        op, region, k = _macro_parts(macro)
        return (tuple(z), (op, int(region), int(k)), int(law_idx))

    def sample(self, z, macro, law_idx: int, rng):
        """Draw one empirical outcome ``(z', reward)`` from ``P_hat(z',r|...)``.

        Raises ``KeyError`` when ``z`` was never observed by :meth:`build_pool`;
        the caller should then fall back to concrete simulation.  The first
        call for a key simulates ``n_samples`` paths and caches the empirical
        joint distribution; later calls resample from the cache.
        """
        z = tuple(z)
        key = self._cache_key(z, macro, law_idx)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            z_next, rewards = cached
            idx = int(rng.integers(rewards.size))
            return z_next[idx], float(rewards[idx])

        if z not in self.pool:
            raise KeyError(z)
        self._misses += 1
        states = self.pool[z]
        pick = int(rng.integers(len(states)))
        _, z_next, rewards = self.simulate(states[pick], macro, law_idx, rng)
        self._cache[key] = (z_next, rewards)
        idx = int(rng.integers(rewards.size))
        return z_next[idx], float(rewards[idx])

    # -------------------------------------------------------------- queries
    def _cached(self, z, macro, law_idx):
        key = self._cache_key(tuple(z), macro, law_idx)
        cached = self._cache.get(key)
        if cached is None:
            raise KeyError(key)
        return cached

    def expected_reward(self, z, macro, law_idx: int) -> float:
        """Mean cached reward of ``(z, macro, law_idx)`` (``KeyError`` if absent)."""
        _, rewards = self._cached(z, macro, law_idx)
        return float(np.mean(rewards))

    def outcome_distribution(self, z, macro, law_idx: int) -> Dict[Tuple, float]:
        """Empirical marginal ``P_hat(z' | z, macro, law)`` over next states."""
        z_next, _ = self._cached(z, macro, law_idx)
        counts: Dict[Tuple, int] = {}
        for item in z_next:
            counts[item] = counts.get(item, 0) + 1
        total = float(len(z_next))
        return {item: count / total for item, count in counts.items()}

    def stats(self) -> Dict[str, int]:
        """Cache/pool sizes and cache hit/miss counters."""
        return {
            "cache_size": len(self._cache),
            "pool_size": int(sum(len(v) for v in self.pool.values())),
            "n_abstract_states": len(self.pool),
            "hits": int(self._hits),
            "misses": int(self._misses),
        }
