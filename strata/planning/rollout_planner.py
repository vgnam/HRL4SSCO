"""L3 rollout planner: one-step macro lookahead with a flat continuation.

This is the STRATA port of ``experiments/kt_law/ktb_hier.rollout_choice``,
generalised to finite beliefs and to both benchmark domains:

* enumerate the macros of the L2 library (:meth:`OperatorLibrary.macros`) and
  add the base policy's own macro, so the planner can never be worse than the
  base policy in expectation;
* evaluate each macro by grounding it under a hypothesis law, taking one
  environment step on ``M`` parallel paths, and continuing with the base *flat*
  policy (IM: ``hub`` on the whole graph with the best fixed spending schedule;
  RPP: ``harvest`` on region ``"all"`` with the even schedule);
* with ``m`` hypotheses, repeat the evaluation under every law with the same
  common-random-number seed, then scalarise the ``(n_macros, m)`` value matrix
  with :func:`strata.planning.objectives.aggregate` and the belief weights;
* return the best :class:`~strata.core.types.Macro`.

The implementation deliberately mirrors the verified kill test: the CRN stream
for a decision at round ``t`` is ``np.random.default_rng(seed * 1000 + t)`` and
the same stream is used for every candidate.  ``run_episode`` replays one
closed-loop episode under a given true law and returns its total reward, which
makes the planner usable as an ablation/baseline in M3b and beyond.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from strata.abstraction import regions as regions_mod
from strata.abstraction.library import make_operator
from strata.core.types import Macro
from strata.domains.im import env as im_env
from strata.domains.rpp import env as rpp_env
from strata.planning.belief import Belief
from strata.planning.objectives import aggregate

ALLOCATIONS = ("even", "normal", "static")
STATIC_PERIOD = 2
AUTO_EPISODES = 8


def allocation_k(alloc, t, T, remaining, period=STATIC_PERIOD):
    """Budget spent at zero-based round ``t`` (port of ``planner._allocation_k``)."""
    remaining = max(0, int(remaining))
    rounds_left = int(T) - int(t)
    if remaining == 0 or rounds_left <= 0:
        return 0
    if alloc == "even":
        return (
            remaining
            if rounds_left == 1
            else int(min(remaining, np.ceil(remaining / rounds_left)))
        )
    if alloc == "normal":
        return remaining if int(t) == 0 else 0
    if alloc == "static":
        period = max(1, int(period))
        if int(t) % period != 0:
            return 0
        n_spends = int(np.ceil(rounds_left / period))
        return int(min(remaining, np.ceil(remaining / max(1, n_spends))))
    raise KeyError("unknown allocation %r" % (alloc,))


def _clone_state(state):
    return state.clone() if hasattr(state, "clone") else state


def _row(array):
    array = np.asarray(array)
    return array[None, :] if array.ndim == 1 else array[:1]


def _expand_im(state, M):
    return type(state)(
        status=np.repeat(_row(state.status), M, axis=0).astype(np.int8),
        E=np.repeat(_row(state.E), M, axis=0).astype(np.float32),
        t=int(state.t),
        budget_left=int(state.budget_left),
    )


def _expand_rpp(state, M):
    current = np.asarray(state.current).reshape(-1)
    budget = np.asarray(state.budget_left).reshape(-1)
    return type(state)(
        profit=np.repeat(_row(state.profit), M, axis=0).copy(),
        visited=np.repeat(_row(state.visited), M, axis=0).copy(),
        current=np.full(M, int(current[0]), dtype=np.int64),
        budget_left=np.full(M, int(budget[0]), dtype=np.int64),
        t=int(state.t),
    )


def _remaining(state, domain):
    if domain == "im":
        return max(0, int(state.budget_left))
    budget = np.asarray(state.budget_left).reshape(-1)
    return max(0, int(budget[0])) if budget.size else 0


def _used(action, domain):
    action = np.asarray(action)
    if action.size == 0:
        return 0
    valid = action >= 0
    if domain == "im":
        return int(valid.sum(axis=1).max()) if valid.ndim > 1 else int(valid.sum())
    lengths = valid.sum(axis=1)
    return int(lengths[0]) if lengths.size else 0


def _trim_actions(actions, budgets):
    """Drop trailing POIs of every path that exceed its own remaining budget."""
    A = np.asarray(actions)
    if A.size == 0 or A.ndim < 2:
        return A
    A = A.copy()
    for m in range(A.shape[0]):
        valid = np.flatnonzero(A[m] >= 0)
        limit = max(0, int(budgets[m]))
        if valid.size > limit:
            A[m, valid[limit:]] = -1
    return A


class RolloutPlanner:
    """One-step lookahead over macros with a flat-policy continuation.

    Parameters
    ----------
    library
        :class:`~strata.abstraction.library.OperatorLibrary`; its ``macros``
        method enumerates the candidates.
    base
        Base flat policy / budget schedule.  ``None`` or ``"auto"`` selects the
        domain default (IM: ``hub`` with the best of ``even|normal|static``
        under the MAP hypothesis; RPP: ``harvest`` on ``"all"`` with ``even``).
        A string is read as an allocation schedule; a dict may carry ``alloc``,
        ``op``, ``region`` and ``period``; a ``FlatPolicy``-like object is read
        through its ``op``/``alloc``/``region`` attributes.
    M
        Number of Monte-Carlo paths used to estimate a macro's value.
    mode
        Aggregation mode of :func:`strata.planning.objectives.aggregate`.
    seed
        Base seed.  The CRN stream at decision round ``t`` is
        ``np.random.default_rng(seed * 1000 + t)``.
    T, alpha, period, n_eval
        Optional overrides of the horizon, CVaR level, static period and the
        number of episodes used by the automatic schedule choice.  ``T`` is
        also read from the RPP environment or from ``base.instance.T``.
    regions
        Optional explicit region list ``[("all", mask), ...]``.  When omitted,
        IM uses the graph communities and RPP uses spatial clusters.
    """

    def __init__(
        self,
        library,
        base=None,
        M=48,
        mode="mean",
        seed=0,
        T=None,
        alpha=0.2,
        period=STATIC_PERIOD,
        n_eval=AUTO_EPISODES,
        regions=None,
    ):
        self.library = library
        self.base = base
        self.M = max(1, int(M))
        self.mode = str(mode).lower()
        self.seed = int(seed)
        self.T = None if T is None else int(T)
        self.alpha = float(alpha)
        self.period = int(period)
        self.n_eval = max(1, int(n_eval))
        self.regions = regions
        self._explicit_regions = regions is not None
        self._regions_obj = None
        self._auto_cache = {}

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _domain(state):
        if hasattr(state, "status"):
            return "im"
        if hasattr(state, "profit"):
            return "rpp"
        raise TypeError("cannot infer the domain of state %r" % (type(state).__name__,))

    def _resolve_T(self, obj):
        if self.T is not None:
            return self.T
        if hasattr(obj, "T"):
            return int(obj.T)
        base = self.base
        instance = getattr(base, "instance", None)
        if instance is not None and hasattr(instance, "T"):
            return int(instance.T)
        raise ValueError("the planner needs T (pass T=... or use an RPP env)")

    def _ensure_regions(self, obj):
        if self._explicit_regions:
            return self.regions
        if self._regions_obj is obj and self.regions is not None:
            return self.regions
        if hasattr(obj, "adj"):
            self.regions = regions_mod.make_regions(obj, k_top=3)
        elif hasattr(obj, "loc"):
            self.regions = regions_mod.make_spatial_regions(
                obj, k_top=3, seed=self.seed
            )
        else:
            raise TypeError("cannot build regions for %r" % (type(obj).__name__,))
        self._regions_obj = obj
        return self.regions

    def _operator(self, name):
        try:
            return self.library.get(name)
        except (KeyError, AttributeError):
            return make_operator(name)

    def _base_spec(self, obj, domain, state):
        spec = {
            "op": "hub" if domain == "im" else "harvest",
            "region": 0,
            "alloc": "auto" if domain == "im" else "even",
            "period": self.period,
        }
        base = self.base
        if base is None:
            return spec
        if isinstance(base, str):
            key = base.lower()
            if key in ALLOCATIONS or key == "auto":
                spec["alloc"] = key
            elif domain == "rpp" and key in ("harvest", "sweep", "reposition"):
                spec["op"] = key
            else:
                spec["alloc"] = key
            return spec
        if isinstance(base, dict):
            spec.update(
                {k: base[k] for k in ("op", "region", "alloc", "period") if k in base}
            )
            return spec
        for key in ("op", "alloc", "region", "period"):
            if hasattr(base, key):
                spec[key] = getattr(base, key)
        instance = getattr(base, "instance", None)
        if self.T is None and instance is not None and hasattr(instance, "T"):
            self.T = int(instance.T)
        return spec

    def _belief(self, belief):
        if belief is None:
            raise ValueError("RolloutPlanner.choose needs a belief")
        if hasattr(belief, "laws"):
            laws = list(belief.laws)
            weights = np.asarray(belief.weights, dtype=np.float64)
            return laws, weights
        items = getattr(belief, "hyps", belief)
        laws = [getattr(h, "law", h) for h in items]
        weights = np.asarray(belief.weights(), dtype=np.float64)
        return laws, weights

    # --------------------------------------------------------- macro search
    def _macros(self, state, t, T, domain, spec=None):
        costs = _remaining(state, domain)
        n_regions = len(self._ensure_regions_for_state(state, domain))
        zeros = tuple(0 for _ in range(n_regions))
        z = (int(t), int(costs)) + zeros
        macros = self.library.macros(
            z,
            b_left=costs,
            t=int(t),
            T=T,
            n_regions=n_regions,
            one_based_t=(domain == "rpp"),
        )
        if spec is None:
            spec = self._base_spec(None, domain, state)
        t0 = int(t) if domain == "im" else int(t) - 1
        k_base = allocation_k(
            spec["alloc"], t0, T, costs, int(spec.get("period", self.period))
        )
        macros = list(macros) + [
            Macro(str(spec["op"]), int(spec["region"]), int(k_base))
        ]

        out = []
        seen = set()
        for macro in macros:
            key = (macro.op, int(macro.region), int(macro.k))
            if key in seen:
                continue
            seen.add(key)
            out.append(macro)
        return out

    def _ensure_regions_for_state(self, state, domain):
        # ``choose`` always calls ``_ensure_regions`` first; this guard keeps
        # ``_macros`` usable on its own with an existing region list.
        if self.regions is None:
            raise ValueError("regions must be resolved before enumerating macros")
        return self.regions

    def _macro_value(self, obj, domain, state, macro, law, T, spec, rng_seed):
        M = self.M
        st = _expand_rpp(state, M) if domain == "rpp" else _expand_im(state, M)
        mask = regions_mod.region_mask(self.regions, macro.region)
        op = self._operator(macro.op)
        costs = _remaining(state, domain)
        k = max(0, min(int(macro.k), costs))
        action = op.refine(obj, law, st, k, mask)
        rng = np.random.default_rng(int(rng_seed))
        if domain == "im":
            _, reward, _ = im_env.step(obj, law, st, action, rng)
        else:
            _, reward, _, _ = rpp_env.step(
                obj, st, action, law=law, rng=rng, return_info=True
            )
        total = np.asarray(reward, dtype=np.float64).copy()
        rem = costs - _used(action, domain)
        total += self._continue_flat(obj, domain, st, law, rem, T, spec, rng)
        return float(total.mean())

    def _continue_flat(self, obj, domain, state, law, remaining, T, spec, rng):
        M = state.status.shape[0] if domain == "im" else state.profit.shape[0]
        alloc = spec["alloc"]
        period = int(spec.get("period", self.period))
        op = self._operator(spec["op"])
        mask = regions_mod.region_mask(self.regions, int(spec["region"]))
        total = np.zeros(M, dtype=np.float64)
        rem = max(0, int(remaining))

        if domain == "im":
            for t in range(int(state.t), int(T)):
                k = min(allocation_k(alloc, t, T, rem, period), rem)
                seeds = op.refine(obj, law, state, k, mask)
                _, reward, _ = im_env.step(obj, law, state, seeds, rng)
                total += reward
                rem -= _used(seeds, domain)
                if rem <= 0:
                    rem = 0
            return total

        while int(state.t) <= int(T):
            budgets = np.maximum(np.asarray(state.budget_left, dtype=np.int64), 0)
            if budgets.size == 0:
                break
            ks = [
                min(
                    allocation_k(alloc, int(state.t) - 1, T, int(budgets[m]), period),
                    int(budgets[m]),
                )
                for m in range(budgets.size)
            ]
            k = max(ks) if ks else 0
            # Paths drift apart, so their remaining budgets can differ; refine
            # with the largest affordable k and trim every path to its own.
            action = op.refine(obj, law, state, int(k), mask)
            action = _trim_actions(action, budgets)
            _, reward, done, _ = rpp_env.step(
                obj, state, action, law=law, rng=rng, return_info=True
            )
            total += reward
            if done:
                break
        return total

    # ------------------------------------------------------- auto allocation
    def _auto_alloc(self, obj, domain, state, belief):
        """Best of ``even|normal|static`` for the base hub/harvest policy."""
        T = self._resolve_T(obj)
        if domain != "im":
            return "even"
        key = (id(obj), int(T), _remaining(state, domain))
        if key in self._auto_cache:
            return self._auto_cache[key]
        laws, weights = self._belief(belief)
        law = laws[int(np.argmax(weights))]
        scores = {}
        for alloc in ALLOCATIONS:
            values = [
                self._flat_episode(obj, domain, state, law, alloc, T, 20261024 + e)
                for e in range(self.n_eval)
            ]
            scores[alloc] = float(np.mean(values))
        best = ALLOCATIONS[0]
        for alloc in ALLOCATIONS[1:]:
            if scores[alloc] > scores[best] + 1e-12:
                best = alloc
        self._auto_cache[key] = best
        return best

    def _flat_episode(self, obj, domain, state, law, alloc, T, ep_seed):
        st = _clone_state(state)
        rem = _remaining(st, domain)
        rng = np.random.default_rng(int(ep_seed))
        spec = self._base_spec(obj, domain, st)
        spec["alloc"] = alloc
        op = self._operator(spec["op"])
        mask = regions_mod.region_mask(self.regions, int(spec["region"]))
        total = 0.0
        if domain == "im":
            for t in range(int(st.t), int(T)):
                k = min(allocation_k(alloc, t, T, rem, self.period), rem)
                seeds = op.refine(obj, law, st, k, mask)
                _, reward, _ = im_env.step(obj, law, st, seeds, rng)
                total += float(reward[0])
                rem = max(0, rem - _used(seeds, domain))
            return total
        while int(st.t) <= int(T):
            k = min(allocation_k(alloc, int(st.t) - 1, T, rem, self.period), rem)
            action = op.refine(obj, law, st, k, mask)
            _, reward, done, _ = rpp_env.step(
                obj, st, action, law=law, rng=rng, return_info=True
            )
            total += float(reward[0])
            rem = max(0, rem - _used(action, domain))
            if done:
                break
        return total

    # ------------------------------------------------------------- Planner API
    def choose(self, env_or_graph, state, belief, t):
        """Return the macro with the best aggregated one-step lookahead value."""
        domain = self._domain(state)
        obj = env_or_graph
        T = self._resolve_T(obj)
        regions = self._ensure_regions(obj)
        spec = self._base_spec(obj, domain, state)
        if str(spec["alloc"]).lower() == "auto":
            spec["alloc"] = self._auto_alloc(obj, domain, state, belief)
        laws, weights = self._belief(belief)
        macros = self._macros(state, int(t), T, domain, spec=spec)
        values = np.zeros((len(macros), len(laws)), dtype=np.float64)
        rng_seed = self.seed * 1000 + int(t)
        for li, law in enumerate(laws):
            for ci, macro in enumerate(macros):
                values[ci, li] = self._macro_value(
                    obj, domain, state, macro, law, T, spec, rng_seed
                )
        scores = np.asarray(
            aggregate(values, weights, self.mode, self.alpha), dtype=np.float64
        )
        best = 0
        best_value = float(scores[0])
        for i in range(1, scores.size):
            if float(scores[i]) > best_value + 1e-9:
                best, best_value = i, float(scores[i])
        self.last_scores = scores
        self.last_macros = macros
        return macros[best]

    # ------------------------------------------------------------- evaluation
    def run_episode(
        self,
        env_or_graph,
        belief,
        true_law=None,
        T=None,
        budget=None,
        seed=0,
    ):
        """Play one closed-loop episode under ``true_law`` and return its reward.

        The planner is queried at every round with the same ``belief`` (it is
        *not* updated inside this helper, so the call is a pure evaluation of a
        fixed planner/belief pair).  Actions are grounded under ``true_law``;
        the environment is also stepped under ``true_law``.
        """
        obj = env_or_graph
        domain = "im" if hasattr(obj, "adj") else "rpp"
        if T is not None:
            self.T = int(T)
        T = self._resolve_T(obj)
        self.seed = int(seed)
        laws, weights = self._belief(belief)
        law = true_law if true_law is not None else laws[int(np.argmax(weights))]
        rng = np.random.default_rng(int(seed))

        if domain == "im":
            state = im_env.make_state(
                obj, M=1, budget_left=int(20 if budget is None else budget)
            )
        else:
            state = obj.reset(M=1)
            if budget is not None:
                state.budget_left[:] = int(budget)

        total = 0.0
        while True:
            if domain == "im" and int(state.t) >= T:
                break
            if domain == "rpp" and int(state.t) > T:
                break
            t = int(state.t)
            macro = self.choose(obj, state, belief, t)
            op = self._operator(macro.op)
            mask = regions_mod.region_mask(self.regions, macro.region)
            action = op.refine(obj, law, state, int(macro.k), mask)
            if domain == "im":
                _, reward, _ = im_env.step(obj, law, state, action, rng)
                total += float(reward[0])
            else:
                _, reward, done, _ = rpp_env.step(
                    obj, state, action, law=law, rng=rng, return_info=True
                )
                total += float(reward[0])
                if done:
                    break
        return total
