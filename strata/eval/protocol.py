"""Benchmark protocol of STRATA: instances, behaviour logs, closed-loop evaluation.

This module is the common ground on which every baseline and (later) the STRATA
planner are compared.  It provides four things:

``Instance``   one benchmark cell: a domain (``im``/``rpp``), a dataset (or a
               synthetic RPP instance), an instance size, a graph/point seed, a
               hidden law and the horizon ``T`` and budget ``B``.  The concrete
               graph / environment is built lazily and cached on the instance.
``make_logs``  behaviour episodes collected under the hidden law through the
               builders in :mod:`strata.domains.im.logs` and
               :mod:`strata.domains.rpp.logs`.
``evaluate``   closed-loop value of a *policy* under the true law of the
               instance, with common random numbers (CRN) across policies: the
               environment noise of episode ``i`` depends only on the outer
               seed and ``i``, so two policies evaluated with the same seed see
               paired trajectories and their per-episode values are directly
               comparable.
``regret``     relative regret of a value vector against the oracle, with a
               paired standard error (episode pairing).

Policy interface
----------------
Other milestones (planners, the STRATA agent) are written in parallel and must
not be imported here.  A policy is any object with::

    reset(instance, rng) -> None
    act(state, t) -> action
    observe(state_before, action, state_after, info) -> None

Actions are ``(k,)`` arrays of node ids for IM (seed sets) and ordered POI
routes ``(k,)`` for RPP.  ``act`` may also return an ``(M, k)`` matrix for
multi-path states; ``evaluate`` uses ``M = 1`` and flattens its action.

For RPP the benchmark uses ``RPPEnv(empty_day_advances=True, ...)`` by default
(the STRATA semantics, where an empty day still drifts profits and advances the
clock); pass ``empty_day_advances=False`` on the instance to recover the
original ``RPP/Env.py`` semantics.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from strata.config import LogConfig
from strata.core.rng import make_rng
from strata.domains.im import env as im_env
from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.domains.im import logs as im_logs
from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import logs as rpp_logs
from strata.domains.rpp import points as rpp_points

DOMAINS = ("im", "rpp")


class Policy:
    """Structural base class documenting the policy interface.

    Policies do not have to inherit from it (duck typing is enough), but the
    default ``observe`` makes stateless policies trivial to write.
    """

    def reset(self, instance: "Instance", rng: np.random.Generator) -> None:
        raise NotImplementedError

    def act(self, state: Any, t: int) -> np.ndarray:
        raise NotImplementedError

    def observe(self, state_before, action, state_after, info) -> None:
        return None


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    """One benchmark cell (see the module docstring).

    ``dataset`` is an IM dataset name (``dblp``, ``wiki``, ...) or an RPP
    dataset (``flickr-Edin``, ...); ``None`` means a synthetic RPP instance of
    ``n`` uniform points.  ``graph_seed`` seeds both the subgraph sampling and
    the synthetic points, so the instance is reproducible.

    ``env_options`` is forwarded to :class:`strata.domains.rpp.env.RPPEnv`
    (e.g. ``profit_factor``, ``max_daily_distance``); it is unused for IM.
    """

    domain: str
    dataset: Optional[str] = None
    n: int = 100
    graph_seed: int = 0
    law: Any = None
    T: int = 10
    B: int = 20
    sample_method: Optional[str] = None
    empty_day_advances: bool = True
    env_options: Dict[str, Any] = field(default_factory=dict)
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.domain = str(self.domain).lower()
        if self.domain not in DOMAINS:
            raise ValueError(
                "unknown domain %r; choose from %s" % (self.domain, DOMAINS)
            )
        self.n = int(self.n)
        self.T = int(self.T)
        self.B = int(self.B)
        self.graph_seed = int(self.graph_seed)

    # ------------------------------------------------------------------ names
    @property
    def dataset_name(self) -> str:
        return "random" if self.dataset is None else str(self.dataset)

    @property
    def law_name(self) -> str:
        if self.law is None:
            return "none"
        return str(getattr(self.law, "name", type(self.law).__name__))

    @property
    def name(self) -> str:
        return "%s-%s-n%d-s%d-%s" % (
            self.domain,
            self.dataset_name,
            self.n,
            self.graph_seed,
            self.law_name,
        )

    # -------------------------------------------------------------- building
    def build(self) -> Dict[str, Any]:
        """Build (once) and return the concrete graph/environment bundle."""
        if self._cache:
            return self._cache
        if self.domain == "im":
            if self.dataset is None:
                raise ValueError("an IM instance needs a dataset name")
            method = self.sample_method or "rw"
            graph = im_graph.sample_instance(
                self.dataset, self.n, self.graph_seed, method=method
            )
            self._cache = {"obj": graph, "graph": graph}
            return self._cache

        method = self.sample_method or "knn"
        if self.dataset is None:
            loc, initial, factor = rpp_points.random_instance(
                self.n, seed=self.graph_seed
            )
            venues = None
        else:
            loc, initial, factor, venues = rpp_points.make_instance_with_venues(
                self.dataset, n=self.n, method=method, seed=self.graph_seed
            )
        options = dict(self.env_options)
        options.setdefault("profit_factor", factor)
        env = rpp_env.RPPEnv(
            loc=loc,
            initial_profit=initial,
            budget=self.B,
            T=self.T,
            seed=self.graph_seed,
            empty_day_advances=self.empty_day_advances,
            **options,
        )
        self._cache = {
            "obj": env,
            "env": env,
            "loc": loc,
            "initial_profit": initial,
            "profit_factor": factor,
            "venues": venues,
        }
        return self._cache

    # ------------------------------------------------------------- accessors
    @property
    def obj(self):
        return self.build()["obj"]

    @property
    def graph(self):
        if self.domain != "im":
            raise AttributeError("graph is only defined for IM instances")
        return self.build()["graph"]

    @property
    def env(self):
        if self.domain != "rpp":
            raise AttributeError("env is only defined for RPP instances")
        return self.build()["env"]

    def replace(self, **changes) -> "Instance":
        """Shallow copy with changed fields (the concrete cache is shared)."""
        cloned = dataclasses.replace(self, **changes)
        cloned._cache = self._cache
        return cloned


def make_instance(
    domain: str,
    dataset: Optional[str] = None,
    n: int = 100,
    graph_seed: int = 0,
    law: Any = "CC2",
    T: int = 10,
    B: int = 20,
    sample_method: Optional[str] = None,
    empty_day_advances: bool = True,
    env_options: Optional[Dict[str, Any]] = None,
    **law_params,
) -> Instance:
    """Build an :class:`Instance` and (when needed) its hidden law.

    ``law`` may be a law object or a law name.  Names are resolved through the
    domain factory *after* the graph/environment exists, which is required by
    ``MIX``/``HET``/``COMM`` (IM) and ``FITTED`` (RPP).
    """
    instance = Instance(
        domain=domain,
        law=None,
        n=n,
        dataset=dataset,
        graph_seed=graph_seed,
        T=T,
        B=B,
        sample_method=sample_method,
        empty_day_advances=empty_day_advances,
        env_options=dict(env_options or {}),
    )
    cache = instance.build()
    if isinstance(law, str):
        key = law.upper()
        if instance.domain == "im":
            instance.law = im_laws.make_law(
                key, graph=cache["graph"], seed=graph_seed, **law_params
            )
        elif key == "FITTED" and dataset is not None:
            instance.law = rpp_laws.make_law(
                "FITTED",
                dataset=dataset,
                venues=cache.get("venues"),
                seed=graph_seed,
                **law_params,
            )
        else:
            instance.law = rpp_laws.make_law(key, seed=graph_seed, **law_params)
    else:
        instance.law = law
    return instance


# ---------------------------------------------------------------------------
# Behaviour logs
# ---------------------------------------------------------------------------


def default_log_policies(domain: str) -> List[str]:
    cfg = LogConfig()
    if str(domain).lower() == "im":
        return list(cfg.im_policies)
    return list(cfg.rpp_policies)


def make_logs(
    instance: Instance,
    policies: Optional[Sequence[str]] = None,
    n_episodes: int = 100,
    seed: int = 0,
):
    """Collect behaviour episodes under the instance's hidden law.

    The returned structure is domain specific: a list of IM episodes
    (``List[Round]``) or RPP episodes (``List[RPPRound]``).  Both are accepted
    by :mod:`strata.laws.fit` and by the model-based baselines.
    """
    if policies is None:
        policies = default_log_policies(instance.domain)
    if isinstance(policies, str):
        policies = [policies]
    policies = [str(p) for p in policies]
    if instance.domain == "im":
        return im_logs.collect_logs(
            instance.graph,
            instance.law,
            T=instance.T,
            B=instance.B,
            n_episodes=int(n_episodes),
            policies=policies,
            seed=seed,
        )
    return rpp_logs.collect_logs(
        instance.env,
        instance.law,
        T=instance.T,
        budget=instance.B,
        n_episodes=int(n_episodes),
        policies=policies,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Closed-loop evaluation
# ---------------------------------------------------------------------------


def _as_action(action) -> np.ndarray:
    """Flatten a policy action to valid int64 ids (``-1`` padding removed)."""
    a = np.asarray(action if action is not None else [], dtype=np.int64).reshape(-1)
    return a[a >= 0]


def _eval_im(policy, instance: Instance, env_rng: np.random.Generator) -> float:
    graph = instance.graph
    state = im_env.make_state(graph, M=1, budget_left=instance.B)
    total = 0.0
    for t in range(instance.T):
        action = _as_action(policy.act(state, t))
        before = state.clone()
        _, reward, info = im_env.step(
            graph, instance.law, state, action, env_rng, return_info=True
        )
        total += float(np.asarray(reward, dtype=np.float64).reshape(-1)[0])
        policy.observe(before, action, state, info)
    return total


def _eval_rpp(policy, instance: Instance, env_rng: np.random.Generator) -> float:
    env = instance.env
    state = env.reset(M=1)
    total = 0.0
    for t in range(instance.T):
        action = _as_action(policy.act(state, t))
        before = state.clone()
        state, reward, done, info = rpp_env.step(
            env, state, action, law=instance.law, rng=env_rng, return_info=True
        )
        total += float(np.asarray(reward, dtype=np.float64).reshape(-1)[0])
        policy.observe(before, action, state, info)
        if done:
            break
    return total


def evaluate(
    policy, instance: Instance, episodes: int = 1, seed: int = 0
) -> np.ndarray:
    """Closed-loop episode totals of ``policy`` under the true law.

    Common random numbers: the environment stream of episode ``i`` is
    ``make_rng("strata-eval-env", seed, i)`` and the policy stream is
    ``make_rng("strata-eval-policy", seed, i)``.  Keeping them separate is what
    makes the pairing valid even when two policies consume different numbers of
    random draws.  The result has shape ``(episodes,)`` and is deterministic
    given ``seed``.
    """
    episodes = int(episodes)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    totals = np.zeros(episodes, dtype=np.float64)
    for ep in range(episodes):
        env_rng = make_rng("strata-eval-env", int(seed), ep)
        policy_rng = make_rng("strata-eval-policy", int(seed), ep)
        policy.reset(instance, policy_rng)
        if instance.domain == "im":
            totals[ep] = _eval_im(policy, instance, env_rng)
        else:
            totals[ep] = _eval_rpp(policy, instance, env_rng)
    return totals


def mean_value(values) -> float:
    """Mean of per-episode totals (the closed-loop value of a policy)."""
    return float(np.mean(np.asarray(values, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Regret
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regret:
    """Relative regret with a paired standard error.

    ``mean`` is ``mean(oracle - value) / mean(oracle)`` (positive = worse than
    the oracle).  ``se`` is the standard error of the *paired* per-episode
    differences, rescaled by the mean oracle value, so it reflects the fact that
    the two value vectors come from the same random numbers.
    """

    mean: float
    se: float
    mean_value: float
    mean_oracle: float
    n: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "regret": self.mean,
            "regret_se": self.se,
            "value": self.mean_value,
            "oracle": self.mean_oracle,
            "n": self.n,
        }


def regret(values, oracle_values) -> Regret:
    """Relative paired regret of ``values`` against ``oracle_values``."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    o = np.asarray(oracle_values, dtype=np.float64).reshape(-1)
    if v.size != o.size:
        raise ValueError(
            "values and oracle_values must have the same length (%d != %d)"
            % (v.size, o.size)
        )
    if v.size == 0:
        raise ValueError("regret needs at least one episode")
    mean_oracle = float(o.mean())
    diff = o - v
    if abs(mean_oracle) <= 1e-12:
        mean_regret = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    else:
        mean_regret = float(diff.mean() / mean_oracle)
        scale = abs(mean_oracle)
        se = float(diff.std(ddof=1) / np.sqrt(v.size) / scale) if v.size > 1 else 0.0
    return Regret(
        mean=mean_regret,
        se=se,
        mean_value=float(v.mean()),
        mean_oracle=mean_oracle,
        n=int(v.size),
    )
