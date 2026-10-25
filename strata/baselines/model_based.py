"""Model-based baselines: fit a law from logs, then play the best flat policy.

All methods in this module share one recipe, differing only in the law they
assume:

``WCDefault``  assume the repository's law (IM: weighted cascade; RPP: IID with
               the default ``profit_factor`` of 0.05).  This is the "no
               learning, no structure" reference.
``LibA``       maximum likelihood over level-1 structures
               (:func:`strata.laws.enumerate.enumerate_structures`, level 1,
               i.e. the independent-cascade family, or the location-free drift
               laws for RPP).
``LibB``       maximum likelihood over level 2 (adds hard gates for IM and
               spatial-region drifts for RPP).
``LibC``       IM: the monotone ``free`` table fitted with weighted PAV;
               RPP: one empirical drift distribution per POI, built from the
               observed profit changes in the log.
``Oracle``     the true law of the instance.  Until the STRATA planner exists
               this is the regret reference: the best flat policy under the
               hidden truth.

Every method then calls :func:`strata.baselines.flat.best_flat` with the fitted
law (the *planning* law) and plays the selected policy under the true law.  The
fitted law and the selected pair are cached per instance, so the fit happens
once even when ``evaluate`` resets the policy for every episode.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from strata.baselines.flat import ALLOCS, best_flat, default_ops
from strata.domains.im import hidden_laws as im_laws
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.eval.protocol import Instance
from strata.laws import fit as law_fit
from strata.laws.enumerate import (
    enumerate_drift_structures,
    enumerate_structures,
    structure_by_name,
)


# ---------------------------------------------------------------------------
# Law fitting helpers
# ---------------------------------------------------------------------------


def fit_im_library(instance: Instance, logs, level: int = 1):
    """MLE over level-``level`` IM structures, selected by training LL."""
    graph = instance.graph
    indeg = graph.indeg
    best = None
    for structure in enumerate_structures(level):
        law = law_fit.fit_program(structure, logs, indeg=indeg, graph=graph)
        ll = law_fit.loglik(law, logs, indeg=indeg, graph=graph)
        if best is None or ll > best[0]:
            best = (ll, law)
    return best[1]


def fit_rpp_library(instance: Instance, logs, level: int = 1):
    """MLE over level-``level`` drift structures, selected by training LL."""
    program, _ = law_fit.fit_drift_library(
        enumerate_drift_structures(level), logs, instance.env
    )
    return program


class EmpiricalDeltas:
    """Per-POI empirical drift: sample from the POI's own observed changes.

    POIs without any observation fall back to the environment's IID uniform
    drift.  The class exposes ``drift(env, state, visited_now, rng)`` and is
    therefore a drop-in RPP law for :mod:`strata.domains.rpp.env`.
    """

    name = "EMPIRICAL"

    def __init__(self, histories: Sequence[np.ndarray], factor: float = 0.05):
        self.histories = [
            np.asarray(h, dtype=np.float64).reshape(-1) for h in histories
        ]
        self.factor = float(factor)

    def drift(self, env, state, visited_now, rng):
        visited = state.visited | np.asarray(visited_now, dtype=bool)
        M, n = state.profit.shape
        out = np.zeros((M, n), dtype=np.float64)
        for j in range(min(n, len(self.histories))):
            rows = np.flatnonzero(~visited[:, j])
            if rows.size == 0:
                continue
            history = self.histories[j]
            if history.size:
                out[rows, j] = rng.choice(history, size=rows.size)
            else:
                out[rows, j] = self.factor * rng.uniform(-1.0, 1.0, rows.size)
        return np.where(visited, 0.0, out)

    def __repr__(self) -> str:
        return "<EmpiricalDeltas n=%d>" % (len(self.histories),)


def empirical_rpp_law(instance: Instance, logs):
    """Build an :class:`EmpiricalDeltas` law from behaviour-log changes."""
    env = instance.env
    n = env.n
    histories: List[List[float]] = [[] for _ in range(n)]
    for episode in logs:
        for round_ in episode:
            before = np.asarray(round_.profit_before, dtype=np.float64)
            after = np.asarray(round_.profit_after, dtype=np.float64)
            visited_after = np.asarray(round_.visited_after, dtype=bool)
            mask = (~visited_after) & (before > 0.0) & (after > env.profit_min + 1e-9)
            for j in np.flatnonzero(mask):
                histories[int(j)].append(float(after[j] - before[j]))
    return EmpiricalDeltas(histories, factor=env.profit_factor)


# ---------------------------------------------------------------------------
# Model-based policies
# ---------------------------------------------------------------------------


class ModelBasedPolicy:
    """Fit a law once per instance, then delegate to ``best_flat``.

    Subclasses implement :meth:`_fit_law`.  ``logs`` are the behaviour episodes
    the law is fitted to (ignored by ``WCDefault`` and ``Oracle``).
    """

    method = "model-based"

    def __init__(
        self,
        logs=None,
        ops: Optional[Sequence[str]] = None,
        allocs: Optional[Sequence[str]] = None,
        selection_episodes: int = 2,
        seed: int = 0,
        period: int = 2,
    ):
        self.logs = logs
        self.ops = tuple(ops) if ops is not None else None
        self.allocs = tuple(allocs) if allocs is not None else ALLOCS
        self.selection_episodes = int(selection_episodes)
        self.seed = int(seed)
        self.period = int(period)

        self.law = None
        self.flat = None
        self._cache_key = None

    # ------------------------------------------------------------- subclasses
    def _fit_law(self, instance: Instance):
        raise NotImplementedError

    # -------------------------------------------------------------- policy API
    def reset(self, instance: Instance, rng: np.random.Generator) -> None:
        if self._cache_key != instance.name:
            self.law = self._fit_law(instance)
            planning = instance.replace(law=self.law)
            ops = self.ops if self.ops is not None else default_ops(instance.domain)
            self.flat = best_flat(
                planning,
                ops=ops,
                allocs=self.allocs,
                episodes=self.selection_episodes,
                seed=self.seed,
                period=self.period,
            )
            self._cache_key = instance.name
        self.flat.reset(instance, rng)

    def act(self, state, t: int) -> np.ndarray:
        if self.flat is None:
            raise RuntimeError("reset(instance, rng) must be called before act")
        return self.flat.act(state, t)

    def observe(self, state_before, action, state_after, info) -> None:
        if self.flat is not None:
            self.flat.observe(state_before, action, state_after, info)

    def __repr__(self) -> str:
        return "<%s law=%s>" % (type(self).__name__, self.law)


class WCDefault(ModelBasedPolicy):
    """Assume the repository default law of the domain."""

    method = "wc-default"

    def _fit_law(self, instance: Instance):
        if instance.domain == "im":
            return im_laws.WeightedCascade()
        return rpp_laws.IID(factor=None, seed=self.seed)


class LibA(ModelBasedPolicy):
    """Level-1 structure library, fitted by maximum likelihood."""

    method = "libA"
    level = 1

    def _fit_law(self, instance: Instance):
        if self.logs is None:
            raise ValueError("LibA needs behaviour logs")
        if instance.domain == "im":
            return fit_im_library(instance, self.logs, level=self.level)
        return fit_rpp_library(instance, self.logs, level=self.level)


class LibB(LibA):
    """Level-2 structure library, fitted by maximum likelihood."""

    method = "libB"
    level = 2


class LibC(ModelBasedPolicy):
    """Rich level-3 representation: free table (IM) / per-POI drift (RPP)."""

    method = "libC"

    def _fit_law(self, instance: Instance):
        if self.logs is None:
            raise ValueError("LibC needs behaviour logs")
        if instance.domain == "im":
            free = structure_by_name("free", level=3)
            return law_fit.fit_free(
                free,
                self.logs,
                indeg=instance.graph.indeg,
                graph=instance.graph,
            )
        return empirical_rpp_law(instance, self.logs)


class Oracle(ModelBasedPolicy):
    """Play the best flat policy under the true law (regret reference)."""

    method = "oracle"

    def _fit_law(self, instance: Instance):
        return instance.law
