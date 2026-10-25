"""Flat baselines: one macro-operator plus one fixed budget schedule.

A flat policy is the simplest closed-loop benchmark policy of STRATA.  At every
round it spends ``k`` seeds on nodal picks made by a single L2 operator, where
``k`` follows one of the schedules of ``experiments/kt_law/planner``:

``even``    ``ceil(remaining / rounds_left)`` (dump the rest on the last round)
``normal``  spend everything in the first round
``static``  spend every ``period`` rounds (period 2 by default)

IM operators are ``hub``, ``gate-2``, ``gate-3`` and ``hedge-2``; RPP operators
are ``harvest`` and ``sweep``.  The operators themselves come from
:mod:`strata.abstraction.operators` (their ``refine`` is the verified greedy
port), so a flat policy has no learning component whatsoever.

``best_flat`` selects the best ``(op, alloc)`` pair by mean closed-loop value
under a *planning* law (the instance's law unless a different one is passed).
The two shorthands used in the plan are:

``flat-hub*``  ``ops = {'hub'}`` (IM) / ``{'harvest'}`` (RPP)
``flat-any*``  every operator of the domain

Model-based baselines use ``best_flat`` twice: once with a fitted law to choose
the pair, and then the chosen policy is rolled out under the true law.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from strata.abstraction.library import make_operator
from strata.eval.protocol import Instance, evaluate

IM_OPS = ("hub", "gate-2", "gate-3", "hedge-2")
RPP_OPS = ("harvest", "sweep")
ALLOCS = ("even", "normal", "static")
STATIC_PERIOD = 2


def default_ops(domain: str) -> Tuple[str, ...]:
    """Operator family of a domain (flat-any*)."""
    return IM_OPS if str(domain).lower() == "im" else RPP_OPS


def hub_ops(domain: str) -> Tuple[str, ...]:
    """The flat-hub* family: one hub-like operator per domain."""
    return ("hub",) if str(domain).lower() == "im" else ("harvest",)


def allocation_k(alloc: str, t: int, T: int, remaining: int, period: int = 2) -> int:
    """Budget spent at round ``t``; exact port of ``planner._allocation_k``."""
    remaining = max(0, int(remaining))
    rounds_left = int(T) - int(t)
    if remaining == 0 or rounds_left <= 0:
        return 0
    if alloc == "even":
        if rounds_left == 1:
            return remaining
        return int(min(remaining, np.ceil(remaining / rounds_left)))
    if alloc == "normal":
        return remaining if int(t) == 0 else 0
    if alloc == "static":
        period = max(1, int(period))
        if int(t) % period != 0:
            return 0
        n_spends = int(np.ceil(rounds_left / period))
        return int(min(remaining, np.ceil(remaining / max(1, n_spends))))
    raise KeyError("unknown allocation %r" % (alloc,))


class FlatPolicy:
    """A single operator with a fixed budget schedule (see module docstring).

    ``law`` is the law used by the operator to score candidate picks (the
    *planning* law).  When omitted, the law of the instance passed to
    :meth:`reset` is used, which is what a fully informed flat baseline does.
    The action is always produced for path 0 (``evaluate`` uses ``M = 1``), but
    ``refine`` is called on the whole state so multi-path callers still work.
    """

    def __init__(
        self,
        op: str,
        alloc: str,
        period: int = STATIC_PERIOD,
        law=None,
        region: int = 0,
    ):
        if alloc not in ALLOCS:
            raise KeyError("unknown allocation %r; choose from %s" % (alloc, ALLOCS))
        self.op = str(op)
        self.alloc = str(alloc)
        self.period = int(period)
        self.law = law
        self.region = int(region)
        self.name = "flat-%s-%s" % (self.op, self.alloc)

        self.operator = None
        self.domain = None
        self.instance = None
        self.obj = None
        self.region_mask = None
        self.last_action = None

    # -------------------------------------------------------------- policy API
    def reset(self, instance: Instance, rng: np.random.Generator) -> None:
        self.instance = instance
        self.domain = instance.domain
        self.obj = instance.obj
        self.operator = make_operator(self.op)
        n = instance.graph.adj.shape[0] if self.domain == "im" else instance.env.n
        self.region_mask = np.ones(int(n), dtype=bool)
        self.last_action = None
        self._planning_law = self.law if self.law is not None else instance.law

    def act(self, state, t: int) -> np.ndarray:
        if self.operator is None:
            raise RuntimeError("reset(instance, rng) must be called before act")
        if self.domain == "im":
            remaining = int(getattr(state, "budget_left", 0))
        else:
            budget = np.asarray(state.budget_left).reshape(-1)
            remaining = int(budget[0]) if budget.size else 0
        k = allocation_k(self.alloc, t, self.instance.T, remaining, self.period)
        if k <= 0:
            action = np.zeros(0, dtype=np.int64)
        else:
            seeds = self.operator.refine(
                self.obj, self._planning_law, state, int(k), self.region_mask
            )
            action = np.asarray(seeds, dtype=np.int64).reshape(
                int(np.asarray(seeds).shape[0]), -1
            )[0]
            action = action[action >= 0]
        self.last_action = action
        return action

    def observe(self, state_before, action, state_after, info) -> None:
        return None

    def __repr__(self) -> str:
        return "<FlatPolicy %s>" % (self.name,)


def best_flat(
    instance: Instance,
    ops: Optional[Sequence[str]] = None,
    allocs: Optional[Sequence[str]] = None,
    episodes: int = 2,
    seed: int = 0,
    law=None,
    period: int = STATIC_PERIOD,
) -> FlatPolicy:
    """Select the best ``(op, alloc)`` by mean closed-loop value.

    The candidates are evaluated under ``law`` (default: the instance's law) on
    ``episodes`` closed-loop episodes with common random numbers, exactly the
    protocol used for the final comparison.  The returned policy carries
    ``scores`` (``{(op, alloc): mean value}``), ``mean_value`` and the planning
    law; call :meth:`FlatPolicy.reset` before using it under another instance.
    """
    domain = instance.domain
    ops = tuple(ops) if ops is not None else default_ops(domain)
    allocs = tuple(allocs) if allocs is not None else ALLOCS
    if not ops:
        raise ValueError("best_flat needs at least one operator")
    plan_instance = instance if law is None else replace(instance, law=law)
    planning_law = plan_instance.law

    scores: Dict[Tuple[str, str], float] = {}
    best = None
    for op in ops:
        for alloc in allocs:
            candidate = FlatPolicy(op, alloc, period=period, law=planning_law)
            values = evaluate(candidate, plan_instance, episodes=episodes, seed=seed)
            score = float(np.mean(values))
            scores[(op, alloc)] = score
            if best is None or score > best[0]:
                best = (score, candidate)

    score, policy = best
    policy.mean_value = score
    policy.scores = scores
    policy.planning_law = planning_law
    return policy
