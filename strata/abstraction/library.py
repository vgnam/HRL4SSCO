"""Operator library: macro enumeration, validation and JSON persistence.

``OperatorLibrary`` owns the set of macro-operators available to the planner.
:meth:`OperatorLibrary.macros` expands an abstract state into concrete
:class:`strata.core.types.Macro` choices using the ``even`` budget schedule
``k in {0, even, 2*even, 4*even, all}``.  :meth:`OperatorLibrary.validate`
implements the M2 acceptance rule: an operator is kept when its checker holds
with probability at least ``tau`` under at least one hypothesis *and* it is not
dominated by another library operator (some scenario where the short-rollout
value differs by more than ``delta``).

A :class:`Scenario` is the minimal context needed to roll an operator forward
once: the domain (``'im'``/``'rpp'``), the graph/environment, a concrete state,
the region mask and the budget ``k``.  Validation is deliberately
domain-agnostic: the domain selects the environment used for the one-step
rollout.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np

from strata.abstraction.operators import (
    Gate,
    Harvest,
    Hedge,
    Hub,
    Prime,
    Reposition,
    Sweep,
    Wait,
)
from strata.config import AbstractionConfig
from strata.core.rng import make_rng
from strata.core.types import Macro
from strata.domains.im import env as im_env
from strata.domains.rpp import env as rpp_env

_FORMAT_VERSION = 1

_REGISTRY = {
    cls.__name__: cls
    for cls in (Hub, Gate, Hedge, Prime, Wait, Harvest, Sweep, Reposition)
}


def make_operator(name, **params):
    """Instantiate an operator from its class name or macro name."""
    key = str(name)
    if key in _REGISTRY:
        return _REGISTRY[key](**params)
    # Macro names: 'hub', 'gate-2', 'hedge-3', 'wait', ...
    base = key.split("-")[0]
    for cls in _REGISTRY.values():
        if getattr(cls, "kind", None) == base or getattr(cls, "name", None) == key:
            if base in ("gate", "hedge") and "gate" not in params:
                parts = key.split("-")
                if len(parts) > 1 and parts[1]:
                    params["gate"] = int(parts[1])
            return cls(**params)
    raise KeyError("unknown operator %r" % (name,))


def _operator_spec(op):
    return {
        "type": type(op).__name__,
        "name": op.name,
        "params": dict(getattr(op, "params", {})),
    }


def _operator_from_spec(spec):
    cls_name = spec["type"]
    if cls_name not in _REGISTRY and cls_name.startswith("LLM"):
        # LLM operators register themselves on import (they are rebuilt from source)
        from strata.abstraction import llm_operators  # noqa: F401
    if cls_name not in _REGISTRY:
        raise ValueError("unknown operator type %r" % (cls_name,))
    return _REGISTRY[cls_name](**dict(spec.get("params", {})))


@dataclass
class Scenario:
    """One concrete context in which a macro is rolled forward once."""

    domain: str
    obj: Any  # Graph for IM, RPPEnv for RPP
    state: Any  # SimState or RPPState
    region_mask: np.ndarray
    k: int


def _as_scenario(item):
    if isinstance(item, Scenario):
        return item
    if isinstance(item, dict):
        return Scenario(
            str(item["domain"]).lower(),
            item["obj"],
            item["state"],
            np.asarray(item["region_mask"], dtype=bool),
            int(item["k"]),
        )
    if isinstance(item, (tuple, list)) and len(item) == 4:
        obj = item[0]
        domain = "im" if hasattr(obj, "adj") else "rpp"
        return Scenario(
            domain, obj, item[1], np.asarray(item[2], dtype=bool), int(item[3])
        )
    if isinstance(item, (tuple, list)) and len(item) == 5:
        return Scenario(
            str(item[0]).lower(),
            item[1],
            item[2],
            np.asarray(item[3], dtype=bool),
            int(item[4]),
        )
    raise TypeError("cannot interpret %r as a validation scenario" % (item,))


def _laws_from(hyps):
    if hyps is None:
        return []
    if hasattr(hyps, "prob"):  # a single law
        return [hyps]
    items = hyps.hyps if hasattr(hyps, "hyps") else hyps
    laws = []
    for h in items:
        laws.append(getattr(h, "law", h))
    return laws


def even_allocation(b_left, t, T):
    """Even schedule: ``ceil(b_left / rounds_left)`` (the plan's ``even``)."""
    rounds_left = max(int(T) - int(t), 1)
    if b_left <= 0:
        return 0
    return int(min(int(np.ceil(float(b_left) / rounds_left)), int(b_left)))


def k_options(b_left, t, T):
    """``{0, even, 2*even, 4*even, all}`` clipped to the remaining budget."""
    b_left = max(0, int(b_left))
    even = even_allocation(b_left, t, T)
    opts = {0, b_left}
    for mult in (1, 2, 4):
        opts.add(min(mult * even, b_left))
    return sorted(int(v) for v in opts)


class OperatorLibrary:
    """A list of macro-operators with macro expansion, validation and I/O."""

    def __init__(self, operators: Optional[Iterable] = None):
        self.operators = list(operators) if operators is not None else []

    # ------------------------------------------------------------- container
    def __len__(self):
        return len(self.operators)

    def __iter__(self):
        return iter(self.operators)

    def __getitem__(self, index):
        return self.operators[index]

    def names(self):
        return [op.name for op in self.operators]

    def get(self, name):
        for op in self.operators:
            if op.name == name:
                return op
        raise KeyError("operator %r not in library" % (name,))

    def add(self, op):
        if op.name not in self.names():
            self.operators.append(op)
        return op

    def remove(self, name):
        self.operators = [op for op in self.operators if op.name != name]

    @classmethod
    def standard(cls, domain="im", gates=(2, 3)):
        """The default operator set for a domain (used by M3 and baselines)."""
        if domain == "im":
            ops = [Hub(), Prime(), Wait()]
            ops += [Gate(g) for g in gates]
            ops += [Hedge(g) for g in gates]
            return cls(ops)
        if domain == "rpp":
            return cls([Harvest(), Sweep(), Reposition(), Wait()])
        raise ValueError("unknown domain %r" % (domain,))

    # ---------------------------------------------------------------- macros
    def macros(self, z, b_left=None, t=None, T=None, n_regions=None, one_based_t=None):
        """Enumerate ``Macro(op, region, k)`` for abstract state ``z``.

        ``z`` follows the contract ``(t, budget_left, *bucket ids per region)``.
        Pass ``n_regions`` explicitly when the abstraction stores several
        bucketed features per region (the L2 ``Abstraction`` does); otherwise it
        is inferred as ``len(z) - 2`` (one bucket per region). ``t`` is
        zero-based for IM and one-based for RPP (``RPPState.t``);
        ``one_based_t`` defaults to ``True`` for the RPP library.
        ``b_left``/``t``/``T`` override the values stored in ``z`` when given.
        """
        z = tuple(z)
        if t is None:
            t = int(z[0]) if len(z) > 0 else 0
        if b_left is None:
            b_left = int(z[1]) if len(z) > 1 else 0
        if T is None:
            raise ValueError("T is required to build the even schedule")
        if one_based_t is None:
            rpp_ops = {"harvest", "sweep", "reposition"}
            one_based_t = any(op.name in rpp_ops for op in self.operators)
        if one_based_t:
            t = t - 1
        if n_regions is None:
            n_regions = max(1, len(z) - 2)

        out: List[Macro] = []
        opts = k_options(b_left, t, T)
        if 0 in opts and any(op.name == "wait" for op in self.operators):
            out.append(Macro("wait", 0, 0))
        for k in opts:
            if k <= 0:
                continue
            for op in self.operators:
                if op.name == "wait":
                    continue
                for region in range(n_regions):
                    out.append(Macro(op.name, region, k))
        return out

    # ------------------------------------------------------------ validation
    def validate(self, op, hyps, states, tau=None, delta=None, seed=0):
        """Keep ``op`` iff its checker holds and no operator dominates it.

        ``hyps`` is a :class:`HypothesisSet`, a list of hypotheses or a list of
        laws.  ``states`` is a sequence of :class:`Scenario` (or equivalent
        tuples/dicts).  Rollouts use common random numbers across operators for
        a given scenario and law.
        """
        cfg = AbstractionConfig()
        tau = cfg.checker_tau if tau is None else float(tau)
        delta = cfg.non_dominance_delta if delta is None else float(delta)
        laws = _laws_from(hyps)
        if not laws:
            raise ValueError("validate needs at least one hypothesis or law")
        scenarios = [_as_scenario(s) for s in states]
        if not scenarios:
            return False

        ops = list(self.operators)
        if not any(o is op for o in ops):
            ops.append(op)

        values = {}
        checker_pass = False
        for si, scenario in enumerate(scenarios):
            for li, law in enumerate(laws):
                for oi, other in enumerate(ops):
                    value, after = _rollout(
                        scenario, law, other, seed=seed, si=si, li=li
                    )
                    values[(si, li, oi)] = value
                    if other is op and after is not None:
                        ok = other.checker(scenario.state, after, scenario.region_mask)
                        if float(np.mean(np.asarray(ok, dtype=float))) >= tau:
                            checker_pass = True

        if not checker_pass:
            return False

        op_index = next(oi for oi, other in enumerate(ops) if other is op)
        others = [(oi, other) for oi, other in enumerate(ops) if other is not op]
        if not others:
            return True
        for other_index, other in others:
            distinguishable = False
            for si in range(len(scenarios)):
                for li in range(len(laws)):
                    a = values[(si, li, op_index)]
                    b = values[(si, li, other_index)]
                    if np.isfinite(a) and np.isfinite(b) and abs(a - b) > delta:
                        distinguishable = True
                        break
                if distinguishable:
                    break
            if not distinguishable:
                return False
        return True

    # ------------------------------------------------------------------- I/O
    def to_dict(self):
        return {
            "version": _FORMAT_VERSION,
            "operators": [_operator_spec(op) for op in self.operators],
        }

    @classmethod
    def from_dict(cls, data):
        version = int(data.get("version", _FORMAT_VERSION))
        if version != _FORMAT_VERSION:
            raise ValueError("unsupported operator-library version %r" % (version,))
        return cls([_operator_from_spec(spec) for spec in data.get("operators", [])])

    def save(self, path):
        path = os.fspath(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path):
        with open(os.fspath(path), "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _rollout(scenario, law, op, seed=0, si=0, li=0):
    """One macro step under ``law``; returns ``(mean reward, after state)``."""
    rng = make_rng("strata-operator-validate", int(seed), int(si), int(li))
    state = scenario.state.clone()
    seeds = op.refine(scenario.obj, law, state, int(scenario.k), scenario.region_mask)
    domain = str(scenario.domain).lower()
    if domain == "im":
        _, reward, _ = im_env.step(scenario.obj, law, state, seeds, rng)
    elif domain == "rpp":
        _, reward, _ = rpp_env.step(scenario.obj, state, seeds, law=law, rng=rng)
    else:
        raise ValueError("unknown validation domain %r" % (scenario.domain,))
    return float(np.mean(reward)), state
