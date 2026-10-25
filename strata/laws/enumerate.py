"""Systematic enumeration of the L1 law DSL.

``enumerate_structures(level)`` returns a deterministic list of
:class:`Structure` descriptors.  A structure knows how to build a
:class:`~strata.laws.dsl.LawProgram` from a parameter dictionary and which
parameter grid to try during fitting:

level 1  independent-cascade family (``ic``, ``wc``, ``uni``, ``lt``) with
         instantaneous, REINF-like and DECAY-like memory
level 2  level 1 plus hard gates ``gate(g)`` for ``g in {1..5}``
level 3  level 2 plus soft gates, ramps, two-hop influence, community
         mixtures and a free monotone table

``enumerate_drift_structures(level)`` does the same for the continuous RPP
drift laws: level 1 is location-free Gaussian/uniform/mixture noise, level 2
adds spatial regions, and level 3 adds time/distance/profit covariates.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from strata.laws import dsl

W_GRID = tuple(float(x) for x in np.logspace(-3.0, -0.15, 16))
C_GRID = tuple(float(x) for x in np.logspace(-3.5, -0.5, 16))
ALPHA_GRID = tuple(float(x) for x in np.logspace(-0.7, 0.5, 16))
GATE_GRID = (1, 2, 3, 4, 5)
RHO_GRID = (0.5, 0.9, 1.0)


@dataclass(frozen=True)
class Structure:
    """A law family plus the metadata needed to fit and name it."""

    name: str
    kind: str
    rho: float = 0.0
    two_hop: bool = False
    community: bool = False
    covariates: Tuple[str, ...] = ()
    mixture: int = 1
    region_mu: bool = False
    season_period: float = 10.0
    # Parameters pinned by the structure itself, e.g. (("g", 3),) for gate-3.
    # Named variants must stay distinct hypotheses: if gate-2..gate-5 all
    # scanned every threshold they would collapse onto the same fit and the
    # hypothesis set could never contain the observationally tied gates (K1).
    fixed: Tuple[Tuple[str, float], ...] = ()

    # ----------------------------------------------------------- IM laws
    def param_grid(self):
        grid = self._free_grid()
        if not self.fixed:
            return grid
        pinned = dict(self.fixed)
        keep = [p for p in grid if all(p.get(k) == v for k, v in pinned.items())]
        return keep if keep else [dict(pinned)]

    def _free_grid(self):
        kind = self.kind
        if kind in ("ic", "wc"):
            return [{"w": w} for w in W_GRID]
        if kind == "uni":
            return [{"c": c} for c in C_GRID]
        if kind == "lt":
            return [{"alpha": a} for a in ALPHA_GRID]
        if kind == "gate":
            return [{"g": g} for g in GATE_GRID]
        if kind == "soft_gate":
            return [
                {"threshold": t, "beta": b} for t in (1.5, 2.5) for b in (2.0, 4.0, 8.0)
            ]
        if kind == "ramp":
            return [{"lo": 0.0, "hi": None}, {"lo": 0.5, "hi": None}]
        if kind == "two_hop":
            return [{"c": c} for c in C_GRID]
        if kind == "community":
            return [{"w": w} for w in W_GRID]
        if kind == "mixture":
            return [{"w": w} for w in W_GRID]
        if kind == "free":
            return [{}]
        raise KeyError("unknown structure kind %r" % (kind,))

    def build(self, params=None):
        """Build a :class:`LawProgram` (or :class:`DriftProgram`)."""
        params = dict(params or {})
        if self.kind == "community":
            effect = dsl.Community(
                dsl.WeightedCascade(w=params.get("w", 1.0)),
                dsl.Gate(g=2),
            )
        elif self.kind == "mixture":
            effect = dsl.Mixture(
                [dsl.WeightedCascade(w=params.get("w", 1.0)), dsl.Gate(g=2)],
                weights=[0.7, 0.3],
            )
        elif self.kind == "free":
            effect = dsl.Free(params.get("table", np.array([0.0, 1.0])))
        else:
            effect = dsl.make_effect(self.kind, **params)
        if self.rho:
            effect = dsl.Memory(effect, self.rho)
        if self.community and self.kind != "community":
            effect = dsl.Community(effect, dsl.Gate(g=2))
        return dsl.LawProgram(effect)

    # --------------------------------------------------------- RPP laws
    def build_drift(self, params):
        params = dict(params)
        if self.kind == "mixture":
            params.setdefault("means", [0.0, 0.0])
            params.setdefault("sigmas", [0.05, 0.05])
            params.setdefault("weights", [0.5, 0.5])
        if self.covariates:
            params.setdefault("distance_scale", 100.0)
        effect = dsl.make_drift_effect(self.kind, **params)
        return dsl.DriftProgram(effect)


def enumerate_structures(level=3):
    """Deterministic structure list for ``level`` in ``{1, 2, 3}``."""
    level = int(level)
    if level < 1:
        return []
    base = [
        Structure("ic", "ic"),
        Structure("wc", "wc"),
        Structure("uni", "uni"),
        Structure("lt", "lt"),
        Structure("reinf", "wc", rho=1.0),
        Structure("decay", "uni", rho=0.5),
    ]
    if level == 1:
        return base

    gates = [Structure("gate-%d" % g, "gate", fixed=(("g", g),)) for g in GATE_GRID]
    if level == 2:
        return base + gates

    extra = [
        Structure("soft_gate-1.5", "soft_gate", fixed=(("threshold", 1.5),)),
        Structure("soft_gate-2.5", "soft_gate", fixed=(("threshold", 2.5),)),
        Structure("ramp", "ramp"),
        Structure("two_hop", "two_hop", two_hop=True),
        Structure("community", "community", community=True),
        Structure("mixture", "mixture"),
        Structure("free", "free"),
    ]
    return base + gates + extra


def enumerate_drift_structures(level=3):
    """Deterministic RPP drift structure list for ``level`` in ``{1, 2, 3}``."""
    level = int(level)
    if level < 1:
        return []
    out = [
        Structure("gaussian", "gaussian"),
        Structure("uniform", "uniform"),
        Structure("mixture-2", "mixture", mixture=2, region_mu=False),
    ]
    if level == 1:
        return out
    out += [
        Structure(
            "gaussian-region", "gaussian", covariates=("region",), region_mu=True
        ),
        Structure("uniform-region", "uniform", covariates=("region",), region_mu=True),
    ]
    if level == 2:
        return out
    out += [
        Structure("gaussian-time", "gaussian", covariates=("t",)),
        Structure("gaussian-distance", "gaussian", covariates=("distance",)),
        Structure("gaussian-profit", "gaussian", covariates=("profit",)),
        Structure("gaussian-reversion", "gaussian", covariates=("reversion",)),
        Structure(
            "gaussian-season",
            "gaussian",
            covariates=("season",),
            season_period=10.0,
        ),
        Structure("gaussian-spill", "gaussian", covariates=("spill",)),
        Structure("uniform-reversion", "uniform", covariates=("reversion",)),
        Structure(
            "uniform-season",
            "uniform",
            covariates=("season",),
            season_period=10.0,
        ),
        Structure("uniform-spill", "uniform", covariates=("spill",)),
        Structure(
            "gaussian-full",
            "gaussian",
            covariates=("t", "distance", "profit", "region"),
            region_mu=True,
        ),
        Structure(
            "uniform-full",
            "uniform",
            covariates=("t", "distance", "profit", "region"),
            region_mu=True,
        ),
    ]
    return out


def structure_by_name(name, level=3):
    """Look up an IM structure by name (convenience for experiments/tests)."""
    for s in enumerate_structures(level):
        if s.name == name:
            return s
    raise KeyError(name)


def drift_structure_by_name(name, level=3):
    for s in enumerate_drift_structures(level):
        if s.name == name:
            return s
    raise KeyError(name)
