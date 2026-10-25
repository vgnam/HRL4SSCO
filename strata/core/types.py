"""Interface contract for STRATA (section 2 of the implementation plan).

This module is the single source of truth for the data structures exchanged
between layers.  M0 writes it; later milestones must not change the contract
(they may import from it freely).  Additions should be reported to the
integrator instead of silently editing this file.

Conventions
-----------
* ``M`` is the number of independent Monte-Carlo paths run in parallel; all
  per-node quantities are therefore ``(M, n)`` arrays unless stated otherwise.
* ``n`` is the number of nodes/POIs in the instance.
* Optional annotations use ``typing.Optional`` because the package targets
  Python 3.8.
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np
import scipy.sparse as sp


# Node states of the IM environment (same values as IM/Env.py and kt_law/sim.py).
STATUS_INACTIVE, STATUS_ACTIVE, STATUS_REMOVED = 0, 1, 2


@dataclass
class Graph:
    """Immutable directed graph of one IM instance.

    ``adj[u, v] = 1`` iff the edge ``u -> v`` exists.  ``adjT`` is its
    transpose.  ``community`` is a Labelled-Louvain partition of the underlying
    undirected graph.  ``node_type`` is only used by hidden laws that mix
    mechanisms (HET/MIX); it is deliberately hidden from the learning agent.
    """

    adj: sp.csr_matrix
    adjT: sp.csr_matrix
    indeg: np.ndarray
    outdeg: np.ndarray
    community: np.ndarray
    node_type: Optional[np.ndarray] = None
    name: str = ""


@dataclass
class SimState:
    """State of ``M`` parallel IM paths.

    ``status`` is ``(M, n)`` int8 with values in
    ``{STATUS_INACTIVE, STATUS_ACTIVE, STATUS_REMOVED}``.  ``E`` is the
    accumulated exposure of every node, ``(M, n)`` float32.  ``t`` is the
    number of rounds already played and ``budget_left`` the remaining number of
    seeds (a single integer because the IM environment shares the seed set
    across the ``M`` paths).
    """

    status: np.ndarray
    E: np.ndarray
    t: int
    budget_left: int

    def clone(self) -> "SimState":
        return SimState(self.status.copy(), self.E.copy(), self.t, self.budget_left)


class Law(Protocol):
    """A hidden activation law.

    ``prob`` receives a :class:`Features` bundle and returns the per-node
    activation probability, shape ``(M, n)``.  ``rho`` is the per-round decay
    of the exposure accumulator (``0`` means instantaneous influence) and
    ``two_hop`` says whether influence travels two hops per round.
    """

    name: str
    rho: float
    two_hop: bool

    def prob(self, feats: "Features") -> np.ndarray: ...

    def n_params(self) -> int: ...


@dataclass
class Features:
    """Per-round features computed by ``laws/features.py`` (or the environment).

    ``a`` -- number of incoming active neighbours this round (weighted by the
    law's influence operator); ``E`` -- exposure after accumulation; ``a2`` --
    optional explicit two-hop count; ``eligible`` -- nodes that may activate;
    ``indeg``/``community`` are static ``(n,)`` arrays.
    """

    a: np.ndarray
    E: np.ndarray
    a2: Optional[np.ndarray]
    eligible: np.ndarray
    indeg: np.ndarray
    community: np.ndarray


@dataclass
class Round:
    """One logged round of an IM episode (``M = 1``; arrays are ``(n,)``).

    ``y`` is the set of nodes that activated this round; ``seeded`` marks the
    seeds of the round; ``before``/``after`` are status snapshots.
    """

    a: np.ndarray
    eligible: np.ndarray
    y: np.ndarray
    seeded: np.ndarray
    before: np.ndarray
    after: np.ndarray
    policy: str


Episode = List[Round]


@dataclass
class Hypothesis:
    """A fitted law together with its evidence for the hypothesis set."""

    law: Law
    loglik: float
    prior: float
    tie_group: int


@dataclass
class HypothesisSet:
    """Hypotheses kept by L1 plus pairwise discrimination counts.

    ``discriminating[i, j]`` is the number of logged (node, round) events on
    which hypotheses ``i`` and ``j`` predict materially different behaviour.
    """

    hyps: List[Hypothesis]
    discriminating: np.ndarray

    def weights(self) -> np.ndarray:
        """Normalised prior weights (uniform fallback for degenerate priors)."""
        if not self.hyps:
            return np.zeros(0, dtype=np.float64)
        prior = np.asarray([h.prior for h in self.hyps], dtype=np.float64)
        prior = np.maximum(prior, 0.0)
        total = float(prior.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full(prior.size, 1.0 / prior.size, dtype=np.float64)
        return prior / total


class Operator(Protocol):
    """A macro-operator of the L2 abstraction (IM version)."""

    name: str  # 'hub', 'gate-2', 'hedge-2', 'prime', 'wait'

    def precondition(self, z: "AbstractState", k: int, region: int) -> bool: ...

    def refine(
        self, graph: Graph, law: Law, state: SimState, k: int, region_mask: np.ndarray
    ) -> np.ndarray: ...

    def checker(
        self, before: SimState, after: SimState, region_mask: np.ndarray
    ) -> np.ndarray: ...


# Hashable abstract state: (t, budget_left, *bucket ids per region).
AbstractState = Tuple


@dataclass(frozen=True)
class Macro:
    """A grounded macro action: operator, region and number of seeds."""

    op: str
    region: int
    k: int


class Planner(Protocol):
    """L3 planner interface."""

    def choose(
        self, graph: Graph, state: SimState, belief: "Belief", t: int
    ) -> Macro: ...
