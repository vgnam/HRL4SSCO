"""Hidden activation laws of the IM benchmark.

The first eight classes are ports of ``experiments/kt_law/laws.py`` -- they
must produce exactly the same probabilities as the verified simulator.  The
remaining four are the richer benchmark structures from the STRATA plan:

``SOFT2``  smooth gate: ``p = sigmoid(4 * (a - 1.5))`` for ``a >= 1``
``MIX``    30% gate-2 nodes, the rest weighted cascade (mask is hidden)
``COMM``   weighted cascade in half of the Louvain communities, CC2 in the rest
``HET``    node types WC / CC2 / LT in proportions 0.5 / 0.3 / 0.2

All laws implement the ``Law`` protocol of :mod:`strata.core.types`: they
receive a :class:`Features` bundle and return an ``(M, n)`` probability matrix.
The environment applies the eligibility mask and the Bernoulli sampling.
"""

import numpy as np

from strata.core.rng import make_rng


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class IMLaw:
    """Base class with the shared weighted-cascade / exposure machinery."""

    name = "law"
    rho = 0.0
    two_hop = False

    def prob(self, feats):
        raise NotImplementedError

    def n_params(self):
        return 0

    def edge_weight(self, indeg):
        """Per-target weight used by the repository's ``score`` heuristic."""
        return 1.0 / np.maximum(indeg, 1)

    def __repr__(self):
        return "<im-law %s>" % self.name


class WeightedCascade(IMLaw):
    name = "WC"

    def prob(self, feats):
        w = 1.0 / np.maximum(feats.indeg, 1)
        return 1.0 - np.power(1.0 - w, feats.E)

    def n_params(self):
        return 1


class Uniform(WeightedCascade):
    name = "UNI"
    c = 0.05

    def prob(self, feats):
        return 1.0 - np.power(1.0 - self.c, feats.E)

    def edge_weight(self, indeg):
        return np.full_like(np.asarray(indeg, dtype=np.float64), self.c)


class LinearThreshold(IMLaw):
    name = "LT"

    def prob(self, feats):
        return np.clip(feats.a / np.maximum(feats.indeg, 1), 0.0, 1.0)

    def n_params(self):
        return 1


class Gate(IMLaw):
    name = "CC2"
    gate = 2

    def prob(self, feats):
        return (feats.a >= self.gate).astype(np.float32)

    def n_params(self):
        return 1


class Gate3(Gate):
    name = "CC3"
    gate = 3


class Reinfection(WeightedCascade):
    """Exposure accumulates across rounds; old influence never fades."""

    name = "REINF"
    rho = 1.0


class Decay(WeightedCascade):
    """Exposure decays with ``rho = 0.5`` and uses the uniform constant."""

    name = "DECAY"
    rho = 0.5
    c = 0.05

    def prob(self, feats):
        return 1.0 - np.power(1.0 - self.c, feats.E)

    def edge_weight(self, indeg):
        return np.full_like(np.asarray(indeg, dtype=np.float64), self.c)

    def n_params(self):
        return 2


class TwoHop(Uniform):
    name = "2HOP"
    two_hop = True


class Soft2(IMLaw):
    """Smooth complex-contagion gate with a soft threshold at ``a = 1.5``."""

    name = "SOFT2"
    beta = 4.0
    threshold = 1.5

    def prob(self, feats):
        p = _sigmoid(self.beta * (feats.a - self.threshold))
        return np.where(feats.a >= 1.0, p, 0.0).astype(np.float32)


class Mix(IMLaw):
    """Mixture of gate-2 and weighted cascade over two hidden node types."""

    name = "MIX"

    def __init__(self, n=None, gate_frac=0.3, gate=2, seed=0, node_type=None):
        self.gate_frac = float(gate_frac)
        self.gate = int(gate)
        if node_type is not None:
            self.gate_mask = np.asarray(node_type).astype(bool).copy()
        elif n is not None:
            rng = make_rng("im-mix", int(seed), int(n), self.gate_frac, self.gate)
            self.gate_mask = rng.random(int(n)) < self.gate_frac
        else:
            self.gate_mask = None

    def prob(self, feats):
        w = 1.0 / np.maximum(feats.indeg, 1)
        wc = 1.0 - np.power(1.0 - w, feats.E)
        gate = (feats.a >= self.gate).astype(np.float32)
        if self.gate_mask is None:
            raise ValueError("MIX needs an instance size or a node_type mask")
        return np.where(self.gate_mask[None, :], gate, wc)

    def n_params(self):
        return 2


class CommunityLaw(IMLaw):
    """Weighted cascade in the first half of communities, CC2 in the second."""

    name = "COMM"

    def __init__(self, community=None, wc_fraction=0.5, gate=2):
        self.gate = int(gate)
        self.wc_fraction = float(wc_fraction)
        self.wc_mask = None
        if community is not None:
            self.wc_mask = self._mask(np.asarray(community))

    def _mask(self, community):
        ids = np.unique(community)
        if ids.size == 0:
            return np.zeros(0, dtype=bool)
        n_wc = int(np.ceil(self.wc_fraction * ids.size))
        n_wc = min(max(n_wc, 1), ids.size)
        return np.isin(community, ids[:n_wc])

    def prob(self, feats):
        w = 1.0 / np.maximum(feats.indeg, 1)
        wc = 1.0 - np.power(1.0 - w, feats.E)
        gate = (feats.a >= self.gate).astype(np.float32)
        mask = self.wc_mask
        if mask is None or mask.size != feats.a.shape[1]:
            mask = self._mask(np.asarray(feats.community))
        return np.where(mask[None, :], wc, gate)


class Heterogeneous(IMLaw):
    """Three hidden node types: WC / CC2 / LT with proportions 0.5/0.3/0.2."""

    name = "HET"

    def __init__(self, n=None, seed=0, node_type=None, proportions=(0.5, 0.3, 0.2)):
        if node_type is not None:
            self.node_type = np.asarray(node_type, dtype=np.int64).copy()
        elif n is not None:
            self.node_type = self._draw_types(int(n), seed, proportions)
        else:
            self.node_type = None

    @staticmethod
    def _draw_types(n, seed, proportions):
        n0 = int(round(proportions[0] * n))
        n1 = int(round(proportions[1] * n))
        n0 = min(max(n0, 0), n)
        n1 = min(max(n1, 0), n - n0)
        types = np.concatenate(
            [
                np.zeros(n0, dtype=np.int64),
                np.ones(n1, dtype=np.int64),
                np.full(n - n0 - n1, 2, dtype=np.int64),
            ]
        )
        rng = make_rng("im-het", int(seed), int(n))
        return types[rng.permutation(n)]

    def prob(self, feats):
        if self.node_type is None:
            raise ValueError("HET needs an instance size or a node_type array")
        w = 1.0 / np.maximum(feats.indeg, 1)
        wc = 1.0 - np.power(1.0 - w, feats.E)
        cc2 = (feats.a >= 2).astype(np.float32)
        lt = np.clip(feats.a / np.maximum(feats.indeg, 1), 0.0, 1.0)
        t = self.node_type[None, :]
        return np.where(t == 0, wc, np.where(t == 1, cc2, lt))


class Fractional(IMLaw):
    """Fractional threshold (Centola): activate iff a_v >= ceil(theta * indeg_v).

    Outside the DSL: the threshold scales with each node's in-degree, so no
    fixed gate g and no IC-family law reproduces it.
    """

    name = "FRAC"

    def __init__(self, theta=0.3):
        self.theta = float(theta)

    def prob(self, feats):
        need = np.maximum(np.ceil(self.theta * np.asarray(feats.indeg, dtype=np.float64)), 1.0)
        return (feats.a >= need[None, :]).astype(np.float32)

    def n_params(self):
        return 1


class Fatigue(IMLaw):
    """Weighted cascade damped by past exposure: repeated exposure backfires.

    ``E = a + E_prev`` (rho = 1) so ``E - a`` is the exposure accumulated in
    earlier rounds; p = WC(a) * exp(-lam * (E - a)). Outside the DSL, whose
    memory wrapper can only make old exposure *help*.
    """

    name = "FATIGUE"
    rho = 1.0

    def __init__(self, lam=0.7):
        self.lam = float(lam)

    def prob(self, feats):
        w = 1.0 / np.maximum(feats.indeg, 1)
        p_now = 1.0 - np.power(1.0 - w, feats.a)
        past = np.maximum(np.asarray(feats.E, dtype=np.float64) - feats.a, 0.0)
        return (p_now * np.exp(-self.lam * past)).astype(np.float64)

    def n_params(self):
        return 2


# Ported mechanisms (verified bit-for-bit against kt_law/sim.py in M0 tests).
PORTED_LAWS = ("WC", "UNI", "LT", "CC2", "CC3", "REINF", "DECAY", "2HOP")
# Laws written so that no DSL structure expresses them (where the LLM must help).
OUTSIDE_DSL = ("FRAC", "FATIGUE")
# Full benchmark family used by the STRATA experiments.
LAW_NAMES = PORTED_LAWS + ("SOFT2", "MIX", "COMM", "HET") + OUTSIDE_DSL

_CLASSES = {
    "WC": WeightedCascade,
    "UNI": Uniform,
    "LT": LinearThreshold,
    "CC2": Gate,
    "CC3": Gate3,
    "REINF": Reinfection,
    "DECAY": Decay,
    "2HOP": TwoHop,
    "SOFT2": Soft2,
    "MIX": Mix,
    "COMM": CommunityLaw,
    "HET": Heterogeneous,
    "FRAC": Fractional,
    "FATIGUE": Fatigue,
}


def make_law(name, graph=None, seed=0, **params):
    """Instantiate a hidden law, wiring graph-dependent hidden masks if given."""
    key = str(name).upper()
    if key not in _CLASSES:
        raise KeyError("unknown IM law %r; choose from %s" % (name, list(LAW_NAMES)))
    cls = _CLASSES[key]
    # MIX and HET draw their hidden node types from `seed` unless `node_type` is
    # passed explicitly; graph.node_type is not read implicitly, because one
    # array cannot serve both laws (MIX is two-valued, HET three-valued).
    if key in ("MIX", "HET"):
        if "n" not in params and "node_type" not in params and graph is not None:
            params["n"] = graph.adj.shape[0]
        params.setdefault("seed", seed)
    elif key == "COMM":
        if "community" not in params and graph is not None:
            params["community"] = graph.community
    return cls(**params)


def get_law(name, graph=None, seed=0, **params):
    """Alias of :func:`make_law` (mirrors the kt_law helper name)."""
    return make_law(name, graph=graph, seed=seed, **params)
