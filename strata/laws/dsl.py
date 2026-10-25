"""The L1 law DSL (OneLife effects, product laws, RPP drift laws).

An IM law program is a pair ``(precondition, effect)``.  The effect maps the
sufficient statistics of a node to an activation probability; a precondition
is an optional mask restricting where the effect applies.  The DSL contains
the mechanisms that the benchmark hides:

``ic(w)``              independent cascade ``1 - (1-w)^a``
``wc(w)``              weighted cascade ``1 - (1-w)^E`` (``per_indeg`` divides w)
``uni(c)``             uniform ``1 - (1-c)^E``
``lt(alpha)``          linear threshold ``clip(alpha * a / indeg)``
``gate(g)``            hard complex-contagion gate ``1[a >= g]``
``soft_gate(g,beta)``  smooth gate ``sigmoid(beta * (a - g))``
``ramp(lo,hi)``        saturating ramp
``memory(rho, base)``  exposure memory around any effect
``two_hop(c)``         influence from one- and two-hop active neighbours
``community(A,B)``     effect A in some communities, B in the rest
``free(table)``        a monotone table over ``a``

``ProductLaw`` combines programs in log-odds (OneLife); a single program
reduces to that program.

For RPP the hidden object is a continuous profit-drift distribution rather
than a Bernoulli probability.  ``DriftProgram`` wraps a ``DriftEffect``
(Gaussian / uniform / mixture) whose location may depend on time, distance to
the POI just visited, current profit and spatial region.  It exposes
``drift(env, state, visited_now, rng)`` so it plugs directly into
``strata.domains.rpp.env``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp

EPS = 1e-12


# =========================================================================
# Bernoulli effects (IM)
# =========================================================================


class Effect:
    """Base class of IM activation effects."""

    name = "effect"
    rho = 0.0
    two_hop = False
    param_names = ()

    def prob(self, feats):
        raise NotImplementedError

    def accumulate(self, E, a):
        rho = float(getattr(self, "rho", 0.0))
        return a + rho * E if rho else a

    def n_params(self):
        return len(self.param_names)

    def get_params(self):
        return {k: getattr(self, k) for k in self.param_names}

    def with_params(self, **params):
        unknown = set(params) - set(self.param_names)
        if unknown:
            raise KeyError(
                "unknown parameters %s for %s" % (sorted(unknown), self.name)
            )
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        for key, value in params.items():
            setattr(new, key, value)
        return new

    def __repr__(self):
        params = ", ".join(
            "%s=%g" % (k, getattr(self, k))
            for k in self.param_names
            if np.isscalar(getattr(self, k))
        )
        return "<effect %s(%s)>" % (self.name, params)


class IndependentCascade(Effect):
    """``p = 1 - (1-w)^a`` (no memory; ``E = a`` when ``rho = 0``)."""

    name = "ic"
    param_names = ("w",)

    def __init__(self, w=0.05):
        self.w = float(w)

    def prob(self, feats):
        w = np.clip(self.w, 0.0, 1.0)
        return 1.0 - np.power(1.0 - w, feats.a)


class WeightedCascade(Effect):
    """``p = 1 - (1-w)^E``; ``per_indeg`` uses ``w / indeg`` per target."""

    name = "wc"
    param_names = ("w",)

    def __init__(self, w=1.0, per_indeg=False):
        self.w = float(w)
        self.per_indeg = bool(per_indeg)

    def prob(self, feats):
        if self.per_indeg:
            w = self.w / np.maximum(feats.indeg, 1)
        else:
            w = np.broadcast_to(self.w, np.shape(feats.E))
        w = np.clip(w, 0.0, 1.0)
        return 1.0 - np.power(1.0 - w, feats.E)


class Uniform(Effect):
    """Uniform independent cascade ``p = 1 - (1-c)^E``."""

    name = "uni"
    param_names = ("c",)

    def __init__(self, c=0.05):
        self.c = float(c)

    def prob(self, feats):
        c = np.clip(self.c, 0.0, 1.0)
        return 1.0 - np.power(1.0 - c, feats.E)


class LinearThreshold(Effect):
    """``p = clip(alpha * a / indeg, 0, 1)``."""

    name = "lt"
    param_names = ("alpha",)

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def prob(self, feats):
        return np.clip(self.alpha * feats.a / np.maximum(feats.indeg, 1), 0.0, 1.0)


class Gate(Effect):
    """Hard complex-contagion gate ``1[a >= g]``."""

    name = "gate"
    param_names = ("g",)

    def __init__(self, g=2):
        self.g = int(g)

    def prob(self, feats):
        return (feats.a >= self.g).astype(np.float32)


class SoftGate(Effect):
    """Smooth gate ``sigmoid(beta * (a - threshold))`` for ``a >= min_a``."""

    name = "soft_gate"
    param_names = ("threshold", "beta")
    min_a = 1.0

    def __init__(self, threshold=1.5, beta=4.0):
        self.threshold = float(threshold)
        self.beta = float(beta)

    def prob(self, feats):
        p = expit(self.beta * (feats.a - self.threshold))
        return np.where(feats.a >= self.min_a, p, 0.0).astype(np.float32)


class Ramp(Effect):
    """``p = clip((a - lo) / (hi - lo), 0, 1)``; ``hi=None`` means ``indeg``."""

    name = "ramp"
    param_names = ("lo", "hi")

    def __init__(self, lo=0.0, hi=None):
        self.lo = float(lo)
        self.hi = None if hi is None else float(hi)

    def prob(self, feats):
        hi = feats.indeg if self.hi is None else self.hi
        denom = np.maximum(np.asarray(hi) - self.lo, EPS)
        return np.clip((feats.a - self.lo) / denom, 0.0, 1.0)


class Memory(Effect):
    """Wrap an effect with an exposure memory ``E <- a + rho * E``."""

    param_names = ()

    def __init__(self, base, rho=1.0):
        self.base = base
        self.rho = float(rho)
        self.name = "memory(%s,rho=%g)" % (base.name, self.rho)
        self.two_hop = bool(getattr(base, "two_hop", False))

    def prob(self, feats):
        return self.base.prob(feats)

    def n_params(self):
        return self.base.n_params()

    def get_params(self):
        params = self.base.get_params()
        params["rho"] = self.rho
        return params

    def with_params(self, **params):
        params = dict(params)
        rho = params.pop("rho", self.rho)
        return Memory(self.base.with_params(**params), rho)


class TwoHop(Effect):
    """Influence from one- and two-hop active neighbours."""

    name = "two_hop"
    param_names = ("c",)
    two_hop = True

    def __init__(self, c=0.05):
        self.c = float(c)

    def prob(self, feats):
        a = feats.a if feats.a2 is None else feats.a + feats.a2
        c = np.clip(self.c, 0.0, 1.0)
        return 1.0 - np.power(1.0 - c, a)


class Community(Effect):
    """Apply ``effect_a`` on a mask and ``effect_b`` elsewhere.

    ``mask`` may be a boolean ``(n,)`` array.  When omitted, the first half of
    the distinct ``feats.community`` ids use ``effect_a`` (the convention of
    the hidden ``COMM`` law).
    """

    def __init__(self, effect_a, effect_b, mask=None):
        self.effect_a = effect_a
        self.effect_b = effect_b
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.name = "community(%s,%s)" % (effect_a.name, effect_b.name)
        self.rho = max(
            float(getattr(effect_a, "rho", 0.0)), float(getattr(effect_b, "rho", 0.0))
        )
        self.two_hop = bool(
            getattr(effect_a, "two_hop", False) or getattr(effect_b, "two_hop", False)
        )

    def _mask(self, feats):
        if self.mask is not None:
            return self.mask
        community = np.asarray(feats.community)
        ids = np.unique(community)
        if ids.size == 0:
            return np.zeros_like(community, dtype=bool)
        n_a = int(np.ceil(0.5 * ids.size))
        return np.isin(community, ids[:n_a])

    def prob(self, feats):
        mask = self._mask(feats)
        pa = self.effect_a.prob(feats)
        pb = self.effect_b.prob(feats)
        if np.ndim(mask) == 1:
            mask = mask[None, :]
        return np.where(mask, pa, pb)

    def n_params(self):
        return self.effect_a.n_params() + self.effect_b.n_params()

    def get_params(self):
        out = {"a." + k: v for k, v in self.effect_a.get_params().items()}
        out.update({"b." + k: v for k, v in self.effect_b.get_params().items()})
        return out

    def with_params(self, **params):
        pa = {k[2:]: v for k, v in params.items() if k.startswith("a.")}
        pb = {k[2:]: v for k, v in params.items() if k.startswith("b.")}
        return Community(
            self.effect_a.with_params(**pa) if pa else self.effect_a,
            self.effect_b.with_params(**pb) if pb else self.effect_b,
            self.mask,
        )


class Mixture(Effect):
    """Convex mixture of effects (a fixed mixing weight by default)."""

    param_names = ()

    def __init__(self, effects, weights=None):
        self.effects = list(effects)
        k = len(self.effects)
        w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, np.float64)
        if w.size != k:
            raise ValueError("one weight per mixture component is required")
        self.weights = w / w.sum()
        self.name = "mixture(" + ",".join(e.name for e in self.effects) + ")"
        self.rho = max(float(getattr(e, "rho", 0.0)) for e in self.effects)
        self.two_hop = any(bool(getattr(e, "two_hop", False)) for e in self.effects)

    def prob(self, feats):
        out = np.zeros_like(np.asarray(self.effects[0].prob(feats), np.float64))
        for w, effect in zip(self.weights, self.effects):
            out = out + w * effect.prob(feats)
        return out

    def n_params(self):
        return sum(e.n_params() for e in self.effects) + max(self.weights.size - 1, 0)

    def get_params(self):
        return {
            "weights": self.weights.copy(),
            "effects": [e.get_params() for e in self.effects],
        }

    def __repr__(self):
        return "<effect %s weights=%s>" % (self.name, np.round(self.weights, 3))


class Free(Effect):
    """A monotone probability table indexed by ``a`` (PAV-fitted)."""

    name = "free"
    param_names = ("table",)

    def __init__(self, table):
        self.table = np.asarray(table, dtype=np.float64).reshape(-1)
        if self.table.size == 0:
            raise ValueError("free effect needs a non-empty table")
        if np.any(np.diff(self.table) < -1e-12):
            raise ValueError("free table must be non-decreasing")

    def prob(self, feats):
        idx = np.clip(np.rint(feats.a).astype(np.int64), 0, self.table.size - 1)
        return self.table[idx]

    def n_params(self):
        return int(self.table.size)

    def get_params(self):
        return {"table": self.table.copy()}

    def with_params(self, **params):
        return Free(params.get("table", self.table))

    def __repr__(self):
        return "<effect free(%d buckets)>" % self.table.size


_EFFECTS = {
    "ic": IndependentCascade,
    "wc": WeightedCascade,
    "uni": Uniform,
    "lt": LinearThreshold,
    "gate": Gate,
    "soft_gate": SoftGate,
    "ramp": Ramp,
    "two_hop": TwoHop,
    "community": Community,
    "mixture": Mixture,
    "free": Free,
}

EFFECT_NAMES = tuple(_EFFECTS)


def make_effect(kind, **params):
    """Instantiate a named effect."""
    key = str(kind)
    if key not in _EFFECTS:
        raise KeyError("unknown effect %r; choose from %s" % (kind, list(EFFECT_NAMES)))
    return _EFFECTS[key](**params)


# Lower-case constructors mirroring the DSL notation used in the plan.
def ic(w=0.05):
    return IndependentCascade(w)


def wc(w=1.0, per_indeg=False):
    return WeightedCascade(w, per_indeg=per_indeg)


def uni(c=0.05):
    return Uniform(c)


def lt(alpha=1.0):
    return LinearThreshold(alpha)


def gate(g=2):
    return Gate(g)


def soft_gate(g=1.5, beta=4.0):
    return SoftGate(threshold=g, beta=beta)


def ramp(lo=0.0, hi=None):
    return Ramp(lo=lo, hi=hi)


def memory(effect, rho=1.0):
    return Memory(effect, rho=rho)


def two_hop(c=0.05):
    return TwoHop(c)


def community(effect_a, effect_b, mask=None):
    return Community(effect_a, effect_b, mask=mask)


def mixture(effects, weights=None):
    return Mixture(effects, weights=weights)


def free(table):
    return Free(table)


# =========================================================================
# Law programs
# =========================================================================


class LawProgram:
    """One DSL program: an effect plus an optional precondition mask."""

    def __init__(self, effect, params=None, precondition=None):
        if isinstance(effect, str):
            effect = make_effect(effect, **(params or {}))
        elif params:
            effect = effect.with_params(**params)
        self.effect = effect
        self.precondition = precondition
        self.name = effect.name
        if precondition is not None:
            self.name = "%s|pre" % self.name

    @property
    def rho(self):
        return float(getattr(self.effect, "rho", 0.0))

    @property
    def two_hop(self):
        return bool(getattr(self.effect, "two_hop", False))

    def accumulate(self, E, a):
        rho = self.rho
        return a + rho * E if rho else a

    def _precondition_mask(self, feats):
        pre = self.precondition
        if pre is None:
            return None
        if callable(pre):
            return np.asarray(pre(feats), dtype=bool)
        mask = np.asarray(pre, dtype=bool)
        if mask.ndim == 1:
            return mask[None, :]
        return mask

    def prob(self, feats):
        p = np.asarray(self.effect.prob(feats), dtype=np.float64)
        mask = self._precondition_mask(feats)
        if mask is not None:
            p = np.where(mask, p, 0.0)
        return p

    def n_params(self):
        return self.effect.n_params()

    def get_params(self):
        return self.effect.get_params()

    def with_params(self, **params):
        return LawProgram(
            self.effect.with_params(**params), precondition=self.precondition
        )

    def __repr__(self):
        return "<LawProgram %s>" % self.name


class ProductLaw:
    """OneLife product of programs, normalised over the two Bernoulli points.

    ``p = sigmoid(sum_i theta_i * logit(phi_i))``.  With a single program and
    ``theta = 1`` this is exactly that program's probability.
    """

    def __init__(self, programs, theta=None):
        self.programs = list(programs)
        if not self.programs:
            raise ValueError("ProductLaw needs at least one program")
        if theta is None:
            theta = np.ones(len(self.programs), dtype=np.float64)
        self.theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if self.theta.size != len(self.programs):
            raise ValueError("theta must have one entry per program")
        self.name = "product(" + ",".join(p.name for p in self.programs) + ")"
        self.rho = max(p.rho for p in self.programs)
        self.two_hop = any(p.two_hop for p in self.programs)

    def accumulate(self, E, a):
        rho = self.rho
        return a + rho * E if rho else a

    def component_probs(self, feats):
        return np.stack([p.prob(feats) for p in self.programs], axis=0)

    def prob(self, feats):
        if self.theta.size == 1 and abs(self.theta[0] - 1.0) < 1e-12:
            return self.programs[0].prob(feats)
        ps = np.clip(self.component_probs(feats), EPS, 1.0 - EPS)
        logits = np.log(ps) - np.log1p(-ps)
        return expit(np.tensordot(self.theta, logits, axes=(0, 0)))

    def n_params(self):
        return sum(p.n_params() for p in self.programs) + int(self.theta.size)

    def get_params(self):
        return {
            "theta": self.theta.copy(),
            "programs": [p.get_params() for p in self.programs],
        }

    def __repr__(self):
        return "<ProductLaw %s theta=%s>" % (self.name, np.round(self.theta, 3))


# =========================================================================
# RPP continuous drift laws
# =========================================================================


@dataclass
class DriftFeatures:
    """Sufficient statistics of one drift draw (``M`` paths).

    ``initial``, ``visited_now`` and ``kernel`` are optional extras used by the
    richer conditional drifts (mean reversion, seasonal cycles, spill-over).
    They default to ``None`` so that hand-built feature bundles keep working.
    """

    profit: np.ndarray  # (M, n) pre-drift profit
    visited: np.ndarray  # (M, n) bool, already visited
    t: np.ndarray  # (M,) day index (1-based)
    distance: np.ndarray  # (M, n) distance to the POI visited this day (0 if none)
    region: np.ndarray  # (n,) spatial region id
    initial: np.ndarray = None  # (M, n) initial popularity
    visited_now: np.ndarray = None  # (M, n) bool, POIs visited this day
    kernel: np.ndarray = None  # (n, n) spatial spill kernel, rows = source


def spill_kernel(env, radius=None):
    """Spill kernel of the benchmark: ``exp(-d / (max_dist / 4))``, no self."""
    dist = np.asarray(env.dist, dtype=np.float64)
    max_dist = float(getattr(env, "max_dist", 0.0)) or float(np.max(dist)) or 1.0
    radius = max_dist / 4.0 if radius is None else float(radius)
    kernel = np.exp(-dist / max(radius, EPS))
    np.fill_diagonal(kernel, 0.0)
    return kernel


def drift_features(env, state, visited_now):
    """Build :class:`DriftFeatures` from a live RPP environment/state."""
    visited_now = np.asarray(visited_now, dtype=bool)
    M, n = state.profit.shape
    visited = state.visited | visited_now
    t = np.full(M, int(state.t), dtype=np.int64)
    if visited_now.any():
        dmat = np.asarray(env.dist)[None, :, :]
        big = np.where(visited_now[:, :, None], dmat, np.inf)
        distance = big.min(axis=2)
        distance[~np.isfinite(distance)] = 0.0
    else:
        distance = np.zeros((M, n), dtype=np.float64)
    region = np.asarray(getattr(env, "regions", np.zeros(n, dtype=np.int64)))
    initial = getattr(env, "initial_profit", None)
    initial = (
        np.zeros((M, n), dtype=np.float64)
        if initial is None
        else np.broadcast_to(np.asarray(initial, dtype=np.float64)[None, :], (M, n))
    )
    return DriftFeatures(
        profit=np.asarray(state.profit, dtype=np.float64).copy(),
        visited=visited,
        t=t,
        distance=distance.astype(np.float64),
        region=region.astype(np.int64),
        initial=np.asarray(initial, dtype=np.float64).copy(),
        visited_now=visited_now.copy(),
        kernel=spill_kernel(env),
    )


class DriftEffect:
    """Base class of continuous drift distributions."""

    name = "drift"
    param_names = ()
    distance_scale = 100.0

    def mean(self, feats):
        return np.zeros_like(feats.profit, dtype=np.float64)

    def logpdf(self, delta, feats):
        raise NotImplementedError

    def sample(self, feats, rng):
        raise NotImplementedError

    def n_params(self):
        return len(self.param_names)

    def __repr__(self):
        return "<drift %s>" % self.name


def _location(params, feats, distance_scale):
    """Shared conditional location function for Gaussian/uniform drifts.

    Location covariates, all optional:

    ``time_coef``     linear day trend (the DSL's ``TREND``/``SEASON`` base)
    ``distance_coef`` distance to the POI visited this day (``SPILL`` proxy)
    ``profit_coef``   current profit level
    ``initial_coef``  mean reversion ``initial - profit`` (the ``AR`` law)
    ``region_mu``     one intercept per spatial region (``TREND``)
    ``season_sin`` / ``season_cos``  per-POI periodic cycle
    ``spill_coef``    one signed coefficient per source POI
    """
    out = np.full_like(feats.profit, float(params.get("mu", 0.0)), dtype=np.float64)
    time_coef = float(params.get("time_coef", 0.0))
    if time_coef:
        out = out + time_coef * (feats.t[:, None] - 1.0)
    dist_coef = float(params.get("distance_coef", 0.0))
    if dist_coef:
        out = out + dist_coef * (feats.distance / max(distance_scale, EPS))
    profit_coef = float(params.get("profit_coef", 0.0))
    if profit_coef:
        out = out + profit_coef * (feats.profit - 1.0)
    initial_coef = float(params.get("initial_coef", 0.0))
    if initial_coef and feats.initial is not None:
        out = out + initial_coef * (feats.initial - feats.profit)
    region_mu = params.get("region_mu")
    if region_mu is not None:
        region_mu = np.asarray(region_mu, dtype=np.float64)
        if region_mu.size:
            idx = np.clip(feats.region, 0, region_mu.size - 1)
            out = out + region_mu[idx][None, :]
    season_sin = params.get("season_sin")
    season_cos = params.get("season_cos")
    if season_sin is not None and season_cos is not None:
        period = float(params.get("season_period", 10.0)) or 10.0
        omega = 2.0 * np.pi * feats.t[:, None] / period
        out = out + np.sin(omega) * np.asarray(season_sin)[None, :]
        out = out + np.cos(omega) * np.asarray(season_cos)[None, :]
    spill_coef = params.get("spill_coef")
    if spill_coef is not None and feats.visited_now is not None:
        sources = np.asarray(feats.visited_now, dtype=np.float64)
        sources = sources * np.asarray(spill_coef, dtype=np.float64)[None, :]
        out = out + sources @ np.asarray(feats.kernel, dtype=np.float64)
    return out


class GaussianDrift(DriftEffect):
    """Gaussian drift with optional location covariates (see ``_location``)."""

    name = "gaussian"
    param_names = ("mu", "sigma", "time_coef", "distance_coef", "profit_coef")

    def __init__(
        self,
        mu=0.0,
        sigma=0.05,
        time_coef=0.0,
        distance_coef=0.0,
        profit_coef=0.0,
        initial_coef=0.0,
        region_mu=None,
        season_sin=None,
        season_cos=None,
        season_period=10.0,
        spill_coef=None,
        distance_scale=100.0,
    ):
        self.params = {
            "mu": float(mu),
            "sigma": max(float(sigma), EPS),
            "time_coef": float(time_coef),
            "distance_coef": float(distance_coef),
            "profit_coef": float(profit_coef),
            "initial_coef": float(initial_coef),
            "region_mu": None
            if region_mu is None
            else np.asarray(region_mu, np.float64),
            "season_sin": None
            if season_sin is None
            else np.asarray(season_sin, np.float64),
            "season_cos": None
            if season_cos is None
            else np.asarray(season_cos, np.float64),
            "season_period": float(season_period),
            "spill_coef": None
            if spill_coef is None
            else np.asarray(spill_coef, np.float64),
        }
        self.distance_scale = float(distance_scale)

    def mean(self, feats):
        return _location(self.params, feats, self.distance_scale)

    def logpdf(self, delta, feats):
        mu = self.mean(feats)
        sigma = self.params["sigma"]
        return (
            -0.5 * ((delta - mu) / sigma) ** 2
            - np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
        )

    def sample(self, feats, rng):
        return self.mean(feats) + self.params["sigma"] * rng.normal(
            size=feats.profit.shape
        )

    def n_params(self):
        extra = 0
        for key in ("region_mu", "season_sin", "season_cos", "spill_coef"):
            value = self.params.get(key)
            if value is not None:
                extra += int(np.asarray(value).size)
        return 6 + extra


class UniformDrift(DriftEffect):
    """Uniform drift on ``[loc - h, loc + h]`` with a conditional ``loc``.

    The benchmark's synthetic laws all use bounded uniform noise, so a uniform
    drift with the correct location covariates is the exact model.  The
    log-density has a Gaussian tail outside the support (scale ``0.05 * h``)
    instead of ``-inf``: with ``h`` estimated as ``max |residual|`` an
    occasional held-out draw just outside the support would otherwise make the
    whole likelihood ``-inf`` and destroy model selection.
    """

    name = "uniform"
    param_names = ("mu", "half_width", "time_coef", "distance_coef", "profit_coef")

    def __init__(
        self,
        mu=0.0,
        half_width=0.05,
        time_coef=0.0,
        distance_coef=0.0,
        profit_coef=0.0,
        initial_coef=0.0,
        region_mu=None,
        season_sin=None,
        season_cos=None,
        season_period=10.0,
        spill_coef=None,
        distance_scale=100.0,
    ):
        self.params = {
            "mu": float(mu),
            "half_width": max(float(half_width), EPS),
            "time_coef": float(time_coef),
            "distance_coef": float(distance_coef),
            "profit_coef": float(profit_coef),
            "initial_coef": float(initial_coef),
            "region_mu": None
            if region_mu is None
            else np.asarray(region_mu, np.float64),
            "season_sin": None
            if season_sin is None
            else np.asarray(season_sin, np.float64),
            "season_cos": None
            if season_cos is None
            else np.asarray(season_cos, np.float64),
            "season_period": float(season_period),
            "spill_coef": None
            if spill_coef is None
            else np.asarray(spill_coef, np.float64),
        }
        self.distance_scale = float(distance_scale)

    def mean(self, feats):
        return _location(self.params, feats, self.distance_scale)

    def logpdf(self, delta, feats):
        mu = self.mean(feats)
        h = self.params["half_width"]
        resid = np.abs(delta - mu)
        core = -np.log(2.0 * h)
        tail = np.maximum(resid - h, 0.0) / max(0.1 * h, EPS)
        return core - 0.5 * tail**2

    def sample(self, feats, rng):
        mu = self.mean(feats)
        h = self.params["half_width"]
        return mu + rng.uniform(-h, h, size=feats.profit.shape)

    def n_params(self):
        extra = 0
        for key in ("region_mu", "season_sin", "season_cos", "spill_coef"):
            value = self.params.get(key)
            if value is not None:
                extra += int(np.asarray(value).size)
        return 6 + extra


class MixtureDrift(DriftEffect):
    """Finite Gaussian mixture away from the location covariates."""

    name = "mixture"

    def __init__(self, means, sigmas, weights):
        self.means = np.asarray(means, dtype=np.float64).reshape(-1)
        self.sigmas = np.maximum(np.asarray(sigmas, dtype=np.float64).reshape(-1), EPS)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        self.weights = w / w.sum()
        if not (self.means.size == self.sigmas.size == self.weights.size):
            raise ValueError("mixture components must have matching sizes")

    def mean(self, feats):
        return np.broadcast_to(
            (self.means * self.weights).sum(), np.shape(feats.profit)
        ).astype(np.float64)

    def logpdf(self, delta, feats):
        delta = np.asarray(delta, dtype=np.float64)
        comp = [
            np.log(w)
            - 0.5 * ((delta - m) / s) ** 2
            - np.log(s)
            - 0.5 * np.log(2 * np.pi)
            for m, s, w in zip(self.means, self.sigmas, self.weights)
        ]
        return logsumexp(np.stack(comp, axis=0), axis=0)

    def sample(self, feats, rng):
        shape = np.shape(feats.profit)
        comp = rng.choice(self.means.size, size=shape, p=self.weights)
        return self.means[comp] + self.sigmas[comp] * rng.normal(size=shape)

    def n_params(self):
        return 3 * int(self.means.size) - 1


class DriftProgram:
    """A drift effect packaged as a drop-in RPP law."""

    def __init__(self, effect):
        self.effect = effect
        self.name = effect.name
        self.param = effect

    def drift(self, env, state, visited_now, rng):
        feats = drift_features(env, state, visited_now)
        delta = self.effect.sample(feats, rng)
        return np.where(feats.visited, 0.0, delta)

    def logpdf(self, delta, feats):
        return self.effect.logpdf(delta, feats)

    def n_params(self):
        return self.effect.n_params()

    def __repr__(self):
        return "<DriftProgram %s>" % self.name


def make_drift_effect(kind, **params):
    if kind == "gaussian":
        return GaussianDrift(**params)
    if kind == "uniform":
        return UniformDrift(**params)
    if kind == "mixture":
        return MixtureDrift(**params)
    raise KeyError("unknown drift effect %r" % (kind,))


DRIFT_EFFECT_NAMES = ("gaussian", "uniform", "mixture")
