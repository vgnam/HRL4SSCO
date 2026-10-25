"""Hidden profit-drift laws of the RPP benchmark.

A drift law sees the realised state *after* the day's visits have been applied
and returns per-node profit changes ``(M, n)``.  The environment then adds
them to the unvisited POIs and applies the ``profit_min`` floor.

The benchmark families are:

``IID``     independent uniform noise ``U(-factor, factor)`` (the original env)
``AR``      mean-reverting around the initial popularity
``TREND``   regional linear trend plus noise
``SPILL``   visiting a POI pushes nearby profits up/down plus noise
``SEASON``  periodic cycle over the day index plus noise
``FITTED``  empirical period-to-period changes from ``RPP/real_data.py``

For ``M == 1`` the IID noise is drawn only for unvisited POIs, in ascending
index order, matching ``RPP/Env.py`` exactly (same seed -> same trajectory).
For ``M > 1`` the draw is vectorised over the whole ``(M, n)`` grid.
"""

import numpy as np

from strata.core.rng import make_rng
from strata.domains.rpp import points


def _uniform_unvisited(rng, visited, factor):
    """Uniform drift for unvisited POIs, bit-compatible with RPP/Env.py."""
    visited = np.asarray(visited, dtype=bool)
    M, n = visited.shape
    if M == 1:
        idx = np.flatnonzero(~visited[0])
        out = np.zeros((1, n), dtype=np.float64)
        if idx.size:
            out[0, idx] = factor * rng.uniform(-1.0, 1.0, idx.size)
        return out
    u = rng.uniform(-1.0, 1.0, (M, n))
    return np.where(visited, 0.0, factor * u)


class DriftLaw:
    """Base class: factor defaults to the environment's ``profit_factor``."""

    name = "drift"

    def __init__(self, factor=None, seed=0):
        self.factor = None if factor is None else float(factor)
        self.seed = int(seed)

    def _factor(self, env):
        return env.profit_factor if self.factor is None else self.factor

    def _iid(self, env, visited, rng):
        return _uniform_unvisited(rng, visited, self._factor(env))

    def _delta(self, env, state, visited, visited_now, rng):
        raise NotImplementedError

    def drift(self, env, state, visited_now, rng):
        """Full drift tensor; visited POIs (old and this round) get zero."""
        visited_now = np.asarray(visited_now, dtype=bool)
        visited = state.visited | visited_now
        delta = np.asarray(
            self._delta(env, state, visited, visited_now, rng), dtype=np.float64
        )
        return np.where(visited, 0.0, delta)

    def __repr__(self):
        return "<rpp-drift %s>" % self.name


class IID(DriftLaw):
    """Independent uniform drift, the benchmark's original mechanism."""

    name = "IID"

    def _delta(self, env, state, visited, visited_now, rng):
        return self._iid(env, visited, rng)


class AR(DriftLaw):
    """Ornstein-Uhlenbeck-like mean reversion towards initial popularity."""

    name = "AR"

    def __init__(self, factor=None, alpha=0.2, seed=0):
        super().__init__(factor, seed=seed)
        self.alpha = float(alpha)

    def _delta(self, env, state, visited, visited_now, rng):
        noise = self._iid(env, visited, rng)
        return noise + self.alpha * (env.initial_profit[None, :] - state.profit)


class TREND(DriftLaw):
    """A different linear trend in each spatial region (plus IID noise)."""

    name = "TREND"

    def __init__(
        self, factor=None, n_regions=4, trend_strength=0.02, seed=0, regions=None
    ):
        super().__init__(factor)
        self.n_regions = int(n_regions)
        self.trend_strength = float(trend_strength)
        self.seed = int(seed)
        self.regions = None if regions is None else np.asarray(regions, dtype=np.int64)
        self._rates = None

    def _ensure(self, env):
        if self.regions is None or self.regions.size != env.n:
            self.regions = points.spatial_clusters(
                env.loc, self.n_regions, seed=self.seed
            )
        if self._rates is None or self._rates.size != int(self.regions.max()) + 1:
            rng = make_rng("rpp-trend-rates", self.seed, env.n, self.n_regions)
            k = int(self.regions.max()) + 1
            self._rates = rng.normal(0.0, self.trend_strength, k)

    def _delta(self, env, state, visited, visited_now, rng):
        self._ensure(env)
        noise = self._iid(env, visited, rng)
        return noise + self._rates[self.regions][None, :]


class SPILL(DriftLaw):
    """Visiting a POI changes the profits of its spatial neighbours."""

    name = "SPILL"

    def __init__(self, factor=None, strength=0.02, radius=None, seed=0):
        super().__init__(factor)
        self.strength = float(strength)
        self.radius = None if radius is None else float(radius)
        self.seed = int(seed)
        self._kernel = None

    def _ensure(self, env):
        if self._kernel is not None and self._kernel.shape == (env.n, env.n):
            return
        rng = make_rng("rpp-spill-signs", self.seed, env.n)
        signs = rng.choice(np.array([-1.0, 1.0]), size=env.n)
        radius = env.max_dist / 4.0 if self.radius is None else self.radius
        radius = max(radius, 1e-9)
        kernel = np.exp(-env.dist / radius)
        np.fill_diagonal(kernel, 0.0)
        # kernel[u, v] is the influence of visiting u on target v.
        self._kernel = kernel * signs[:, None]

    def _delta(self, env, state, visited, visited_now, rng):
        self._ensure(env)
        noise = self._iid(env, visited, rng)
        spill = self.strength * (visited_now.astype(np.float64) @ self._kernel)
        return noise + spill


class SEASON(DriftLaw):
    """A periodic cycle over the day index, with a per-POI phase."""

    name = "SEASON"

    def __init__(self, factor=None, period=10.0, amplitude=0.05, seed=0, phase=None):
        super().__init__(factor)
        self.period = float(period)
        self.amplitude = float(amplitude)
        self.seed = int(seed)
        self.phase = None if phase is None else np.asarray(phase, dtype=np.float64)

    def _ensure(self, env):
        if self.phase is None or self.phase.size != env.n:
            rng = make_rng("rpp-season-phase", self.seed, env.n)
            self.phase = rng.uniform(0.0, 2.0 * np.pi, env.n)

    def _delta(self, env, state, visited, visited_now, rng):
        self._ensure(env)
        noise = self._iid(env, visited, rng)
        cycle = self.amplitude * np.sin(
            2.0 * np.pi * state.t / self.period + self.phase
        )
        return noise + cycle[None, :]


class FITTED(DriftLaw):
    """Empirical drift from real popularity changes (RPP/real_data.py).

    If ``deltas`` is available and aligned with the instance, each unvisited
    POI samples one of its observed period-to-period profit changes.  Otherwise
    the law falls back to IID with the fitted ``profit_factor``.

    For a sampled instance pass ``venues`` (from
    ``points.make_instance_with_venues``): row ``i`` of ``deltas`` is then the
    observed history of POI ``i``. Without ``venues`` the deltas only align
    with a whole-city instance.
    """

    name = "FITTED"

    def __init__(self, factor=None, dataset=None, deltas=None, seed=0, venues=None):
        if factor is None and dataset is not None:
            factor = points.load(dataset)["fitted_profit_factor"]
        super().__init__(factor)
        self.dataset = dataset
        self.seed = int(seed)
        if deltas is None and dataset is not None and venues is not None:
            deltas = points.venue_profit_deltas(dataset, venues)
        if deltas is None and dataset is not None:
            try:
                deltas = points.period_profit_deltas(dataset)
            except (FileNotFoundError, ValueError):
                deltas = None
        self.deltas = None if deltas is None else np.asarray(deltas, dtype=np.float64)

    def _delta(self, env, state, visited, visited_now, rng):
        d = self.deltas
        if d is None or d.ndim != 2 or d.shape[0] != env.n or d.shape[1] < 1:
            return self._iid(env, visited, rng)
        M, n = state.profit.shape
        P = d.shape[1]
        if M == 1:
            idx = np.flatnonzero(~visited[0])
            out = np.zeros((1, n), dtype=np.float64)
            if idx.size:
                cols = rng.integers(P, size=idx.size)
                out[0, idx] = d[idx, cols]
            return out
        cols = rng.integers(P, size=(M, n))
        return d[np.arange(n)[None, :], cols]


class CROWD(DriftLaw):
    """Congestion: each day a POI loses c per already-visited POI within radius r.

    The penalty accumulates as the agent visits the neighbourhood, so the
    effect depends on the whole visit history, not on today's visits (SPILL)
    or on space alone (TREND). Outside the DSL.
    """

    name = "CROWD"

    def __init__(self, factor=None, c=0.02, radius=None, seed=0):
        super().__init__(factor)
        self.c = float(c)
        self.radius = None if radius is None else float(radius)
        self.seed = int(seed)
        self._near = None

    def _ensure(self, env):
        if self._near is not None and self._near.shape == (env.n, env.n):
            return
        radius = env.max_dist / 4.0 if self.radius is None else self.radius
        near = env.dist <= radius
        np.fill_diagonal(near, False)
        self._near = near.astype(np.float64)

    def _delta(self, env, state, visited, visited_now, rng):
        self._ensure(env)
        noise = self._iid(env, visited, rng)
        crowd = visited.astype(np.float64) @ self._near
        return noise - self.c * crowd


class REGIME(DriftLaw):
    """Regional trends that flip sign at a hidden day t* of each region. Outside the DSL."""

    name = "REGIME"

    def __init__(self, factor=None, n_regions=4, trend_strength=0.03, seed=0, regions=None):
        super().__init__(factor)
        self.n_regions = int(n_regions)
        self.trend_strength = float(trend_strength)
        self.seed = int(seed)
        self.regions = None if regions is None else np.asarray(regions, dtype=np.int64)
        self._rates = None
        self._switch = None

    def _ensure(self, env):
        if self.regions is None or self.regions.size != env.n:
            self.regions = points.spatial_clusters(env.loc, self.n_regions, seed=self.seed)
        k = int(self.regions.max()) + 1
        if self._rates is None or self._rates.size != k:
            rng = make_rng("rpp-regime", self.seed, env.n, self.n_regions)
            self._rates = rng.choice([-1.0, 1.0], size=k) * self.trend_strength
            self._switch = rng.integers(2, max(3, env.T), size=k)

    def _delta(self, env, state, visited, visited_now, rng):
        self._ensure(env)
        noise = self._iid(env, visited, rng)
        sign = np.where(int(state.t) < self._switch, 1.0, -1.0)
        return noise + (self._rates * sign)[self.regions][None, :]


LAW_NAMES = ("IID", "AR", "TREND", "SPILL", "SEASON", "FITTED", "CROWD", "REGIME")
OUTSIDE_DSL = ("CROWD", "REGIME")

_CLASSES = {
    "IID": IID,
    "AR": AR,
    "TREND": TREND,
    "SPILL": SPILL,
    "SEASON": SEASON,
    "FITTED": FITTED,
    "CROWD": CROWD,
    "REGIME": REGIME,
}


def make_law(name, **params):
    key = str(name).upper()
    if key not in _CLASSES:
        raise KeyError("unknown RPP law %r; choose from %s" % (name, list(LAW_NAMES)))
    return _CLASSES[key](**params)


def get_law(name, **params):
    return make_law(name, **params)
