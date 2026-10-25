"""POI instances for the RPP benchmark.

This is a port of ``RPP/real_data.py`` (Flickr tour trajectories and
Foursquare check-ins) plus the synthetic instances used in the paper (uniform
points in ``[0, 100]^2``).  It also provides the geometric primitives needed by
later milestones: the Euclidean distance matrix and a small deterministic
k-means used to define spatial regions.

An instance is ``(loc, initial_profit, profit_factor)``:

* ``loc`` -- ``(n, 2)`` coordinates, in ``[0, 100]^2``; node 0 is the start;
* ``initial_profit`` -- popularity mapped to ``[0.5, 1]``;
* ``profit_factor`` -- half-width of the daily uniform profit drift (the
  benchmark default is ``0.05``; FITTED uses the value fitted from real
  period-to-period popularity changes).
"""

import csv
import datetime
import os
import random
from collections import defaultdict

import numpy as np

from strata.core.rng import make_rng

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_DIR = os.path.join(REPO_ROOT, "RPP", "datasets")

FLICKR_CITIES = ["Edin", "Glas", "Melb", "Osak", "Toro"]
FOURSQUARE_CITIES = ["NYC", "TKY"]
DATASETS = ["flickr-%s" % c for c in FLICKR_CITIES] + [
    "foursquare-%s" % c for c in FOURSQUARE_CITIES
]

DEFAULT_PROFIT_FACTOR = 0.05
EARTH_RADIUS = 6371.0
FLICKR_MIN_VISITS_PER_YEAR = 100

_cache = {}
_delta_cache = {}


# --------------------------------------------------------------- real datasets


def load_flickr(city):
    """Return lat, lon, total visits and per-year visits of every POI."""
    pois = list(
        csv.DictReader(open(os.path.join(DATA_DIR, "flickr", "poi-%s.csv" % city)))
    )
    index = {p["poiID"]: i for i, p in enumerate(pois)}
    lat = np.array([float(p["poiLat"]) for p in pois])
    lon = np.array([float(p["poiLon"]) for p in pois])

    epoch = datetime.datetime(1970, 1, 1)
    visits = defaultdict(lambda: np.zeros(len(pois)))
    seen = set()
    for r in csv.DictReader(
        open(os.path.join(DATA_DIR, "flickr", "traj-%s.csv" % city))
    ):
        # Count each POI once per trajectory.
        key = (r["userID"], r["trajID"], r["poiID"])
        if key in seen:
            continue
        seen.add(key)
        year = (epoch + datetime.timedelta(seconds=int(r["startTime"]))).year
        visits[year][index[r["poiID"]]] += 1

    count = sum(visits.values())
    periods = np.array(
        [v for y, v in sorted(visits.items()) if v.sum() >= FLICKR_MIN_VISITS_PER_YEAR]
    )
    return lat, lon, count, periods.T


def load_foursquare(city):
    """Return lat, lon, total check-ins and per-month check-ins per venue."""
    path = os.path.join(DATA_DIR, "foursquare", "venues-%s.csv" % city)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found, run datasets/prepare_foursquare.py first" % path
        )
    rows = list(csv.reader(open(path, encoding="utf-8")))
    header, rows = rows[0], rows[1:]
    month_cols = header.index("users") + 1
    lat = np.array([float(r[2]) for r in rows])
    lon = np.array([float(r[3]) for r in rows])
    count = np.array([float(r[4]) for r in rows])
    periods = np.array([[float(x) for x in r[month_cols:]] for r in rows])
    return lat, lon, count, periods


def load(name):
    """Load (and cache) raw dataset statistics plus the fitted drift width."""
    if name not in _cache:
        if name not in DATASETS:
            raise ValueError(
                "unknown dataset %r, choose from %s" % (name, list(DATASETS))
            )
        source, city = name.split("-")
        if source == "flickr":
            lat, lon, count, periods = load_flickr(city)
        else:
            lat, lon, count, periods = load_foursquare(city)
        log_count = np.log1p(count)
        data = {
            "lat": lat,
            "lon": lon,
            "count": count,
            "periods": periods,
            "log_range": (log_count.min(), log_count.max()),
        }
        data["fitted_profit_factor"] = fit_profit_factor(data)
        _cache[name] = data
    return _cache[name]


def popularity_to_profit(count, log_range):
    """Map popularity on a log scale into ``[0.5, 1]``."""
    lo, hi = log_range
    return 0.5 + 0.5 * (np.log1p(count) - lo) / (hi - lo)


def fit_profit_factor(data):
    """Uniform half-width matching the std of period-to-period profit changes."""
    periods = data["periods"]
    scaled = periods / periods.sum(axis=0, keepdims=True) * data["count"].sum()
    profit = popularity_to_profit(scaled, data["log_range"])
    delta = np.diff(profit, axis=1)
    # std of U(-a, a) is a / sqrt(3).
    return float(np.sqrt(3) * delta.std())


def project(lat, lon):
    """Equirectangular projection to km, rescaled into ``[0, 100]^2``."""
    lat0 = np.radians(lat.mean())
    xy = np.stack([np.radians(lon) * np.cos(lat0), np.radians(lat)], axis=1)
    xy = xy * EARTH_RADIUS
    xy -= xy.min(axis=0)
    return xy / xy.max() * 100


def build_instance(data, idx, drift="default"):
    """Build ``(loc, initial_profit, profit_factor)`` for a set of venues."""
    lat, lon, count = data["lat"][idx], data["lon"][idx], data["count"][idx]
    loc = project(lat, lon)
    # Start at the POI closest to the centroid, placed at index 0.
    start = int(np.argmin(((loc - loc.mean(axis=0)) ** 2).sum(axis=1)))
    order = np.r_[start, np.delete(np.arange(len(idx)), start)]
    loc = project(lat[order], lon[order])
    initial_profit = popularity_to_profit(count[order], data["log_range"])
    profit_factor = (
        data["fitted_profit_factor"] if drift == "fitted" else DEFAULT_PROFIT_FACTOR
    )
    return loc, initial_profit, profit_factor


def period_profit_deltas(name):
    """Empirical period-to-period profit changes, aligned with the full instance.

    Shape ``(n, P - 1)``; used by the ``FITTED`` drift law.  Only valid when the
    environment uses the whole city (the returned order matches
    ``build_instance(data, np.arange(n))``).
    """
    if name in _delta_cache:
        return _delta_cache[name]
    data = load(name)
    periods = data["periods"]
    if periods.ndim != 2 or periods.shape[1] < 2:
        out = np.zeros((len(data["count"]), 0), dtype=np.float64)
    else:
        scaled = periods / periods.sum(axis=0, keepdims=True) * data["count"].sum()
        profit = popularity_to_profit(scaled, data["log_range"])
        delta = np.diff(profit, axis=1)
        loc = project(data["lat"], data["lon"])
        start = int(np.argmin(((loc - loc.mean(axis=0)) ** 2).sum(axis=1)))
        order = np.r_[start, np.delete(np.arange(len(data["count"])), start)]
        out = delta[order]
    _delta_cache[name] = out
    return out


def instance_order(data, idx):
    """Dataset venue ids of ``build_instance(data, idx)`` in instance order."""
    idx = np.asarray(idx)
    loc = project(data["lat"][idx], data["lon"][idx])
    start = int(np.argmin(((loc - loc.mean(axis=0)) ** 2).sum(axis=1)))
    return idx[np.r_[start, np.delete(np.arange(len(idx)), start)]]


def venue_profit_deltas(name, venues):
    """Empirical period-to-period profit changes of the given dataset venues.

    ``venues`` are dataset ids in instance order (see :func:`instance_order`),
    so row ``i`` holds the observed changes of POI ``i`` of the instance. This
    is what makes ``FITTED`` a real-data law on sampled instances instead of
    an IID law with a fitted width.
    """
    data = load(name)
    periods = data["periods"]
    if periods.ndim != 2 or periods.shape[1] < 2:
        return np.zeros((len(venues), 0), dtype=np.float64)
    scaled = periods / periods.sum(axis=0, keepdims=True) * data["count"].sum()
    profit = popularity_to_profit(scaled, data["log_range"])
    return np.diff(profit, axis=1)[np.asarray(venues)]


def sample_venues(data, n, method, rng):
    """Sample venue indices for foursquare-style datasets."""
    num = len(data["count"])
    if method == "uniform":
        return np.array(rng.sample(range(num), n))
    if method == "knn":
        xy = project(data["lat"], data["lon"])
        center = rng.randrange(num)
        return np.argsort(((xy - xy[center]) ** 2).sum(axis=1))[:n]
    raise ValueError("unknown sampling method %r" % (method,))


def instance_sampler(name, n=None, method="knn", drift="default", seed=None):
    """Return a callable producing an instance on each call."""
    data = load(name)
    num = len(data["count"])
    if name.startswith("flickr"):
        if n is not None and n != num:
            raise ValueError("%s has %d POIs, n = %d is not available" % (name, num, n))
        return lambda: build_instance(data, np.arange(num), drift)
    if n is None or n > num:
        raise ValueError("%s needs 0 < n <= %d" % (name, num))
    rng = random.Random(seed)
    return lambda: build_instance(data, sample_venues(data, n, method, rng), drift)


# --------------------------------------------------------------- synthetic / API


def random_instance(n, seed=0, profit_factor=DEFAULT_PROFIT_FACTOR):
    """Synthetic instance of the paper: uniform points in ``[0, 100]^2``."""
    rng = make_rng("rpp-random-instance", int(seed), int(n))
    loc = rng.random((int(n), 2)) * 100.0
    initial_profit = rng.uniform(0.5, 1.0, int(n))
    return loc, initial_profit, float(profit_factor)


def make_instance(name=None, n=None, method="knn", drift="default", seed=0):
    """Build an instance from a real dataset name or a synthetic seed."""
    if name is None:
        if n is None:
            raise ValueError("n is required for a synthetic instance")
        return random_instance(n, seed=seed)
    return instance_sampler(name, n=n, method=method, drift=drift, seed=seed)()


def make_instance_with_venues(name, n=None, method="knn", drift="default", seed=0):
    """Like :func:`make_instance` for a real dataset, plus the venue ids.

    Returns ``(loc, initial_profit, profit_factor, venues)`` where ``venues``
    are dataset ids in instance order; pass them to ``FITTED(venues=...)``.
    Draws the same venues as ``instance_sampler(..., seed=seed)()``.
    """
    data = load(name)
    num = len(data["count"])
    if name.startswith("flickr"):
        idx = np.arange(num)
    else:
        if n is None or n > num:
            raise ValueError("%s needs 0 < n <= %d" % (name, num))
        idx = sample_venues(data, n, method, random.Random(seed))
    loc, initial_profit, profit_factor = build_instance(data, idx, drift)
    return loc, initial_profit, profit_factor, instance_order(data, idx)


def distance_matrix(loc):
    """Euclidean pairwise distance matrix."""
    loc = np.asarray(loc, dtype=np.float64)
    return np.sqrt(((loc[:, np.newaxis, :] - loc[np.newaxis, :, :]) ** 2).sum(axis=2))


def spatial_clusters(loc, k, seed=0):
    """Deterministic k-means labels (small, dependency-free implementation)."""
    loc = np.asarray(loc, dtype=np.float64)
    n = loc.shape[0]
    k = int(min(max(1, int(k)), n))
    rng = make_rng("rpp-spatial-clusters", int(seed), n, k)
    centers = loc[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(50):
        d = ((loc[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = labels == j
            if members.any():
                centers[j] = loc[members].mean(axis=0)
            else:
                centers[j] = loc[rng.integers(n)]
    return labels
