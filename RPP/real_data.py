"""Real-world RPP instances from Flickr tour trajectories and Foursquare check-ins.

flickr-{Edin,Glas,Melb,Osak,Toro}: Flickr POIs and visit trajectories of Lim et al. (IJCAI 2015),
    as released in github.com/computationalmedia/tour-cikm16. One instance per city (27-88 POIs).
foursquare-{NYC,TKY}: Foursquare check-ins of Yang et al. (IEEE TSMC 2015), aggregated per venue by
    datasets/prepare_foursquare.py. Instances are n venues sampled from the city.

An instance is (loc, initial_profit, profit_factor), matching the synthetic RPP of Env:
- loc: lat/lon projected to km and rescaled into [0, 100]^2 (aspect ratio kept); node 0 is the
  start node, chosen as the POI closest to the centroid.
- initial_profit: popularity (number of visits / check-ins) on a log scale mapped to [0.5, 1].
- profit_factor: half-width of the uniform daily profit drift, either the default 0.05 of the
  synthetic RPP or fitted from period-to-period popularity changes (years for Flickr, months for
  Foursquare). The fitted value includes count sampling noise, so it is an upper estimate.
"""
import csv
import datetime
import os
import random
from collections import defaultdict

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
FLICKR_CITIES = ['Edin', 'Glas', 'Melb', 'Osak', 'Toro']
FOURSQUARE_CITIES = ['NYC', 'TKY']
DATASETS = [f'flickr-{c}' for c in FLICKR_CITIES] + [f'foursquare-{c}' for c in FOURSQUARE_CITIES]

DEFAULT_PROFIT_FACTOR = 0.05
EARTH_RADIUS = 6371.0
FLICKR_MIN_VISITS_PER_YEAR = 100

_cache = {}


def load_flickr(city):
    """Return lat, lon, total visits and per-year visits of every POI of the city."""
    pois = list(csv.DictReader(open(os.path.join(DATA_DIR, 'flickr', f'poi-{city}.csv'))))
    index = {p['poiID']: i for i, p in enumerate(pois)}
    lat = np.array([float(p['poiLat']) for p in pois])
    lon = np.array([float(p['poiLon']) for p in pois])

    epoch = datetime.datetime(1970, 1, 1)
    visits = defaultdict(lambda: np.zeros(len(pois)))
    seen = set()
    for r in csv.DictReader(open(os.path.join(DATA_DIR, 'flickr', f'traj-{city}.csv'))):
        # count each POI once per trajectory
        key = (r['userID'], r['trajID'], r['poiID'])
        if key in seen:
            continue
        seen.add(key)
        year = (epoch + datetime.timedelta(seconds=int(r['startTime']))).year
        visits[year][index[r['poiID']]] += 1

    count = sum(visits.values())
    periods = np.array([v for y, v in sorted(visits.items()) if v.sum() >= FLICKR_MIN_VISITS_PER_YEAR])
    return lat, lon, count, periods.T


def load_foursquare(city):
    """Return lat, lon, total check-ins and per-month check-ins of every venue of the city."""
    path = os.path.join(DATA_DIR, 'foursquare', f'venues-{city}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found, run datasets/prepare_foursquare.py first')
    rows = list(csv.reader(open(path, encoding='utf-8')))
    header, rows = rows[0], rows[1:]
    month_cols = header.index('users') + 1
    lat = np.array([float(r[2]) for r in rows])
    lon = np.array([float(r[3]) for r in rows])
    count = np.array([float(r[4]) for r in rows])
    periods = np.array([[float(x) for x in r[month_cols:]] for r in rows])
    return lat, lon, count, periods


def load(name):
    if name not in _cache:
        if name not in DATASETS:
            raise ValueError(f"unknown dataset '{name}', choose from {DATASETS}")
        source, city = name.split('-')
        lat, lon, count, periods = load_flickr(city) if source == 'flickr' else load_foursquare(city)
        log_count = np.log1p(count)
        _cache[name] = {
            'lat': lat, 'lon': lon, 'count': count, 'periods': periods,
            'log_range': (log_count.min(), log_count.max()),
        }
        _cache[name]['fitted_profit_factor'] = fit_profit_factor(_cache[name])
    return _cache[name]


def popularity_to_profit(count, log_range):
    lo, hi = log_range
    return 0.5 + 0.5 * (np.log1p(count) - lo) / (hi - lo)


def fit_profit_factor(data):
    """Uniform half-width matching the std of period-to-period profit changes."""
    periods = data['periods']
    # rescale each period to the total volume so that profits share the scale of initial_profit
    scaled = periods / periods.sum(axis=0, keepdims=True) * data['count'].sum()
    profit = popularity_to_profit(scaled, data['log_range'])
    delta = np.diff(profit, axis=1)
    # std of U(-a, a) is a / sqrt(3)
    return float(np.sqrt(3) * delta.std())


def project(lat, lon):
    """Equirectangular projection to km, rescaled into [0, 100]^2."""
    lat0 = np.radians(lat.mean())
    xy = np.stack([np.radians(lon) * np.cos(lat0), np.radians(lat)], axis=1) * EARTH_RADIUS
    xy -= xy.min(axis=0)
    return xy / xy.max() * 100


def build_instance(data, idx, drift='default'):
    lat, lon, count = data['lat'][idx], data['lon'][idx], data['count'][idx]
    loc = project(lat, lon)
    # start at the POI closest to the centroid, placed at index 0
    start = int(np.argmin(((loc - loc.mean(axis=0)) ** 2).sum(axis=1)))
    order = np.r_[start, np.delete(np.arange(len(idx)), start)]
    loc = project(lat[order], lon[order])
    initial_profit = popularity_to_profit(count[order], data['log_range'])
    profit_factor = data['fitted_profit_factor'] if drift == 'fitted' else DEFAULT_PROFIT_FACTOR
    return loc, initial_profit, profit_factor


def sample_venues(data, n, method, rng):
    num = len(data['count'])
    if method == 'uniform':
        return np.array(rng.sample(range(num), n))
    if method == 'knn':
        # n nearest venues around a random center venue: a neighbourhood to tour
        xy = project(data['lat'], data['lon'])
        center = rng.randrange(num)
        return np.argsort(((xy - xy[center]) ** 2).sum(axis=1))[:n]
    raise ValueError(f"unknown sampling method '{method}'")


def instance_sampler(name, n=None, method='knn', drift='default', seed=None):
    """Return a callable producing (loc, initial_profit, profit_factor) on each call.

    Flickr datasets always return the whole city (n must be None or the number of POIs).
    """
    data = load(name)
    num = len(data['count'])
    if name.startswith('flickr'):
        if n is not None and n != num:
            raise ValueError(f'{name} has {num} POIs, n = {n} is not available')
        return lambda: build_instance(data, np.arange(num), drift)
    if n is None or n > num:
        raise ValueError(f'{name} needs 0 < n <= {num}')
    rng = random.Random(seed)
    return lambda: build_instance(data, sample_venues(data, n, method, rng), drift)
