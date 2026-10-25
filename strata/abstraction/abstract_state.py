"""Abstract state ``z`` of the L2 abstraction (IM and RPP).

The abstract state is the planner's finite view of a concrete state::

    z = (t, budget_left, *bucket_ids)          # hashable tuple

with ``bucket_ids`` laid out region-major.  Every region contributes a fixed
number of bucketed features:

* IM  -- ``coverage`` (share of the region's nodes that are no longer
  inactive) and ``frontier`` (number of still-inactive region nodes with at
  least one active in-neighbour).
* RPP -- ``profit`` (mean current profit of the region's unvisited POIs),
  ``count`` (number of unvisited POIs in the region) and ``distance``
  (distance from the current position to the nearest unvisited POI of the
  region, normalised by the largest pairwise distance).  The distance feature
  is "global" in the sense that it uses the path's current position, but it is
  bucketed once per region so that ``z`` keeps a uniform layout.

Buckets are defined by sorted cut points stored in ``Abstraction.edges``::

    edges['coverage'] = np.array([0.25, 0.5, 0.75])

looks up a per-feature (optionally per-region, key ``(region, feature)``)
array and ``np.searchsorted(edges, value, side='right')`` maps a raw feature
value to ``0 .. len(edges)``.  Defaults are four quantile-like levels per
feature (three cut points); :meth:`Abstraction.split` inserts one extra cut
point into a single bucket key and returns a **new** abstraction, leaving the
original untouched (the planner's refinement loop relies on this).

``encode`` is vectorised over the ``M`` Monte-Carlo paths of the state and
returns one Python tuple per path, so the result is directly usable as a dict
key / pool key.  There is no hidden global state: the abstraction is a pure
function of its regions and edges.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from strata.config import AbstractionConfig
from strata.core.types import STATUS_ACTIVE, STATUS_INACTIVE

IM_FEATURES = ("coverage", "frontier")
RPP_FEATURES = ("profit", "count", "distance")

_FEATURES = {"im": IM_FEATURES, "rpp": RPP_FEATURES}


def _unique_cuts(values):
    """Sorted, strictly increasing float cut points."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.unique(arr)


def default_edges(domain, regions):
    """Four-level (three cut point) default bucket edges for a domain.

    The plan calls for quantile-like defaults, so count-like features get cut
    points derived from the instance size while bounded features use fixed
    fractions of their natural range.  ``AbstractionConfig`` may override any
    feature through ``bucket_edges`` (a ``{feature: cuts}`` mapping) when the
    integrator adds it; otherwise these constants are used.
    """
    domain = str(domain).lower()
    n = int(np.asarray(regions[0][1]).sum()) if regions else 0
    n = max(n, 1)
    if domain == "im":
        frontier = [
            max(1.0, float(round(0.01 * n))),
            max(2.0, float(round(0.05 * n))),
            max(3.0, float(round(0.15 * n))),
        ]
        edges = {
            "coverage": np.array([0.25, 0.5, 0.75], dtype=np.float64),
            "frontier": _unique_cuts(frontier),
        }
    elif domain == "rpp":
        count = [
            max(1.0, float(round(0.05 * n))),
            max(2.0, float(round(0.15 * n))),
            max(3.0, float(round(0.35 * n))),
        ]
        edges = {
            "profit": np.array([0.5, 0.65, 0.8], dtype=np.float64),
            "count": _unique_cuts(count),
            "distance": np.array([0.15, 0.35, 0.6], dtype=np.float64),
        }
    else:
        raise ValueError("unknown abstraction domain %r" % (domain,))

    cfg = AbstractionConfig()
    custom = getattr(cfg, "bucket_edges", None)
    if isinstance(custom, dict):
        for key, cuts in custom.items():
            edges[key] = _unique_cuts(cuts)
    return edges


class Abstraction:
    """Region-wise bucketing of concrete states into hashable abstract states."""

    def __init__(self, domain: str, regions: Sequence, edges: Dict = None):
        domain = str(domain).lower()
        if domain not in _FEATURES:
            raise ValueError("unknown abstraction domain %r" % (domain,))
        if not regions:
            raise ValueError("an abstraction needs at least one region")
        self.domain = domain
        self.regions: List[Tuple[str, np.ndarray]] = [
            (str(name), np.asarray(mask, dtype=bool)) for name, mask in regions
        ]
        n = int(self.regions[0][1].size)
        if n <= 0:
            raise ValueError("region masks must be non-empty")
        for _, mask in self.regions:
            if mask.shape != (n,):
                raise ValueError("every region mask must have shape (n,)")

        base = default_edges(domain, self.regions)
        if edges:
            for key, cuts in dict(edges).items():
                base[key] = _unique_cuts(cuts)
        for feature in _FEATURES[domain]:
            if feature not in base:
                raise ValueError("missing default edges for feature %r" % (feature,))
        self.edges: Dict = base

    # ------------------------------------------------------------- properties
    @property
    def feature_names(self) -> Tuple[str, ...]:
        return _FEATURES[self.domain]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    @property
    def n(self) -> int:
        return int(self.regions[0][1].size)

    def region_mask(self, region: int = 0) -> np.ndarray:
        idx = int(region)
        if idx < 0 or idx >= self.n_regions:
            idx = 0
        return self.regions[idx][1]

    def n_buckets(self, feature: str, region: int = 0) -> int:
        return int(self._edges_for(int(region), str(feature)).size) + 1

    # ----------------------------------------------------------------- edges
    def _edges_for(self, region: int, feature: str) -> np.ndarray:
        key = (int(region), str(feature))
        if key in self.edges:
            return np.asarray(self.edges[key], dtype=np.float64)
        if feature not in self.edges:
            raise KeyError("no bucket edges for feature %r" % (feature,))
        return np.asarray(self.edges[feature], dtype=np.float64)

    def _edge_key(self, bucket_key, feature: str):
        """Resolve the ``edges`` key a ``split`` call should refine."""
        if isinstance(bucket_key, tuple):
            if len(bucket_key) != 2:
                raise ValueError("tuple bucket_key must be (region, feature)")
            return (int(bucket_key[0]), str(bucket_key[1]))
        if isinstance(bucket_key, (int, np.integer)):
            return (int(bucket_key), str(feature))
        key = str(bucket_key)
        if key in self.edges or key in self.feature_names:
            return key
        return str(feature)

    def _bucket(self, feature: str, region: int, values) -> np.ndarray:
        edges = self._edges_for(int(region), str(feature))
        return np.searchsorted(
            edges, np.asarray(values, dtype=np.float64), side="right"
        ).astype(np.int64)

    def split(self, bucket_key, feature: str, new_edge):
        """Refine exactly one bucket key and return a new :class:`Abstraction`.

        ``bucket_key`` identifies the entry of ``edges`` to modify (a feature
        name, a region index or a ``(region, feature)`` pair); ``feature`` is
        the feature the cut belongs to (used when ``bucket_key`` is a region
        index or a fallback key).  ``new_edge`` is a scalar or a sequence of
        extra cut points.  The original abstraction is never mutated.
        """
        feature = str(feature)
        if feature not in self.feature_names:
            raise KeyError("unknown feature %r" % (feature,))
        key = self._edge_key(bucket_key, feature)
        base = np.asarray(
            self.edges.get(key, self.edges.get(feature, [])), dtype=np.float64
        )
        extra = np.atleast_1d(np.asarray(new_edge, dtype=np.float64)).ravel()
        edges = dict(self.edges)
        edges[key] = _unique_cuts(np.concatenate([base, extra]))
        return Abstraction(self.domain, list(self.regions), edges)

    # ---------------------------------------------------------------- encode
    def encode(self, env_or_graph, state) -> List[Tuple]:
        """Encode every path of ``state`` as an :data:`AbstractState` tuple."""
        if self.domain == "im":
            return self._encode_im(env_or_graph, state)
        return self._encode_rpp(env_or_graph, state)

    def _encode_im(self, graph, state) -> List[Tuple]:
        status = np.asarray(state.status)
        if status.ndim != 2:
            raise ValueError("IM state.status must be (M, n)")
        M, n = status.shape
        if n != self.n:
            raise ValueError("state has %d nodes, abstraction has %d" % (n, self.n))

        alive = status != STATUS_INACTIVE
        active = status == STATUS_ACTIVE
        if active.any():
            a = np.asarray(active.astype(np.float32) @ graph.adj, dtype=np.float32)
        else:
            a = np.zeros((M, n), dtype=np.float32)
        frontier = (status == STATUS_INACTIVE) & (a > 0.0)

        t = int(state.t)
        budget = int(state.budget_left)
        width = self.n_regions * len(IM_FEATURES)
        buckets = np.zeros((M, width), dtype=np.int64)
        for r, (_, mask) in enumerate(self.regions):
            size = max(int(mask.sum()), 1)
            coverage = (alive & mask[None, :]).sum(axis=1) / float(size)
            frontier_count = (frontier & mask[None, :]).sum(axis=1)
            col = r * len(IM_FEATURES)
            buckets[:, col] = self._bucket("coverage", r, coverage)
            buckets[:, col + 1] = self._bucket("frontier", r, frontier_count)
        return [(t, budget) + tuple(int(x) for x in row) for row in buckets]

    def _encode_rpp(self, env, state) -> List[Tuple]:
        profit = np.asarray(state.profit, dtype=np.float64)
        if profit.ndim != 2:
            raise ValueError("RPP state.profit must be (M, n)")
        M, n = profit.shape
        if n != self.n:
            raise ValueError("state has %d POIs, abstraction has %d" % (n, self.n))

        available = profit > 0.0
        current = np.asarray(state.current, dtype=np.int64).reshape(-1)
        dist = np.asarray(env.dist, dtype=np.float64)[current] / max(
            float(env.max_dist), 1e-12
        )

        width = self.n_regions * len(RPP_FEATURES)
        buckets = np.zeros((M, width), dtype=np.int64)
        for r, (_, mask) in enumerate(self.regions):
            avail = available & mask[None, :]
            count = avail.sum(axis=1).astype(np.float64)
            total = np.where(avail, profit, 0.0).sum(axis=1)
            mean_profit = np.where(count > 0.0, total / np.maximum(count, 1.0), 0.0)
            nearest = np.where(avail, dist, np.inf).min(axis=1)
            nearest = np.where(np.isfinite(nearest), nearest, 1.0)
            col = r * len(RPP_FEATURES)
            buckets[:, col] = self._bucket("profit", r, mean_profit)
            buckets[:, col + 1] = self._bucket("count", r, count)
            buckets[:, col + 2] = self._bucket("distance", r, nearest)

        t = int(state.t)
        budget = np.asarray(state.budget_left).reshape(-1)
        return [
            (t, int(budget[m])) + tuple(int(x) for x in buckets[m]) for m in range(M)
        ]

    # -------------------------------------------------------------- debugging
    def __repr__(self):
        return "<Abstraction %s regions=%d features=%s>" % (
            self.domain,
            self.n_regions,
            ",".join(self.feature_names),
        )
