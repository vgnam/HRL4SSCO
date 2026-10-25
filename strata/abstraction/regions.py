"""Regions of the L2 abstraction (whole instance + the largest communities).

For IM a region is a boolean node mask: ``'all'`` plus the ``k_top`` largest
Louvain communities of the graph (the partition stored in
:class:`strata.core.types.Graph`).  For RPP regions are spatial clusters of POI
coordinates (k-means, :func:`strata.domains.rpp.points.spatial_clusters`).

Region lists are ordered: index 0 is always the whole instance, followed by
communities/clusters ``c0, c1, ...`` sorted by decreasing size.  Macros refer
to a region by this integer index, so the ordering must stay deterministic.
"""

import numpy as np

from strata.core.rng import make_rng

ALL_REGION_NAME = "all"


def _ordered_ids(labels):
    """Unique labels sorted by decreasing label size (deterministic ties)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(0, dtype=np.int64)
    ids, counts = np.unique(labels, return_counts=True)
    # Stable sort on -count keeps the smaller label first on ties.
    order = np.argsort(-counts, kind="stable")
    return ids[order]


def _masks_from_labels(labels, k_top):
    n = int(np.asarray(labels).size)
    out = [(ALL_REGION_NAME, np.ones(n, dtype=bool))]
    ids = _ordered_ids(labels)[: max(0, int(k_top))]
    for i, cid in enumerate(ids):
        out.append(("c%d" % i, np.asarray(labels) == cid))
    return out


def make_regions(graph, k_top=3):
    """Return ``[('all', mask), ('c0', mask), ...]`` for an IM :class:`Graph`.

    Uses the Louvain partition already stored on the graph (computed with
    ``seed=0`` in :mod:`strata.domains.im.graph`), so regions are consistent
    across every planner that sees the same instance.
    """
    community = np.asarray(graph.community, dtype=np.int64)
    if community.size != graph.adj.shape[0]:
        raise ValueError("graph.community does not match the number of nodes")
    return _masks_from_labels(community, k_top)


def make_spatial_regions(loc, k_top=3, seed=0):
    """Return whole-instance plus the ``k_top`` largest spatial clusters.

    ``loc`` is either an ``(n, 2)`` coordinate array or an object with a
    ``loc`` attribute (e.g. :class:`strata.domains.rpp.env.RPPEnv`).
    """
    from strata.domains.rpp import points

    if hasattr(loc, "loc"):
        loc = loc.loc
    loc = np.asarray(loc, dtype=np.float64)
    if loc.ndim != 2 or loc.shape[1] != 2:
        raise ValueError("loc must have shape (n, 2)")
    labels = points.spatial_clusters(loc, max(1, int(k_top)), seed=seed)
    return _masks_from_labels(labels, k_top)


def make_regions_for(obj, k_top=3, seed=0):
    """Dispatch to :func:`make_regions` (IM) or :func:`make_spatial_regions`."""
    if hasattr(obj, "adj"):
        return make_regions(obj, k_top=k_top)
    if hasattr(obj, "loc") or (
        isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 2
    ):
        return make_spatial_regions(obj, k_top=k_top, seed=seed)
    raise TypeError("cannot build regions for %r" % (type(obj).__name__,))


def region_mask(regions, region):
    """Mask of region index ``region`` (negative/oversized indexes are safe)."""
    if not regions:
        raise ValueError("empty region list")
    idx = int(region)
    if idx < 0 or idx >= len(regions):
        idx = 0
    return regions[idx][1]


def random_mask(n, k_top=1, seed=0):
    """Convenience: a random non-empty region mask (used by tests/ablations)."""
    rng = make_rng("strata-region-mask", int(seed), int(n))
    m = rng.random(int(n)) < 0.5
    if not m.any():
        m[rng.integers(int(n))] = True
    return m
