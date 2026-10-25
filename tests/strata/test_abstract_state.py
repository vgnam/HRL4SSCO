"""M2b acceptance tests for the L2 abstract state (IM and RPP).

The central check is that :meth:`Abstraction.encode` agrees with a slow,
explicit per-path computation of the region features and bucket boundaries,
and that :meth:`Abstraction.split` refines exactly one feature without
mutating the original abstraction.
"""

import os
import sys

import networkx as nx
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from strata.abstraction import regions as regions_mod  # noqa: E402
from strata.abstraction.abstract_state import (  # noqa: E402
    IM_FEATURES,
    RPP_FEATURES,
    Abstraction,
)
from strata.core.types import (  # noqa: E402
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    SimState,
)
from strata.domains.im import graph as im_graph  # noqa: E402
from strata.domains.rpp import env as rpp_env  # noqa: E402
from strata.domains.rpp import points  # noqa: E402


def _bucket(cuts, value):
    return int(np.searchsorted(np.asarray(cuts, dtype=np.float64), value, side="right"))


def _reference_im(ab, graph, state):
    """Slow per-path reference for the IM coverage/frontier abstraction."""
    status = np.asarray(state.status)
    M = status.shape[0]
    out = []
    for m in range(M):
        row = [int(state.t), int(state.budget_left)]
        active = status[m] == STATUS_ACTIVE
        a = active.astype(np.float32) @ graph.adj
        for r, (_, mask) in enumerate(ab.regions):
            alive = status[m] != STATUS_INACTIVE
            coverage = float(alive[mask].mean())
            frontier_node = (status[m] == STATUS_INACTIVE) & (np.asarray(a) > 0.0)
            frontier = int(frontier_node[mask].sum())
            row.append(_bucket(ab.edges["coverage"], coverage))
            row.append(_bucket(ab.edges["frontier"], frontier))
        out.append(tuple(row))
    return out


def _reference_rpp(ab, env, state):
    """Slow per-path reference for the RPP profit/count/distance abstraction."""
    profit = np.asarray(state.profit)
    M = profit.shape[0]
    out = []
    for m in range(M):
        row = [int(state.t), int(state.budget_left[m])]
        available = profit[m] > 0.0
        dist = env.dist[int(state.current[m])] / env.max_dist
        for r, (_, mask) in enumerate(ab.regions):
            avail = available & mask
            count = int(avail.sum())
            mean_profit = float(profit[m][avail].mean()) if count else 0.0
            nearest = float(dist[avail].min()) if count else 1.0
            row.append(_bucket(ab.edges["profit"], mean_profit))
            row.append(_bucket(ab.edges["count"], count))
            row.append(_bucket(ab.edges["distance"], nearest))
        out.append(tuple(row))
    return out


@pytest.fixture(scope="module")
def im_setup():
    G = im_graph.build_graph(
        nx.gnp_random_graph(45, 0.12, seed=4, directed=True), name="m2b-im"
    )
    regions = regions_mod.make_regions(G, k_top=3)
    ab = Abstraction("im", regions)
    rng = np.random.default_rng(0)
    n = G.adj.shape[0]
    M = 4
    status = rng.integers(0, 3, (M, n)).astype(np.int8)
    E = rng.random((M, n)).astype(np.float32)
    state = SimState(status=status, E=E, t=3, budget_left=17)
    return G, regions, ab, state


@pytest.fixture(scope="module")
def rpp_setup():
    loc, initial, _ = points.random_instance(40, seed=7)
    env = rpp_env.RPPEnv(loc, initial, budget=20, T=8, seed=0, empty_day_advances=True)
    regions = regions_mod.make_spatial_regions(env, k_top=3)
    ab = Abstraction("rpp", regions)
    rng = np.random.default_rng(1)
    state = env.reset(M=3)
    state.profit = rng.uniform(0.3, 1.2, (3, env.n))
    state.profit[rng.random((3, env.n)) < 0.2] = 0.0
    state.visited = state.profit <= 0.0
    state.current = rng.integers(0, env.n, 3)
    state.budget_left = np.array([11, 7, 20], dtype=np.int64)
    state.t = 4
    return env, regions, ab, state


# ------------------------------------------------------------------- IM encode


def test_im_encode_matches_reference(im_setup):
    G, _, ab, state = im_setup
    zs = ab.encode(G, state)
    assert len(zs) == state.status.shape[0]
    expected = _reference_im(ab, G, state)
    assert zs == expected

    width = 2 + len(IM_FEATURES) * ab.n_regions
    for z in zs:
        assert isinstance(z, tuple)
        assert len(z) == width
        assert z[0] == 3 and z[1] == 17
        assert all(isinstance(v, int) for v in z)
        hash(z)  # hashable, usable as a pool key


def test_im_bucket_boundaries_and_split(im_setup):
    G, _, ab, state = im_setup
    # Build a tiny path graph with a known single frontier node.
    P = nx.DiGraph()
    P.add_nodes_from(range(4))
    P.add_edges_from([(0, 1), (1, 2), (2, 3)])
    Gp = im_graph.build_graph(P, name="path")
    tiny = SimState(
        status=np.array(
            [[STATUS_ACTIVE, STATUS_INACTIVE, STATUS_INACTIVE, STATUS_INACTIVE]],
            dtype=np.int8,
        ),
        E=np.zeros((1, 4), dtype=np.float32),
        t=0,
        budget_left=5,
    )
    regions = [("all", np.ones(4, dtype=bool))]
    tiny_ab = Abstraction("im", regions, edges={"coverage": [0.5], "frontier": [1]})
    z = tiny_ab.encode(Gp, tiny)[0]
    # coverage = 1 / 4 -> bucket 0; frontier = 1 -> searchsorted([1], 1) = 1.
    assert z == (0, 5, 0, 1)

    original_coverage = ab.edges["coverage"].copy()
    original_frontier = ab.edges["frontier"].copy()
    refined = ab.split("coverage", "coverage", 0.9)
    assert refined is not ab
    assert 0.9 in refined.edges["coverage"]
    # Only the coverage edges changed; every other feature is untouched.
    assert not np.array_equal(refined.edges["coverage"], ab.edges["coverage"])
    assert np.array_equal(refined.edges["frontier"], ab.edges["frontier"])
    # The original abstraction was not mutated (immutable-style split).
    assert np.array_equal(ab.edges["coverage"], original_coverage)
    assert np.array_equal(ab.edges["frontier"], original_frontier)

    # A tuple bucket key refines a single per-region feature only.
    per_region = ab.split((1, "frontier"), "frontier", 2.5)
    assert (1, "frontier") in per_region.edges
    assert np.array_equal(per_region.edges["coverage"], ab.edges["coverage"])
    assert np.array_equal(per_region.edges["frontier"], ab.edges["frontier"])


def test_default_edges_are_four_levels(im_setup, rpp_setup):
    _, _, im_ab, _ = im_setup
    _, _, rpp_ab, _ = rpp_setup
    for feature in IM_FEATURES:
        assert im_ab.n_buckets(feature) == 4
    for feature in RPP_FEATURES:
        assert rpp_ab.n_buckets(feature) == 4


# ------------------------------------------------------------------ RPP encode


def test_rpp_encode_matches_reference(rpp_setup):
    env, _, ab, state = rpp_setup
    zs = ab.encode(env, state)
    assert len(zs) == state.profit.shape[0]
    expected = _reference_rpp(ab, env, state)
    assert zs == expected

    width = 2 + len(RPP_FEATURES) * ab.n_regions
    for m, z in enumerate(zs):
        assert len(z) == width
        assert z[0] == 4
        assert z[1] == int(state.budget_left[m])
        hash(z)


def test_rpp_split_refines_one_feature(rpp_setup):
    env, _, ab, state = rpp_setup
    before = ab.encode(env, state)
    refined = ab.split("profit", "profit", 0.9)

    assert refined is not ab
    changed = [
        feature
        for feature in RPP_FEATURES
        if not np.array_equal(refined.edges[feature], ab.edges[feature])
    ]
    assert changed == ["profit"]

    # The old abstraction still encodes exactly as before.
    assert ab.encode(env, state) == before
    # Profit buckets can only move up when a cut point is added.
    after = refined.encode(env, state)
    n_profit = len(RPP_FEATURES)
    step = len(RPP_FEATURES)
    for z_before, z_after in zip(before, after):
        assert z_before[0] == z_after[0] and z_before[1] == z_after[1]
        for r in range(ab.n_regions):
            col = 2 + r * step
            assert z_after[col] >= z_before[col]
            assert (
                z_after[col + 1 : col + n_profit] == z_before[col + 1 : col + n_profit]
            )
