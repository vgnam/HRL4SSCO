"""M2a acceptance tests for L2 regions (IM communities and RPP spatial clusters)."""

import numpy as np

from strata.abstraction import regions
from strata.domains.im import graph as im_graph
from strata.domains.rpp import points


def _two_communities():
    import networkx as nx

    G = nx.DiGraph()
    for base in (0, 10):
        for i in range(base, base + 6):
            for j in range(base, base + 6):
                if i != j:
                    G.add_edge(i, j)
    return im_graph.build_graph(G, name="two-communities")


def test_im_regions_whole_graph_and_largest_communities():
    G = _two_communities()
    regs = regions.make_regions(G, k_top=3)
    assert regs[0][0] == "all"
    assert np.all(regs[0][1])
    assert len(regs) == 3  # two communities exist, only two can be returned
    sizes = [int(mask.sum()) for _, mask in regs[1:]]
    assert sizes == sorted(sizes, reverse=True)
    # Community masks are disjoint and stay inside the graph.
    assert not np.any(regs[1][1] & regs[2][1])
    assert np.all(regs[1][1] | regs[2][1])


def test_im_regions_are_deterministic():
    G = im_graph.sample_instance("dblp", 120, seed=101)
    a = regions.make_regions(G, k_top=3)
    b = regions.make_regions(G, k_top=3)
    assert [name for name, _ in a] == [name for name, _ in b]
    for (_, ma), (_, mb) in zip(a, b):
        assert np.array_equal(ma, mb)


def test_spatial_regions_partition_the_points():
    loc, _, _ = points.random_instance(80, seed=5)
    regs = regions.make_spatial_regions(loc, k_top=3)
    assert regs[0][0] == "all"
    assert np.all(regs[0][1])
    union = np.zeros(loc.shape[0], dtype=bool)
    for _, mask in regs[1:]:
        assert not np.any(union & mask)
        union |= mask
    assert np.all(union)
    sizes = [int(m.sum()) for _, m in regs[1:]]
    assert sizes == sorted(sizes, reverse=True)


def test_make_regions_dispatches_by_domain():
    from strata.domains.rpp import env as rpp_env

    G = _two_communities()
    loc, profit, _ = points.random_instance(30, seed=1)
    env = rpp_env.RPPEnv(loc, profit, budget=10, T=5)
    im_regs = regions.make_regions_for(G, k_top=2)
    rpp_regs = regions.make_regions_for(env, k_top=2)
    assert im_regs[0][0] == rpp_regs[0][0] == "all"
    assert len(im_regs) <= 3 and len(rpp_regs) <= 3
