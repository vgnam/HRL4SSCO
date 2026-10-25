"""M0 acceptance tests for behaviour-log collection (IM and RPP).

The IM test is the quantitative acceptance criterion from the plan:
``collect_logs`` under CC2 on dblp (n=100, seed=101, 200 episodes) reproduces
the total number of activations of ``kt_law/t2_fit.collect_logs`` exactly.
"""

import os
import sys
import tempfile

import networkx as nx
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KT_DIR = os.path.join(REPO, "experiments", "kt_law")
if KT_DIR not in sys.path:
    sys.path.insert(0, KT_DIR)

import laws as kt_laws  # noqa: E402
import t2_fit  # noqa: E402

from strata.domains.im import graph as im_graph  # noqa: E402
from strata.domains.im import hidden_laws as im_laws  # noqa: E402
from strata.domains.im import logs as im_logs  # noqa: E402
from strata.domains.rpp import env as rpp_env  # noqa: E402
from strata.domains.rpp import hidden_laws as rpp_laws  # noqa: E402
from strata.domains.rpp import logs as rpp_logs  # noqa: E402
from strata.domains.rpp import points  # noqa: E402


def test_im_collect_logs_reproduces_t2_fit_cc2_dblp():
    G = im_graph.sample_instance("dblp", 100, seed=101)
    law = im_laws.make_law("CC2", G)

    ours = im_logs.collect_logs(G, law, T=10, B=20, n_episodes=200, seed=101)
    ours_total = sum(int(round_.y.sum()) for ep in ours for round_ in ep)

    reference = t2_fit.collect_logs(
        kt_laws.get_law("CC2"), G.adj, G.indeg, 10, 20, 200, seed=101
    )
    ref_total = sum(int(r["y"].sum()) for ep in reference for r in ep)
    assert ours_total == ref_total

    # Per-episode counts should match as well (stronger than the total).
    ours_per_ep = [sum(int(r.y.sum()) for r in ep) for ep in ours]
    ref_per_ep = [sum(int(r["y"].sum()) for r in ep) for ep in reference]
    assert ours_per_ep == ref_per_ep


def test_im_logs_save_load_roundtrip():
    G = im_graph.build_graph(
        nx.gnp_random_graph(40, 0.15, seed=4, directed=True), name="tiny"
    )
    episodes = im_logs.collect_logs(
        G, im_laws.make_law("CC3", G), T=5, B=10, n_episodes=3, seed=7
    )
    path = os.path.join(tempfile.gettempdir(), "strata_m0_im_logs.npz")
    written = im_logs.save_logs(path, episodes)
    loaded = im_logs.load_logs(written)
    assert len(loaded) == len(episodes)
    for ep_a, ep_b in zip(episodes, loaded):
        assert len(ep_a) == len(ep_b)
        for ra, rb in zip(ep_a, ep_b):
            np.testing.assert_array_equal(ra.a, rb.a)
            np.testing.assert_array_equal(ra.eligible, rb.eligible)
            np.testing.assert_array_equal(ra.y, rb.y)
            np.testing.assert_array_equal(ra.seeded, rb.seeded)
            np.testing.assert_array_equal(ra.before, rb.before)
            np.testing.assert_array_equal(ra.after, rb.after)
            assert ra.policy == rb.policy


def test_rpp_collect_logs_reproducible_and_budget_safe():
    loc, initial, _ = points.random_instance(35, seed=3)
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, budget=12, T=6)
    law = rpp_laws.get_law("SEASON", seed=0)
    eps_a = rpp_logs.collect_logs(env, law, n_episodes=6, seed=11)
    eps_b = rpp_logs.collect_logs(env, law, n_episodes=6, seed=11)
    assert len(eps_a) == len(eps_b) == 6
    for ep_a, ep_b in zip(eps_a, eps_b):
        assert len(ep_a) == len(ep_b)
        assert sum(len(r.action) for r in ep_a) <= env.budget
        for ra, rb in zip(ep_a, ep_b):
            np.testing.assert_array_equal(ra.action, rb.action)
            assert ra.reward == rb.reward
            np.testing.assert_array_equal(ra.profit_after, rb.profit_after)
        # Days advance whenever a non-empty action is played.
        assert [r.t for r in ep_a] == list(range(1, len(ep_a) + 1))


def test_rpp_logs_save_load_roundtrip():
    loc, initial, _ = points.random_instance(20, seed=5)
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, budget=8, T=5)
    episodes = rpp_logs.collect_logs(
        env, rpp_laws.get_law("SPILL", seed=1), n_episodes=4, seed=2
    )
    path = os.path.join(tempfile.gettempdir(), "strata_m0_rpp_logs.npz")
    written = rpp_logs.save_logs(path, episodes)
    loaded = rpp_logs.load_logs(written)
    assert len(loaded) == len(episodes)
    for ep_a, ep_b in zip(episodes, loaded):
        assert len(ep_a) == len(ep_b)
        for ra, rb in zip(ep_a, ep_b):
            np.testing.assert_array_equal(ra.profit_before, rb.profit_before)
            np.testing.assert_array_equal(ra.profit_after, rb.profit_after)
            np.testing.assert_array_equal(ra.visited_before, rb.visited_before)
            np.testing.assert_array_equal(ra.visited_after, rb.visited_after)
            np.testing.assert_array_equal(ra.action, rb.action)
            assert ra.reward == rb.reward
            assert ra.distance == rb.distance
            assert ra.t == rb.t
            assert ra.current_before == rb.current_before
            assert ra.current_after == rb.current_after
            assert ra.policy == rb.policy
