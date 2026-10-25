"""Tests for ``strata.laws.features``.

The live-state feature builder must reproduce the statistics that
``domains.im.env.step`` puts in its ``info`` dict, including the post-seeding
eligibility mask and the exposure recursion of REINF/DECAY/2HOP.  The logged
round path is checked against a manual recursion on a small episode.
"""

import numpy as np
import pytest

from strata.core.types import STATUS_INACTIVE
from strata.domains.im import env as im_env
from strata.domains.im import hidden_laws as im_laws
from strata.domains.im import logs as im_logs
from strata.laws import features


@pytest.fixture(scope="module")
def graph():
    from strata.domains.im import graph as im_graph

    G = im_graph.build_graph(
        im_graph.dataset_sampler("dblp", 60, method="rw", seed=101)(),
        name="features-dblp60",
    )
    return G


@pytest.mark.parametrize("law_name", ["WC", "REINF", "DECAY", "2HOP"])
def test_compute_features_matches_env_info(graph, law_name):
    law = im_laws.make_law(law_name, graph)
    state = im_env.make_state(graph, M=3, budget_left=20)
    rng = np.random.default_rng(4)
    seeds = np.array([1, 2, 3, 4])

    feats = features.compute_features(graph, state, seeds, law)
    _, _, info = im_env.step(graph, law, state, seeds, rng, return_info=True)

    np.testing.assert_allclose(feats.a, info["a"])
    np.testing.assert_allclose(
        feats.E, im_env.accumulate(law, np.zeros_like(info["a"]), info["a"])
    )
    np.testing.assert_array_equal(feats.eligible, info["eligible"])
    assert feats.a2 is not None if getattr(law, "two_hop", False) else feats.a2 is None
    assert not feats.eligible[:, seeds].any()


def test_compute_features_without_seeds(graph):
    law = im_laws.make_law("WC", graph)
    state = im_env.make_state(graph, M=2, budget_left=0)
    state.status[:, :3] = 0
    feats = features.compute_features(graph, state, None, law)
    assert np.all(feats.eligible)


def test_episode_features_exposure_recursion(graph):
    law = im_laws.make_law("DECAY", graph)
    episodes = im_logs.collect_logs(graph, law, T=5, B=10, n_episodes=2, seed=9)
    for episode in episodes:
        got = features.episode_features(graph, episode, law)
        E = np.zeros(graph.adj.shape[0], dtype=np.float32)
        for feats, round_ in zip(got, episode):
            E = round_.a + law.rho * E
            np.testing.assert_allclose(feats.E[0], E)
            np.testing.assert_array_equal(feats.eligible[0], round_.eligible)
            y = round_.y & round_.eligible
            E = np.where(y, 0.0, E)
            E[round_.after != STATUS_INACTIVE] = 0.0


def test_round_features_direct_two_hop(graph):
    law = im_laws.make_law("2HOP", graph)
    episode = im_logs.collect_logs(graph, law, T=3, B=6, n_episodes=1, seed=2)[0]
    feats = features.episode_features(graph, episode, law, direct=True)
    for f, r in zip(feats, episode):
        before = r.before
        act = (before == 1) | r.seeded
        expected = (
            act.astype(np.float32)[None, :] @ (graph.adj + graph.adj @ graph.adj)
        )[0]
        np.testing.assert_allclose(f.a[0] + f.a2[0], expected)
