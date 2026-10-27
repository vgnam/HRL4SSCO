"""Classic adaptive IM baselines (strata.baselines.im_classic)."""

from __future__ import annotations

import numpy as np
import pytest

from strata.baselines import im_classic as ic
from strata.domains.im import env as im_env
from strata.eval import protocol


@pytest.fixture(scope="module")
def wc():
    return protocol.make_instance("im", dataset="dblp", n=120, graph_seed=0, law="WC", T=5, B=10)


@pytest.mark.parametrize("cls", [ic.DegreeDiscountPolicy, ic.RRSetPolicy, ic.AdaptiveGreedyPolicy])
def test_policies_pick_distinct_inactive_seeds_within_budget(wc, cls):
    policy = cls("even", seed=0)
    policy.reset(wc, np.random.default_rng(0))
    state = im_env.make_state(wc.graph, M=1, budget_left=wc.B)
    im_env.step(wc.graph, wc.law, state, np.asarray([0, 1, 2]), np.random.default_rng(1))
    seeds = policy.act(state, 1)
    assert 0 < len(seeds) <= wc.B
    assert len(set(seeds.tolist())) == len(seeds)
    assert np.all(state.status[0, seeds] == 0)  # only still-inactive nodes


def test_adaptive_greedy_first_seed_maximises_the_crn_estimate(wc):
    policy = ic.AdaptiveGreedyPolicy("normal", seed=0, n_paths=8)
    policy.reset(wc, np.random.default_rng(3))
    state = im_env.make_state(wc.graph, M=1, budget_left=wc.B)
    first = int(policy.select(state, 0, 1)[0])
    # brute force with the same uniforms
    policy.rng = np.random.default_rng(3)
    H, n, R = wc.T, wc.graph.adj.shape[0], policy.n_paths
    U = policy.rng.random((H, R, n))
    cands = np.arange(n)
    seeds = np.repeat(cands, R)[:, None]
    tot = ic._rollout_crn(wc.graph, wc.law, state, seeds, np.tile(U, (1, n, 1)))
    assert first == int(np.argmax(tot.reshape(n, R).mean(axis=1)))


def test_starred_variants_select_a_schedule_and_run(wc):
    for method in ("ddic*", "imm-rr*"):
        policy = ic.make_policy(method, wc, seed=0, selection_episodes=1)
        assert policy.selected_alloc in ("even", "normal", "static")
        values = protocol.evaluate(policy, wc, episodes=1, seed=0)
        assert np.isfinite(values).all() and values[0] > 0
