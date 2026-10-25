"""M3a: L3 objective scalarisation and finite belief over hypotheses.

The IM test observes a flat (hub + even) episode under the hidden CC2 law and
checks that the belief collapses onto CC2 within three rounds.  The RPP test
fits the IID and TREND drift families on behaviour logs and checks that the
TREND posterior exceeds 0.9 within five days of a closed-loop episode.
"""

from __future__ import annotations

import numpy as np
import pytest

from strata.abstraction.operators import Harvest, Hub
from strata.core.types import Hypothesis, HypothesisSet
from strata.domains.im import env as im_env
from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import logs as rpp_logs
from strata.domains.rpp import points
from strata.laws import enumerate as en
from strata.laws import fit
from strata.planning.belief import Belief
from strata.planning.objectives import aggregate
from strata.planning.rollout_planner import allocation_k


# =========================================================================
# objectives: hand-computed cases
# =========================================================================


def test_mean_weighted_and_zero_weights():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 0.25, 0.25, 0.0])
    # 0.5*1 + 0.25*2 + 0.25*3 = 1.75
    assert aggregate(values, weights, "mean") == pytest.approx(1.75)
    # A zero-weight hypothesis is ignored by the robust modes.
    assert aggregate(values, weights, "maxmin") == pytest.approx(1.0)
    assert aggregate(values, weights, "optimistic") == pytest.approx(3.0)
    # Degenerate all-zero weights fall back to a uniform belief.
    zeros = np.zeros(4)
    assert aggregate(values, zeros, "mean") == pytest.approx(2.5)
    assert aggregate(values, zeros, "maxmin") == pytest.approx(1.0)
    assert aggregate(values, zeros, "optimistic") == pytest.approx(4.0)


def test_cvar_interpolates_the_boundary_atom():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 0.25, 0.25, 0.0])
    # Sorted atoms: 1 (0.5), 2 (0.25), 3 (0.25).  alpha=0.5 ends exactly on
    # the first atom; alpha=0.75 includes all of atom 2; alpha=0.6 takes 0.1
    # of atom 2.
    assert aggregate(values, weights, "cvar", alpha=0.5) == pytest.approx(1.0)
    assert aggregate(values, weights, "cvar", alpha=0.75) == pytest.approx(
        (0.5 * 1.0 + 0.25 * 2.0) / 0.75
    )
    assert aggregate(values, weights, "cvar", alpha=0.6) == pytest.approx(
        (0.5 * 1.0 + 0.1 * 2.0) / 0.6
    )
    # alpha=1 is the weighted mean.
    assert aggregate(values, weights, "cvar", alpha=1.0) == pytest.approx(1.75)


def test_ties_and_leading_dimensions():
    values = np.array([2.0, 2.0, 2.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert aggregate(values, weights, "mean") == pytest.approx(2.0)
    assert aggregate(values, weights, "cvar", alpha=0.3) == pytest.approx(2.0)
    assert aggregate(values, weights, "maxmin") == pytest.approx(2.0)
    assert aggregate(values, weights, "optimistic") == pytest.approx(2.0)

    batch = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = aggregate(batch, np.array([0.75, 0.25]), "mean")
    np.testing.assert_allclose(out, [1.25, 3.25, 5.25])
    out_cvar = aggregate(batch, np.array([0.75, 0.25]), "cvar", alpha=0.5)
    np.testing.assert_allclose(out_cvar, [1.0, 3.0, 5.0])


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        aggregate([1.0, 2.0], [0.5, 0.5], "median")


# =========================================================================
# belief bookkeeping
# =========================================================================


def test_entropy_map_and_sample():
    laws = [im_laws.make_law("WC"), im_laws.make_law("CC2"), im_laws.make_law("CC3")]
    hset = HypothesisSet(
        [Hypothesis(law, 0.0, 1.0, i) for i, law in enumerate(laws)],
        np.zeros((3, 3), dtype=np.int64),
    )
    belief = Belief(hset, "im")
    assert belief.entropy() == pytest.approx(np.log(3.0))
    assert belief.map_index() == 0
    rng = np.random.default_rng(0)
    draws = [belief.sample(rng) for _ in range(50)]
    assert set(draws) <= {0, 1, 2}
    assert len(set(draws)) > 1


# =========================================================================
# IM acceptance: observe CC2 and identify it
# =========================================================================


@pytest.fixture(scope="module")
def im_belief_instance():
    graph = im_graph.sample_instance("dblp", 300, seed=101)
    laws = [
        Hypothesis(im_laws.make_law(name, graph), 0.0, 1.0, i)
        for i, name in enumerate(("WC", "CC2", "CC3"))
    ]
    hset = HypothesisSet(laws, np.zeros((3, 3), dtype=np.int64))
    return graph, hset


def test_im_belief_identifies_cc2_within_three_rounds(im_belief_instance):
    graph, hset = im_belief_instance
    true = im_laws.make_law("CC2", graph)
    belief = Belief(hset, "im")
    state = im_env.make_state(graph, M=1, budget_left=20)
    hub = Hub()
    full = np.ones(graph.adj.shape[0], dtype=bool)
    rng = np.random.default_rng(7)

    for t in range(3):
        k = allocation_k("even", t, 10, int(state.budget_left))
        seeds = hub.refine(graph, true, state, k, full)
        before = state.clone()
        _, _, info = im_env.step(graph, true, state, seeds, rng)
        belief.update(graph, before, seeds, state, info)

    assert belief.weights[1] >= 0.95
    assert belief.map_index() == 1


# =========================================================================
# RPP acceptance: observe TREND and identify it
# =========================================================================


@pytest.fixture(scope="module")
def rpp_belief_instance():
    loc, initial, factor = points.random_instance(40, seed=3)
    env = rpp_env.RPPEnv(
        loc=loc,
        initial_profit=initial,
        profit_factor=factor,
        budget=12,
        T=6,
        empty_day_advances=True,
    )
    true = rpp_laws.get_law("TREND", factor=factor, seed=0)
    train = rpp_logs.collect_logs(env, true, n_episodes=40, seed=11)
    iid = fit.fit_drift_program(en.drift_structure_by_name("uniform", 1), train, env)
    trend = fit.fit_drift_program(
        en.drift_structure_by_name("gaussian-region", 2), train, env
    )
    hset = HypothesisSet(
        [Hypothesis(iid, 0.0, 1.0, 0), Hypothesis(trend, 0.0, 1.0, 1)],
        np.zeros((2, 2), dtype=np.int64),
    )
    return env, true, hset


def test_rpp_belief_identifies_trend_within_five_days(rpp_belief_instance):
    env, true, hset = rpp_belief_instance
    belief = Belief(hset, "rpp")
    state = env.reset(M=1)
    harvest = Harvest()
    full = np.ones(env.n, dtype=bool)
    rng = np.random.default_rng(123)

    for _ in range(5):
        k = allocation_k("even", int(state.t) - 1, env.T, int(state.budget_left[0]))
        action = harvest.refine(env, true, state, k, full)
        before = state.clone()
        _, _, done, info = rpp_env.step(
            env, state, action, law=true, rng=rng, return_info=True
        )
        belief.update(env, before, action, state, info)
        if done:
            break

    assert belief.weights[1] >= 0.9
    assert belief.map_index() == 1
