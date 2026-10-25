"""M1a acceptance tests for maximum-likelihood fitting.

The T2 check uses the same graph sampler as ``experiments/kt_law/common.py``
(``dataset_sampler``, whose floating seed draw differs from
``sample_instance``); with that instance, level 1 reproduces the reference
test log-likelihood on dblp (``-2008`` in ``out/t2_fit_CC2_100.out``) and
level 2 selects ``gate(g=2)`` with a log-likelihood of essentially zero.
"""

import os
import sys

import numpy as np
import pytest

from strata.core.types import Features
from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.domains.im import logs as im_logs
from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import logs as rpp_logs
from strata.domains.rpp import points
from strata.laws import dsl
from strata.laws import enumerate as en
from strata.laws import fit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KT_DIR = os.path.join(REPO, "experiments", "kt_law")
if KT_DIR not in sys.path:
    sys.path.insert(0, KT_DIR)


def _t2_graph(dataset="dblp"):
    # ``common.load_subgraph`` calls ``dataset_sampler`` which draws a float
    # seed from random.Random(seed); this is the graph behind the T2 output.
    sampler = im_graph.dataset_sampler(dataset, 100, method="rw", seed=101)
    return im_graph.build_graph(sampler(), name="t2-%s100-seed101" % dataset)


def test_loglik_matches_t2_fit_implementation():
    import t2_fit
    import laws as kt_laws

    sampler = im_graph.dataset_sampler("dblp", 60, method="rw", seed=7)
    G = im_graph.build_graph(sampler(), name="loglik-check")
    episodes = im_logs.collect_logs(
        G, im_laws.make_law("CC2", G), T=6, B=12, n_episodes=12, seed=7
    )
    reference = [
        [
            {
                "a": r.a,
                "eligible": r.eligible,
                "y": r.y,
                "seeded": r.seeded,
                "before": r.before,
                "after": r.after,
            }
            for r in ep
        ]
        for ep in episodes
    ]
    for law_name in ("WC", "UNI", "LT", "CC2", "REINF", "DECAY"):
        ref_law = kt_laws.get_law(law_name)
        our_law = im_laws.make_law(law_name, G)
        ref_ll = t2_fit.loglik(ref_law, reference, G.indeg)
        our_ll = fit.loglik(our_law, episodes, G.indeg, G)
        assert our_ll == pytest.approx(ref_ll, abs=1e-6)


@pytest.mark.parametrize(
    "dataset,expected_level1_ll",
    [("wiki", -2424.0), ("dblp", -2008.0), ("reddit", -505.0)],
)
def test_t2_level1(dataset, expected_level1_ll):
    G = _t2_graph(dataset)
    episodes = im_logs.collect_logs(
        G, im_laws.make_law("CC2", G), T=10, B=20, n_episodes=200, seed=101
    )
    n_test = int(round(0.3 * len(episodes)))
    ep_test, ep_train = episodes[:n_test], episodes[n_test:]

    law1, _, test1 = fit.fit_library(
        en.enumerate_structures(1), ep_train, ep_test, G.indeg, G, polish=False
    )
    assert law1.effect.name in ("ic", "wc", "uni")
    # Reference: out/t2_fit_CC2_100.out (30% test split).
    assert test1 == pytest.approx(expected_level1_ll, abs=1.0)


def test_t2_level2_selects_gate2():
    G = _t2_graph("dblp")
    episodes = im_logs.collect_logs(
        G, im_laws.make_law("CC2", G), T=10, B=20, n_episodes=200, seed=101
    )
    n_test = int(round(0.3 * len(episodes)))
    ep_test, ep_train = episodes[:n_test], episodes[n_test:]
    gates = [s for s in en.enumerate_structures(2) if s.kind == "gate"]
    law2, _, test2 = fit.fit_library(gates, ep_train, ep_test, G.indeg, G, polish=False)
    assert law2.effect.name == "gate"
    assert law2.params["g"] == 2
    assert abs(test2) < 0.01


def test_fit_free_under_cc2_is_the_indicator():
    G = _t2_graph()
    episodes = im_logs.collect_logs(
        G, im_laws.make_law("CC2", G), T=6, B=12, n_episodes=100, seed=101
    )
    structure = en.structure_by_name("free", 3)
    law = fit.fit_free(structure, episodes, G.indeg, G)
    shorthand = fit.fit_free(episodes, G.indeg)
    np.testing.assert_allclose(shorthand.params["table"], law.params["table"])
    via_program = fit.fit_program(structure, episodes, G.indeg, G)
    np.testing.assert_allclose(via_program.params["table"], law.params["table"])
    table = law.params["table"]
    assert table[0] == pytest.approx(0.0, abs=0.01)
    assert table[1] == pytest.approx(0.0, abs=0.01)
    assert np.all(table[2:] == pytest.approx(1.0, abs=0.01))
    assert np.all(np.diff(table) >= -1e-12)


def test_fit_product_is_at_least_as_good_as_one_program():
    G = _t2_graph()
    episodes = im_logs.collect_logs(
        G, im_laws.make_law("CC2", G), T=6, B=12, n_episodes=30, seed=3
    )
    programs = [dsl.LawProgram(dsl.Gate(2)), dsl.LawProgram(dsl.Uniform(0.1))]
    single = fit.fit_program(en.structure_by_name("gate-2", 2), episodes, G.indeg, G)
    product = fit.fit_product(programs, episodes, G.indeg, G)
    assert product.loglik >= single.loglik - 1e-6


def test_drift_fit_recovers_iid_uniform_and_gaussian():
    loc, initial, _ = points.random_instance(40, seed=3)
    env = rpp_env.RPPEnv(
        loc=loc, initial_profit=initial, budget=15, T=8, empty_day_advances=True
    )
    episodes = rpp_logs.collect_logs(
        env, rpp_laws.get_law("IID", factor=0.05), n_episodes=60, seed=11
    )
    gaussian = fit.fit_drift_program(
        en.drift_structure_by_name("gaussian", 1), episodes, env
    )
    uniform = fit.fit_drift_program(
        en.drift_structure_by_name("uniform", 1), episodes, env
    )
    assert gaussian.effect.params["sigma"] == pytest.approx(0.05 / np.sqrt(3), rel=0.15)
    assert uniform.effect.params["half_width"] == pytest.approx(0.05, rel=0.15)
    assert uniform.loglik > gaussian.loglik


def test_drift_fit_on_fitted_real_data_uses_venues():
    loc, initial, _, venues = points.make_instance_with_venues(
        "foursquare-NYC", n=30, seed=1
    )
    env = rpp_env.RPPEnv(
        loc=loc, initial_profit=initial, budget=12, T=6, empty_day_advances=True
    )
    law = rpp_laws.make_law("FITTED", dataset="foursquare-NYC", venues=venues, seed=0)
    episodes = rpp_logs.collect_logs(env, law, n_episodes=20, seed=2)
    program = fit.fit_drift_program(
        en.drift_structure_by_name("gaussian-full", 3), episodes, env
    )
    assert np.isfinite(program.loglik)
    assert program.effect.params["sigma"] > 0.0


def test_fitted_drift_program_plugs_into_the_environment():
    loc, initial, _ = points.random_instance(25, seed=5)
    env = rpp_env.RPPEnv(
        loc=loc, initial_profit=initial, budget=8, T=5, empty_day_advances=True
    )
    episodes = rpp_logs.collect_logs(
        env, rpp_laws.get_law("IID", factor=0.05), n_episodes=20, seed=5
    )
    program = fit.fit_drift_program(
        en.drift_structure_by_name("uniform", 1), episodes, env
    )
    state = env.reset(M=1)
    action = np.array([1, 2])
    before = state.profit.copy()
    rpp_env.step(env, state, action, law=program, rng=np.random.default_rng(0))
    assert state.profit[0, 1] == 0.0 and state.profit[0, 2] == 0.0
    unvisited = ~state.visited[0]
    assert np.all(state.profit[0, unvisited] >= env.profit_min)
    assert not np.array_equal(state.profit, before)
