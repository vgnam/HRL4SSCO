"""M1b acceptance tests: L1 hypothesis set and identifiability certificate.

The IM tests reproduce K1 (``experiments/kt_law/k1_rarity.py``): under a hidden
CC2 law and ``indeg`` behaviour logs the gate thresholds 2..5 are tied and
indistinguishable, while mixed-policy logs identify ``gate(2)`` only.  The RPP
tests check that the exact IID family and its Gaussian approximation coexist in
the hypothesis set while loud trend/spill programs are refuted, and that a
trend program dominates the prior under hidden TREND logs.
"""

from __future__ import annotations

import numpy as np
import pytest

from strata.core.types import Round
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
from strata.laws import hypotheses as hyp

N_TEST_FRACTION = 0.3


# =========================================================================
# IM helpers
# =========================================================================


def _wiki_cc2_instance():
    sampler = im_graph.dataset_sampler("wiki", 100, method="rw", seed=101)
    graph = im_graph.build_graph(sampler(), name="m1b-wiki100-seed101")
    law = im_laws.make_law("CC2", graph)
    episodes = im_logs.collect_logs(graph, law, T=10, B=20, n_episodes=200, seed=101)
    return graph, episodes


def _fit_level2(episodes, graph):
    """Fit every level-2 structure exactly as the pipeline does.

    Each named gate structure pins its own threshold (``Structure.fixed``), so
    gate-2..gate-5 stay distinct candidates and the hypothesis set can certify
    their tie on K1 without any test-side grid.
    """
    return [
        fit.fit_program(structure, episodes, graph.indeg, graph, polish=False)
        for structure in en.enumerate_structures(2)
    ]


def _gate_index(hset):
    """Map gate threshold -> hypothesis index."""
    out = {}
    for i, h in enumerate(hset.hyps):
        if getattr(h.law, "effect", None) is not None and h.law.effect.name == "gate":
            out[int(h.law.params["g"])] = i
    return out


# =========================================================================
# IM unit tests
# =========================================================================


def _synthetic_round(a, y):
    a = np.asarray(a, dtype=np.float32)
    y = np.asarray(y, dtype=bool)
    n = a.size
    return [
        Round(
            a=a,
            eligible=np.ones(n, dtype=bool),
            y=y,
            seeded=np.zeros(n, dtype=bool),
            before=np.zeros(n, dtype=np.int8),
            after=np.where(y, 1, 0).astype(np.int8),
            policy="synthetic",
        )
    ]


def test_tie_groups_cap_and_prior_hook():
    # ``a`` is 0 or 1, so gate(2) and gate(3) predict the same all-zero log.
    episodes = [_synthetic_round([0, 1, 0, 1], [0, 0, 0, 0])]
    gate2 = dsl.LawProgram(dsl.Gate(2))
    gate3 = dsl.LawProgram(dsl.Gate(3))
    uni_a = dsl.LawProgram(dsl.Uniform(1e-3))
    uni_b = dsl.LawProgram(dsl.Uniform(5e-4))
    indeg = np.ones(4, dtype=np.float64)

    hset = hyp.build_hypothesis_set(
        [gate2, gate3, uni_a, uni_b], episodes, eps=1.0, cap=None, indeg=indeg
    )
    assert len(hset.hyps) == 4
    assert hset.hyps[0].tie_group == hset.hyps[1].tie_group
    assert hset.hyps[2].tie_group != hset.hyps[3].tie_group

    capped = hyp.build_hypothesis_set(
        [gate2, gate3, uni_a, uni_b], episodes, eps=1.0, cap=2, indeg=indeg
    )
    # The top tie group fits; the lower group is dropped whole.
    assert len(capped.hyps) == 2
    assert all(h.law.effect.name == "gate" for h in capped.hyps)

    doubled = hyp.build_hypothesis_set(
        [gate2, gate3, uni_a, uni_b],
        episodes,
        eps=1.0,
        cap=None,
        indeg=indeg,
        prior_fn=lambda h: 2.0,
    )
    np.testing.assert_allclose(
        [h.prior for h in doubled.hyps], 2.0 * np.asarray([h.prior for h in hset.hyps])
    )


def test_identifiability_report_lists_ties_and_zero_pairs():
    episodes = [_synthetic_round([0, 1, 0, 1], [0, 0, 0, 0])]
    indeg = np.ones(4, dtype=np.float64)
    hset = hyp.build_hypothesis_set(
        [dsl.LawProgram(dsl.Gate(2)), dsl.LawProgram(dsl.Gate(3))],
        episodes,
        eps=1e-6,
        cap=None,
        indeg=indeg,
    )
    report = hyp.identifiability_report(hset)
    assert report["m"] == 2
    assert report["tie_groups"] == [[0, 1]]
    assert report["zero_discriminating"] == [(0, 1)]
    assert int(hset.discriminating[0, 1]) == 0


def test_discriminating_events_counts_probability_differences():
    episodes = [_synthetic_round([0, 1, 2, 1], [0, 0, 1, 1])]
    indeg = np.ones(4, dtype=np.float64)
    gate2 = dsl.LawProgram(dsl.Gate(2))
    gate3 = dsl.LawProgram(dsl.Gate(3))
    # gate(2) vs gate(3) differ exactly at the node with a = 2.
    assert hyp.discriminating_events(gate2, gate3, episodes, tau=1e-3, indeg=indeg) == 1
    # Two non-deterministic laws with different rates are separated by KL
    # at every eligible event with a >= 1 (a = 0 always gives p = 0).
    uni_a = dsl.LawProgram(dsl.Uniform(0.01))
    uni_b = dsl.LawProgram(dsl.Uniform(0.9))
    assert hyp.discriminating_events(uni_a, uni_b, episodes, tau=1e-3, indeg=indeg) == 3


# =========================================================================
# IM acceptance: K1
# =========================================================================


@pytest.fixture(scope="module")
def im_instance():
    return _wiki_cc2_instance()


@pytest.fixture(scope="module")
def im_mixed_fitted(im_instance):
    graph, episodes = im_instance
    n_test = int(round(N_TEST_FRACTION * len(episodes)))
    train = episodes[n_test:]
    return graph, episodes, n_test, _fit_level2(train, graph)


def test_k1_indeg_logs_tie_gates_2_to_5(im_instance):
    graph, episodes = im_instance
    n_test = int(round(N_TEST_FRACTION * len(episodes)))
    behaviour = im_logs.BEHAVIOUR
    indeg_train = [
        episodes[i]
        for i in range(n_test, len(episodes))
        if behaviour[i % len(behaviour)] == "indeg"
    ]
    assert len(indeg_train) > 0

    fitted = _fit_level2(indeg_train, graph)
    hset = hyp.build_hypothesis_set(
        fitted, indeg_train, eps=1e-6, cap=64, domain="im", graph=graph
    )
    gates = _gate_index(hset)
    assert {2, 3, 4, 5} <= set(gates)
    tie_groups = {hset.hyps[gates[g]].tie_group for g in (2, 3, 4, 5)}
    assert len(tie_groups) == 1

    for g in (2, 3, 4, 5):
        for h in (2, 3, 4, 5):
            if g < h:
                assert int(hset.discriminating[gates[g], gates[h]]) == 0

    report = hyp.identifiability_report(hset)
    tied = next(group for group in report["tie_groups"] if gates[2] in group)
    assert {gates[g] for g in (2, 3, 4, 5)} <= set(tied)


def test_k1_mixed_logs_identify_only_gate_2(im_mixed_fitted):
    graph, episodes, n_test, fitted = im_mixed_fitted
    validation = episodes[:n_test]
    hset = hyp.build_hypothesis_set(
        fitted, validation, eps=1e-6, cap=64, domain="im", graph=graph
    )
    gates = _gate_index(hset)
    assert set(gates) == {2}


def test_ic_family_excluded_at_eps_10_once_gates_available(im_mixed_fitted):
    graph, episodes, n_test, fitted = im_mixed_fitted
    validation = episodes[:n_test]
    hset = hyp.build_hypothesis_set(
        fitted, validation, eps=10.0, cap=64, domain="im", graph=graph
    )
    kinds = {
        getattr(getattr(h.law, "structure", None), "kind", None) for h in hset.hyps
    }
    assert not (kinds & {"ic", "wc", "uni", "lt"})
    gates = _gate_index(hset)
    assert 2 in gates


# =========================================================================
# RPP acceptance
# =========================================================================


@pytest.fixture(scope="module")
def rpp_instance():
    loc, initial, factor = points.random_instance(40, seed=3)
    env = rpp_env.RPPEnv(
        loc=loc,
        initial_profit=initial,
        profit_factor=factor,
        budget=15,
        T=8,
        empty_day_advances=True,
    )
    return env, factor


def _collect_and_split(env, law, n_episodes=60, seed=11):
    episodes = rpp_logs.collect_logs(env, law, n_episodes=n_episodes, seed=seed)
    n_train = int(round(0.7 * len(episodes)))
    return episodes[:n_train], episodes[n_train:]


def _loud_drift_programs(env, factor):
    regions = getattr(env, "regions", None)
    if regions is None:
        regions = points.spatial_clusters(env.loc, 4, seed=0)
        env.regions = regions
    k = int(regions.max()) + 1
    region_mu = np.asarray(
        [0.5 if r % 2 == 0 else -0.5 for r in range(k)], dtype=np.float64
    )
    signs = np.where(np.arange(env.n) % 2 == 0, 0.5, -0.5).astype(np.float64)
    sigma = factor / np.sqrt(3.0)
    trend = dsl.DriftProgram(
        dsl.GaussianDrift(
            mu=0.0, sigma=sigma, region_mu=region_mu, distance_scale=env.max_dist
        )
    )
    spill = dsl.DriftProgram(
        dsl.GaussianDrift(
            mu=0.0, sigma=sigma, spill_coef=signs, distance_scale=env.max_dist
        )
    )
    return trend, spill


def test_rpp_iid_keeps_iid_and_gaussian_refutes_loud_trend_spill(rpp_instance):
    env, factor = rpp_instance
    law = rpp_laws.get_law("IID", factor=factor)
    train, validation = _collect_and_split(env, law)

    iid = fit.fit_drift_program(en.drift_structure_by_name("uniform", 1), train, env)
    gaussian = fit.fit_drift_program(
        en.drift_structure_by_name("gaussian", 1), train, env
    )
    loud_trend, loud_spill = _loud_drift_programs(env, factor)

    hset = hyp.build_hypothesis_set(
        [iid, gaussian, loud_trend, loud_spill],
        validation,
        eps=5.0,
        cap=64,
        domain="rpp",
        env=env,
    )
    laws = [h.law for h in hset.hyps]
    assert any(law is iid for law in laws)
    assert any(law is gaussian for law in laws)
    assert all(law is not loud_trend for law in laws)
    assert all(law is not loud_spill for law in laws)


def test_rpp_trend_has_highest_prior_weight(rpp_instance):
    env, factor = rpp_instance
    law = rpp_laws.get_law("TREND", factor=factor, trend_strength=0.5, seed=0)
    train, validation = _collect_and_split(env, law)

    iid = fit.fit_drift_program(en.drift_structure_by_name("uniform", 1), train, env)
    gaussian = fit.fit_drift_program(
        en.drift_structure_by_name("gaussian", 1), train, env
    )
    trend = fit.fit_drift_program(
        en.drift_structure_by_name("gaussian-region", 2), train, env
    )

    hset = hyp.build_hypothesis_set(
        [iid, gaussian, trend],
        validation,
        eps=5.0,
        cap=64,
        domain="rpp",
        env=env,
    )
    assert any(h.law is trend for h in hset.hyps)
    best = max(hset.hyps, key=lambda h: h.prior)
    assert best.law is trend
    trend_hyp = next(h for h in hset.hyps if h.law is trend)
    assert trend_hyp.loglik == pytest.approx(max(h.loglik for h in hset.hyps))


def test_rpp_discriminating_events_are_positive_between_distributions(rpp_instance):
    env, factor = rpp_instance
    law = rpp_laws.get_law("IID", factor=factor)
    _, validation = _collect_and_split(env, law)
    iid = fit.fit_drift_program(
        en.drift_structure_by_name("uniform", 1), validation[:5], env
    )
    gaussian = fit.fit_drift_program(
        en.drift_structure_by_name("gaussian", 1), validation[:5], env
    )
    count = hyp.discriminating_events(
        iid, gaussian, validation, tau=1e-3, domain="rpp", env=env
    )
    assert count > 0
