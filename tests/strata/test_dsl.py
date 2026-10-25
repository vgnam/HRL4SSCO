"""Tests for the L1 DSL (activation effects, product law, drift laws).

Besides the unit checks, the DSL is validated against the *hidden* IM laws:
each of ``WC, UNI, LT, CC2, CC3, REINF, DECAY`` must be expressible as a
:class:`~strata.laws.dsl.LawProgram` that reproduces the hidden probabilities
exactly and drives the environment along the same trajectories under a shared
RNG seed.
"""

import networkx as nx
import numpy as np
import pytest

from strata.core.types import Features
from strata.domains.im import env as im_env
from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.laws import dsl


def _feats(n=8, a=None, E=None, indeg=None, community=None, a2=None):
    if a is None:
        a = np.arange(n, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)[None, :]
    E = a if E is None else np.asarray(E, dtype=np.float32)[None, :]
    indeg = np.full(n, 4) if indeg is None else indeg
    community = np.zeros(n, dtype=np.int64) if community is None else community
    return Features(
        a=a,
        E=E,
        a2=None if a2 is None else np.asarray(a2, np.float32)[None, :],
        eligible=np.ones((1, n), bool),
        indeg=indeg,
        community=community,
    )


def test_basic_effects():
    f = _feats(a=[0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(
        dsl.IndependentCascade(0.5).prob(f)[0], 1.0 - 0.5 ** f.a[0]
    )
    np.testing.assert_allclose(
        dsl.WeightedCascade(0.25).prob(f)[0], 1.0 - 0.75 ** f.E[0]
    )
    np.testing.assert_allclose(dsl.Uniform(0.1).prob(f)[0], 1.0 - 0.9 ** f.E[0])
    lt = dsl.LinearThreshold(2.0).prob(f)[0]
    np.testing.assert_allclose(lt, np.clip(2.0 * f.a[0] / f.indeg, 0, 1))
    np.testing.assert_array_equal(
        dsl.Gate(2).prob(f)[0], (f.a[0] >= 2).astype(np.float32)
    )
    soft = dsl.SoftGate(1.5, 4.0).prob(f)[0]
    assert np.isclose(soft[0], 0.0)
    assert 0.0 < soft[2] < 1.0
    ramp = dsl.Ramp(0.0, None).prob(f)[0]
    np.testing.assert_allclose(ramp, np.clip(f.a[0] / f.indeg, 0, 1))


def test_weighted_cascade_per_indeg():
    f = _feats(a=[3, 3, 3, 3], E=[3, 3, 3, 3], indeg=np.array([1, 2, 3, 4]))
    law = dsl.WeightedCascade(w=1.0, per_indeg=True)
    expected = 1.0 - np.power(1.0 - 1.0 / f.indeg, 3)
    np.testing.assert_allclose(law.prob(f)[0], expected)
    assert np.all(law.prob(f) <= 1.0)


def test_memory_and_two_hop():
    f = _feats(a=[1, 1, 1, 1], E=[4, 4, 4, 4], a2=[3, 3, 3, 3])
    mem = dsl.Memory(dsl.Uniform(0.1), rho=0.5)
    assert mem.rho == 0.5
    np.testing.assert_allclose(mem.prob(f), dsl.Uniform(0.1).prob(f))
    np.testing.assert_allclose(mem.accumulate(np.ones((1, 4)), np.ones((1, 4))), 1.5)
    th = dsl.TwoHop(0.1)
    np.testing.assert_allclose(th.prob(f)[0], 1.0 - 0.9 ** (f.a[0] + f.a2[0]))
    assert not dsl.WeightedCascade(0.1).two_hop and th.two_hop


def test_community_and_free():
    community = np.array([0, 0, 1, 1])
    f = _feats(a=[2, 2, 2, 2], E=[2, 2, 2, 2], community=community)
    law = dsl.Community(dsl.WeightedCascade(1.0), dsl.Gate(2))
    p = law.prob(f)[0]
    np.testing.assert_allclose(p[:2], 1.0 - 0.0**2)
    np.testing.assert_allclose(p[2:], 1.0)

    table = np.array([0.0, 0.0, 0.5, 1.0])
    free = dsl.Free(table)
    np.testing.assert_allclose(free.prob(_feats(a=[0, 1, 2, 3]))[0], table)
    with pytest.raises(ValueError):
        dsl.Free([0.0, 0.5, 0.2])


def test_precondition_and_product():
    f = _feats(a=[1, 2, 3, 4])
    pre = np.array([False, True, True, False])
    law = dsl.LawProgram(dsl.Gate(2), precondition=pre)
    p = law.prob(f)[0]
    assert p[0] == 0.0 and p[1] == 1.0 and p[2] == 1.0 and p[3] == 0.0

    single = dsl.ProductLaw([dsl.LawProgram(dsl.Uniform(0.2))])
    np.testing.assert_allclose(single.prob(f), dsl.Uniform(0.2).prob(f))
    product = dsl.ProductLaw(
        [dsl.LawProgram(dsl.Uniform(0.5)), dsl.LawProgram(dsl.Gate(2))],
        theta=[2.0, 1.0],
    )
    p = product.prob(f)[0]
    assert p[0] == pytest.approx(0.0, abs=1e-9)
    assert p[2] == pytest.approx(1.0, abs=1e-9)


def test_drift_effects_and_program():
    feats = dsl.DriftFeatures(
        profit=np.array([[0.9, 0.8, 0.7]]),
        visited=np.zeros((1, 3), dtype=bool),
        t=np.array([2]),
        distance=np.array([[1.0, 2.0, 3.0]]),
        region=np.array([0, 1, 1]),
    )
    g = dsl.GaussianDrift(
        mu=0.0, sigma=0.1, time_coef=0.01, region_mu=np.array([0.0, 0.02])
    )
    mu = g.mean(feats)
    np.testing.assert_allclose(mu[0, 0], 0.01)
    np.testing.assert_allclose(mu[0, 1], 0.03)
    assert g.logpdf(np.zeros((1, 3)), feats).shape == (1, 3)
    rng = np.random.default_rng(0)
    assert g.sample(feats, rng).shape == (1, 3)

    u = dsl.UniformDrift(mu=0.0, half_width=0.1)
    assert np.all(np.abs(u.sample(feats, rng) - u.mean(feats)) <= 0.1 + 1e-12)
    m = dsl.MixtureDrift([-0.1, 0.1], [0.02, 0.02], [0.5, 0.5])
    assert np.all(np.isfinite(m.logpdf(np.zeros((1, 3)), feats)))

    program = dsl.DriftProgram(g)
    env = type(
        "E",
        (),
        {"dist": np.zeros((3, 3)), "regions": np.array([0, 1, 1]), "max_dist": 1.0},
    )()
    state = type(
        "S",
        (),
        {
            "profit": np.array([[0.9, 0.8, 0.7]]),
            "visited": np.zeros((1, 3), bool),
            "t": 2,
        },
    )()
    delta = program.drift(env, state, np.zeros((1, 3), bool), rng)
    assert delta.shape == (1, 3)


# =========================================================================
# Hidden IM laws as DSL programs
# =========================================================================


def _im_programs():
    """DSL equivalents of the hidden benchmark laws (same mechanisms)."""
    return {
        "WC": dsl.LawProgram(dsl.WeightedCascade(w=1.0, per_indeg=True)),
        "UNI": dsl.LawProgram(dsl.Uniform(0.05)),
        "LT": dsl.LawProgram(dsl.LinearThreshold(1.0)),
        "CC2": dsl.LawProgram(dsl.Gate(2)),
        "CC3": dsl.LawProgram(dsl.Gate(3)),
        "REINF": dsl.LawProgram(
            dsl.Memory(dsl.WeightedCascade(w=1.0, per_indeg=True), rho=1.0)
        ),
        "DECAY": dsl.LawProgram(dsl.Memory(dsl.Uniform(0.05), rho=0.5)),
    }


IM_EQUIVALENT_LAWS = ("WC", "UNI", "LT", "CC2", "CC3", "REINF", "DECAY")


def _random_features(seed, M, n):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 7, size=(M, n)).astype(np.float32)
    E = (a + rng.random((M, n)) * 3.0).astype(np.float32)
    indeg = rng.integers(1, 12, size=n).astype(np.int64)
    community = rng.integers(0, 4, size=n).astype(np.int64)
    return Features(
        a=a,
        E=E,
        a2=None,
        eligible=np.ones((M, n), bool),
        indeg=indeg,
        community=community,
    )


@pytest.mark.parametrize("law_name", IM_EQUIVALENT_LAWS)
@pytest.mark.parametrize("M,n,seed", [(1, 7, 0), (2, 19, 1), (5, 31, 2)])
def test_hidden_im_law_equals_dsl_program(law_name, M, n, seed):
    hidden = im_laws.make_law(law_name)
    program = _im_programs()[law_name]
    feats = _random_features(seed, M, n)
    p_hidden = hidden.prob(feats)
    p_program = program.prob(feats)
    np.testing.assert_allclose(p_program, p_hidden, rtol=0.0, atol=1e-12)
    assert program.rho == hidden.rho
    assert program.two_hop == hidden.two_hop


@pytest.fixture(scope="module")
def rollout_graph():
    G = nx.gnp_random_graph(50, 6.0 / 50.0, seed=7, directed=True)
    return im_graph.build_graph(G, name="dsl-rollout")


@pytest.mark.parametrize("law_name", IM_EQUIVALENT_LAWS)
def test_hidden_im_law_rollout_equals_dsl_program(law_name, rollout_graph):
    G = rollout_graph
    n = G.adj.shape[0]
    M = 3
    hidden = im_laws.make_law(law_name, G)
    program = _im_programs()[law_name]

    state_h = im_env.make_state(G, M=M, budget_left=100)
    state_d = im_env.make_state(G, M=M, budget_left=100)
    rng_h = np.random.default_rng(2024)
    rng_d = np.random.default_rng(2024)
    rng_seeds = np.random.default_rng(77)

    for t in range(6):
        if t == 4:
            seeds = np.stack([rng_seeds.integers(0, n, size=4) for _ in range(M)])
            seeds[:, -2:] = -1
        else:
            seeds = rng_seeds.integers(0, n, size=3 if t % 2 == 0 else 5)

        p_h = im_env.next_probabilities(G, hidden, state_h)
        p_d = im_env.next_probabilities(G, program, state_d)
        np.testing.assert_allclose(p_d, p_h, rtol=0.0, atol=1e-12)

        _, reward_h, info_h = im_env.step(
            G, hidden, state_h, seeds, rng_h, return_info=True
        )
        _, reward_d, info_d = im_env.step(
            G, program, state_d, seeds, rng_d, return_info=True
        )

        np.testing.assert_array_equal(state_d.status, state_h.status)
        np.testing.assert_array_equal(state_d.E, state_h.E)
        np.testing.assert_array_equal(reward_d, reward_h)
        for key in ("a", "eligible", "new", "p", "seeded"):
            np.testing.assert_array_equal(info_d[key], info_h[key])
