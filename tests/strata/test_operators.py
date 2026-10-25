"""M2a acceptance tests for the L2 macro-operators (IM ports + RPP routes).

The IM check is the plan's exact requirement: ``Hub``/``Gate``/``Hedge``
``refine`` must return the same seed sets as ``ktb_hier.ground`` on the same
input.  The remaining checks cover region containment, de-duplication, budget
``k``, the empty ``Wait`` action and the RPP distance constraint.
"""

import os
import sys

import networkx as nx
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KT_DIR = os.path.join(REPO, "experiments", "kt_law")
if KT_DIR not in sys.path:
    sys.path.insert(0, KT_DIR)

import ktb_hier  # noqa: E402
import laws as kt_laws  # noqa: E402

from strata.abstraction import operators as ops  # noqa: E402
from strata.abstraction import regions  # noqa: E402
from strata.core.types import STATUS_ACTIVE, SimState  # noqa: E402
from strata.domains.im import env as im_env  # noqa: E402
from strata.domains.im import graph as im_graph  # noqa: E402
from strata.domains.im import hidden_laws as im_laws  # noqa: E402
from strata.domains.rpp import env as rpp_env  # noqa: E402
from strata.domains.rpp import hidden_laws as rpp_laws  # noqa: E402
from strata.domains.rpp import points  # noqa: E402


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def im_instance():
    G = im_graph.build_graph(
        nx.gnp_random_graph(40, 0.15, seed=4, directed=True), name="tiny"
    )
    rng = np.random.default_rng(0)
    n = G.adj.shape[0]
    M = 3
    status = rng.integers(0, 3, (M, n)).astype(np.int8)
    E = rng.random((M, n)).astype(np.float32)
    return G, status, E


@pytest.fixture(scope="module")
def rpp_instance():
    loc, profit, _ = points.random_instance(60, seed=3)
    env = rpp_env.RPPEnv(loc, profit, budget=40, T=8, seed=0, empty_day_advances=True)
    return env, rpp_laws.IID(seed=0)


# ------------------------------------------------------------------ IM: parity


def _our_op(kind, gate):
    if kind == "hub":
        return ops.Hub()
    if kind == "gate":
        return ops.Gate(gate)
    if kind == "hedge":
        return ops.Hedge(gate)
    raise KeyError(kind)


@pytest.mark.parametrize("law_name", ["WC", "CC2", "CC3", "LT", "REINF"])
@pytest.mark.parametrize(
    "kind_gate", [("hub", 0), ("gate", 2), ("gate", 3), ("hedge", 2)]
)
def test_im_refine_matches_ktb_hier_ground(im_instance, law_name, kind_gate):
    G, status, E = im_instance
    kind, gate = kind_gate
    n = G.adj.shape[0]
    mask = np.ones(n, dtype=bool)
    k = 7

    op = _our_op(kind, gate)
    our_law = im_laws.make_law(law_name, G)
    state = SimState(status.copy(), E.copy(), 0, 100)
    ours = op.refine(G, our_law, state, k, mask)

    ref_name = kind if kind == "hub" else "%s-%d" % (kind, gate)
    ref = ktb_hier.ground(
        ref_name,
        mask,
        k,
        kt_laws.get_law(law_name),
        G.adj,
        G.adjT,
        G.indeg,
        status,
        E,
    )
    assert np.array_equal(ours, ref)


def test_im_refine_constraints(im_instance):
    G, status, E = im_instance
    n = G.adj.shape[0]
    M = status.shape[0]
    mask = G.community == np.unique(G.community)[0]
    state = SimState(status.copy(), E.copy(), 0, 100)
    law = im_laws.make_law("CC2", G)

    for op in [ops.Hub(), ops.Gate(2), ops.Hedge(2), ops.Prime()]:
        k = 5
        seeds = op.refine(G, law, state, k, mask)
        assert seeds.shape == (M, k)
        for m in range(M):
            valid = seeds[m][seeds[m] >= 0]
            assert valid.size <= k
            assert len(set(int(v) for v in valid)) == valid.size
            assert np.all(mask[valid])
            assert np.all(status[m, valid] != STATUS_ACTIVE)


def test_prime_requires_decaying_exposure():
    G = im_graph.build_graph(
        nx.gnp_random_graph(30, 0.2, seed=2, directed=True), name="prime"
    )
    state = im_env.make_state(G, M=2, budget_left=20)
    mask = np.ones(G.adj.shape[0], dtype=bool)
    fresh = ops.Prime().refine(G, im_laws.make_law("WC", G), state, 4, mask)
    assert np.all(fresh == -1)
    reinf = ops.Prime().refine(G, im_laws.make_law("REINF", G), state, 4, mask)
    assert np.any(reinf >= 0)
    assert np.all(reinf[:, 0] >= 0)


def test_wait_places_no_seeds(im_instance):
    G, status, E = im_instance
    state = SimState(status.copy(), E.copy(), 0, 100)
    seeds = ops.Wait().refine(
        G, im_laws.make_law("WC", G), state, 5, np.ones(G.adj.shape[0], bool)
    )
    assert seeds.shape[1] == 0
    assert ops.Wait().checker(state, state, np.ones(G.adj.shape[0], bool)).all()


# --------------------------------------------------------------------- RPP


def test_rpp_route_operators_respect_constraints(rpp_instance):
    env, law = rpp_instance
    state = env.reset(M=3)
    regs = regions.make_spatial_regions(env, k_top=3)
    mask = regs[2][1]
    k = 8
    for op in [ops.Harvest(), ops.Sweep()]:
        action = op.refine(env, law, state, k, mask)
        assert action.shape == (3, k)
        dist = rpp_env.total_distance(env, state, action)
        lengths = rpp_env.action_lengths(action)
        assert np.all(dist <= env.max_daily_distance + 1e-9)
        for m in range(3):
            nodes = action[m][action[m] >= 0]
            assert nodes.size <= k
            assert len(set(int(v) for v in nodes)) == nodes.size
            assert np.all(mask[nodes])
            assert np.all(state.profit[m, nodes] > 0)
        after = state.clone()
        rpp_env.step(env, after, action, law=law, rng=np.random.default_rng(1))
        assert op.checker(state, after, mask).all()
        assert lengths.sum() > 0


def test_rpp_two_opt_improves_crossing_route():
    loc = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]])
    env = rpp_env.RPPEnv(loc, np.array([0.0, 1.0, 1.0, 1.0]), budget=5, T=5)
    crossing = [1, 2, 3]
    before = ops._order_distance(env, 0, crossing, closed=False)
    improved = ops._two_opt(env, 0, crossing, closed=False)
    after = ops._order_distance(env, 0, improved, closed=False)
    assert after <= before
    assert after == pytest.approx(30.0)


def test_rpp_reposition_ends_inside_region(rpp_instance):
    env, law = rpp_instance
    state = env.reset(M=2)
    regs = regions.make_spatial_regions(env, k_top=3)
    mask = regs[1][1]
    op = ops.Reposition()
    action = op.refine(env, law, state, 4, mask)
    assert action.shape == (2, 1)
    after = state.clone()
    rpp_env.step(env, after, action, law=law, rng=np.random.default_rng(2))
    assert np.all(mask[after.current])
    assert op.checker(state, after, mask).all()


def test_rpp_wait_drifts_and_advances_day(rpp_instance):
    env, law = rpp_instance
    state = env.reset(M=2)
    mask = np.ones(env.n, dtype=bool)
    op = ops.Wait()
    action = op.refine(env, law, state, 5, mask)
    assert action.shape == (2, 0)
    before_t = state.t
    before_budget = state.budget_left.copy()
    before = state.clone()
    rpp_env.step(env, state, action, law=law, rng=np.random.default_rng(3))
    assert state.t == before_t + 1
    assert np.array_equal(state.budget_left, before_budget)
    assert op.checker(before, state, mask).all()
