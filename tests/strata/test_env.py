"""M0 acceptance tests for the two environments and their hidden laws.

The IM checks are strict: for every ported law and for two graphs,
``domains.im.env.step`` must be bit-for-bit identical to
``experiments/kt_law/sim.transition`` under the same RNG seed.  The RPP check
replays ``RPP/Env.py`` with the same global seed and compares the whole
profit/reward trajectory.
"""

import importlib.util
import os
import sys
from functools import lru_cache

import networkx as nx
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KT_DIR = os.path.join(REPO, "experiments", "kt_law")
if KT_DIR not in sys.path:
    sys.path.insert(0, KT_DIR)

import laws as kt_laws  # noqa: E402
import sim as kt_sim  # noqa: E402

from strata.core.types import (
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    STATUS_REMOVED,
    Features,
    Graph,
)  # noqa: E402
from strata.domains.im import env as im_env  # noqa: E402
from strata.domains.im import graph as im_graph  # noqa: E402
from strata.domains.im import hidden_laws as im_laws  # noqa: E402
from strata.domains.rpp import env as rpp_env  # noqa: E402
from strata.domains.rpp import hidden_laws as rpp_laws  # noqa: E402
from strata.domains.rpp import points  # noqa: E402


# --------------------------------------------------------------------- fixtures


@lru_cache(maxsize=None)
def _graph(kind):
    if kind == "synthetic":
        G = nx.gnp_random_graph(120, 10.0 / 120.0, seed=3, directed=True)
        return im_graph.build_graph(G, name="synthetic")
    if kind == "dblp":
        return im_graph.sample_instance("dblp", 100, seed=101)
    raise KeyError(kind)


def _load_rpp_reference():
    path = os.path.join(REPO, "RPP", "Env.py")
    spec = importlib.util.spec_from_file_location("_strata_rpp_ref_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Env


RPPRefEnv = _load_rpp_reference()


# ------------------------------------------------------------------ IM: simulator


@pytest.mark.parametrize("law_name", im_laws.PORTED_LAWS)
@pytest.mark.parametrize("graph_kind", ["synthetic", "dblp"])
def test_im_step_bit_for_bit_with_kt_law(law_name, graph_kind):
    G = _graph(graph_kind)
    n = G.adj.shape[0]
    M = 3
    kt_law = kt_laws.get_law(law_name)
    our_law = im_laws.make_law(law_name, G)
    op = kt_law.op(G.adj)

    st_ref = np.zeros((M, n), dtype=np.int8)
    E_ref = np.zeros((M, n), dtype=np.float32)
    state = im_env.make_state(G, M=M)
    rng_ref = np.random.default_rng(2024)
    rng_our = np.random.default_rng(2024)
    rng_seeds = np.random.default_rng(77)

    for t in range(5):
        if t == 3:
            # Exercise per-path seed sets with -1 padding as well.
            seeds = np.stack([rng_seeds.integers(0, n, size=4) for _ in range(M)])
            seeds[:, -2:] = -1
        else:
            seeds = rng_seeds.integers(0, n, size=3 if t % 2 == 0 else 5)

        st_ref, E_ref, rew_ref, info_ref = kt_sim.transition(
            kt_law, op, G.indeg, st_ref, E_ref, seeds, rng_ref, return_info=True
        )
        _, rew_our, info_our = im_env.step(
            G, our_law, state, seeds, rng_our, op=op, return_info=True
        )

        assert st_ref.tobytes() == state.status.tobytes()
        assert E_ref.dtype == state.E.dtype
        assert E_ref.tobytes() == state.E.tobytes()
        np.testing.assert_array_equal(rew_ref, rew_our)
        for key in ("a", "eligible", "new", "p", "seeded"):
            np.testing.assert_array_equal(info_ref[key], info_our[key])


def test_im_rollout_matches_kt_law():
    G = _graph("synthetic")
    law_name = "CC3"
    kt_law = kt_laws.get_law(law_name)
    our_law = im_laws.make_law(law_name, G)
    seed_sets = [
        np.array([1, 2, 3]),
        np.array([4]),
        np.zeros(0, np.int64),
        np.array([5, 6]),
    ]
    R_ref = kt_sim.rollout(
        kt_law, G.adj, G.indeg, seed_sets, M=7, seed=31, op=kt_law.op(G.adj)
    )
    R_our = im_env.rollout(G, our_law, seed_sets, M=7, seed=31)
    np.testing.assert_array_equal(R_ref, R_our)


def test_im_l0_budget_and_reseeding():
    G = _graph("synthetic")
    state = im_env.make_state(G, M=1)
    state.budget_left = 5
    ask = np.array([1, 2, 3])
    assert im_env.budget_ok(state, ask)
    assert not im_env.budget_ok(2, ask)
    state.status[0, 1] = STATUS_REMOVED
    _, reward, info = im_env.step(
        G,
        im_laws.make_law("WC", G),
        state,
        ask,
        np.random.default_rng(0),
        return_info=True,
    )
    assert state.budget_left == 2
    assert info["seeded"][0, 1]
    # The re-seeded node acts this round and is then retired, exactly like
    # kt_law/sim.transition (removed nodes are not eligible to re-activate).
    assert state.status[0, 1] == STATUS_REMOVED
    # A removed node is not inactive, so re-seeding it earns no seed reward.
    assert reward[0] >= 1  # other inactive seeds may activate
    assert (
        im_env.reward_count(
            np.array([[STATUS_INACTIVE, STATUS_REMOVED]]),
            np.array([[STATUS_ACTIVE, STATUS_ACTIVE]]),
        )[0]
        == 1
    )
    assert im_env.seed_count(ask) == 3


# --------------------------------------------------------------- IM: hidden laws


@pytest.mark.parametrize("law_name", im_laws.LAW_NAMES)
def test_im_hidden_law_probabilities(law_name):
    G = _graph("synthetic")
    n = G.adj.shape[0]
    law = im_laws.make_law(law_name, G, seed=5)
    M = 2
    a = np.zeros((M, n), dtype=np.float32)
    a[:, :5] = 1.0
    a[:, 5:9] = 2.0
    E = a * 2.0
    feats = Features(
        a=a,
        E=E,
        a2=None,
        eligible=np.ones((M, n), bool),
        indeg=G.indeg,
        community=G.community,
    )
    p = law.prob(feats)
    assert p.shape == (M, n)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert law.n_params() >= 0
    assert isinstance(law.rho, float)
    assert isinstance(law.two_hop, bool)


def test_im_mix_comm_het_masks():
    G = _graph("synthetic")
    n = G.adj.shape[0]
    M = 1
    a = np.full((M, n), 2.0, dtype=np.float32)
    E = a.copy()
    feats = Features(
        a=a,
        E=E,
        a2=None,
        eligible=np.ones((M, n), bool),
        indeg=G.indeg,
        community=G.community,
    )

    mix = im_laws.Mix(n=n, gate_frac=0.3, seed=1)
    assert 0.15 < mix.gate_mask.mean() < 0.45
    p = mix.prob(feats)
    np.testing.assert_array_equal(p[0, mix.gate_mask], 1.0)

    comm_ids = np.r_[np.zeros(n // 2, np.int64), np.ones(n - n // 2, np.int64)]
    G2 = Graph(
        adj=G.adj,
        adjT=G.adjT,
        indeg=G.indeg,
        outdeg=G.outdeg,
        community=comm_ids,
        name="two-communities",
    )
    comm = im_laws.make_law("COMM", G2)
    pc = comm.prob(
        Features(
            a=a,
            E=E,
            a2=None,
            eligible=np.ones((M, n), bool),
            indeg=G.indeg,
            community=comm_ids,
        )
    )
    cc2_side = comm_ids == 1
    wc_side = comm_ids == 0
    np.testing.assert_array_equal(pc[0, cc2_side], 1.0)
    informative = wc_side & (G.indeg >= 2)
    assert np.all(pc[0, informative] < 1.0)

    het = im_laws.Heterogeneous(n=n, seed=2)
    counts = np.bincount(het.node_type, minlength=3)
    assert abs(counts[0] / n - 0.5) <= 0.05
    assert abs(counts[1] / n - 0.3) <= 0.05
    assert abs(counts[2] / n - 0.2) <= 0.05
    ph = het.prob(feats)
    np.testing.assert_array_equal(ph[0, het.node_type == 1], 1.0)


# ------------------------------------------------------------------ RPP: env


def _rpp_fixture():
    loc = np.array(
        [[0, 0], [10, 0], [20, 0], [0, 20], [20, 20], [5, 15], [15, 10]],
        dtype=np.float64,
    )
    initial = np.array([1.0, 0.8, 0.6, 0.9, 0.7, 0.75, 0.85])
    params = dict(
        budget=8,
        T=5,
        max_daily_distance=1.5,
        distance_penalty=5.0,
        start_node=0,
        profit_factor=0.05,
    )
    actions = [
        np.array([1, 2]),
        np.array([4]),
        np.array([3]),
        np.array([2, 4]),
        np.array([1]),
    ]
    return loc, initial, params, actions


def test_rpp_env_matches_reference_same_seed():
    loc, initial, params, actions = _rpp_fixture()
    np.random.seed(123)
    ref = RPPRefEnv(loc=loc, initial_profit=initial, **params)
    expected = []
    for action in actions:
        _, reward, done = ref.step(list(action))
        expected.append(
            (
                float(reward),
                ref.profit.copy(),
                int(ref.remaining_budget),
                int(ref.current_node),
                int(ref.t),
                bool(done),
            )
        )

    np.random.seed(123)
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, **params)
    state = env.reset()
    law = rpp_laws.IID()
    for i, action in enumerate(actions):
        state, reward, done, info = rpp_env.step(
            env, state, action, law=law, rng=np.random, return_info=True
        )
        r_ref, profit_ref, budget_ref, current_ref, t_ref, done_ref = expected[i]
        assert float(reward[0]) == r_ref
        np.testing.assert_array_equal(state.profit[0], profit_ref)
        assert int(state.budget_left[0]) == budget_ref
        assert int(state.current[0]) == current_ref
        assert state.t == t_ref
        assert bool(done) == done_ref


def test_rpp_l0_cost_and_penalty():
    loc, initial, params, _ = _rpp_fixture()
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, **params)
    state = env.reset()
    # The submodular cost is zero for k=0/1 and eta*(k-1)^2 otherwise.
    A = np.array([[1, 2, 3]])
    base = rpp_env.profit_reward(env, state.profit, A, np.array([3]))
    assert base[0] == pytest.approx(
        initial[1] + initial[2] + initial[3] - env.eta * 4.0
    )
    dist = rpp_env.total_distance(env, state, A)
    assert dist[0] > 0.0
    penalised = rpp_env.distance_penalty_reward(
        env, np.zeros(1), np.array([env.max_daily_distance + 0.25])
    )
    assert penalised[0] == pytest.approx(-params["distance_penalty"] * 0.25)


def test_rpp_empty_action_keeps_day():
    loc, initial, params, _ = _rpp_fixture()
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, **params)
    state = env.reset()
    _, reward, done, _ = rpp_env.step(
        env,
        state,
        np.zeros(0, np.int64),
        law=rpp_laws.IID(),
        rng=np.random.default_rng(0),
        return_info=True,
    )
    assert reward[0] == 0.0
    assert state.t == 1
    assert not done


def test_rpp_real_data_loader_and_regions():
    loc, initial, factor = points.instance_sampler(
        "foursquare-NYC", n=20, method="uniform", drift="fitted", seed=0
    )()
    assert loc.shape == (20, 2)
    assert np.all(loc >= 0.0) and np.all(loc <= 100.0)
    assert np.all((initial >= 0.5) & (initial <= 1.0))
    assert factor > 0.0
    labels = points.spatial_clusters(loc, 3, seed=0)
    assert labels.shape == (20,)
    assert 1 <= len(np.unique(labels)) <= 3
    deltas = points.period_profit_deltas("foursquare-NYC")
    assert deltas.shape[0] == len(points.load("foursquare-NYC")["count"])
    assert np.all(np.isfinite(deltas))


@pytest.mark.parametrize("law_name", rpp_laws.LAW_NAMES)
def test_rpp_hidden_laws_are_well_behaved(law_name):
    loc, initial, _ = points.random_instance(25, seed=2)
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, budget=10, T=6)
    state = env.reset(M=3)
    visited_now = np.zeros_like(state.visited)
    visited_now[:, 1] = True

    law = rpp_laws.get_law(law_name, seed=0)
    d1 = law.drift(env, state, visited_now, np.random.default_rng(5))
    d2 = law.drift(env, state, visited_now, np.random.default_rng(5))
    assert d1.shape == (3, env.n)
    assert np.all(np.isfinite(d1))
    np.testing.assert_array_equal(d1, d2)
    # Visited POIs never receive drift.
    assert np.all(d1[state.visited] == 0.0)
    assert np.all(d1[visited_now] == 0.0)


def test_rpp_empty_day_advances_when_enabled():
    loc, initial, params, _ = _rpp_fixture()
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, empty_day_advances=True, **params)
    state = env.reset()
    before = state.profit.copy()
    _, reward, done, info = rpp_env.step(
        env,
        state,
        np.zeros(0, np.int64),
        law=rpp_laws.IID(),
        rng=np.random.default_rng(0),
        return_info=True,
    )
    assert reward[0] == 0.0
    assert state.t == 2
    assert not done
    # Profits of unvisited POIs drifted; the start node (visited) did not.
    unvisited = before[0] > 0
    assert np.any(state.profit[0][unvisited] != before[0][unvisited])
    assert state.profit[0][env.start_node] == 0.0


def test_rpp_fitted_uses_each_venues_own_history():
    loc, initial, factor, venues = points.make_instance_with_venues(
        "foursquare-NYC", n=20, method="uniform", drift="fitted", seed=0
    )
    # Same venues as the plain sampler with the same seed.
    loc2, initial2, _ = points.instance_sampler(
        "foursquare-NYC", n=20, method="uniform", drift="fitted", seed=0
    )()
    np.testing.assert_array_equal(loc, loc2)
    np.testing.assert_array_equal(initial, initial2)

    law = rpp_laws.FITTED(dataset="foursquare-NYC", venues=venues)
    assert law.deltas is not None and law.deltas.shape[0] == 20
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, profit_factor=factor)
    state = env.reset(M=200)
    d = law.drift(env, state, np.zeros_like(state.visited), np.random.default_rng(1))
    # Every drawn change of POI i is one of POI i's observed changes.
    for i in range(1, 20):
        assert set(np.round(d[:, i], 12)) <= set(np.round(law.deltas[i], 12))
