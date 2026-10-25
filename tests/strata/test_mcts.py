"""M3b: MCTS planner tests (root sampling, decision rule, value vs rollout)."""

from __future__ import annotations

import numpy as np
import pytest

from strata.abstraction.library import OperatorLibrary
from strata.core.types import Hypothesis, HypothesisSet
from strata.domains.im import env as im_env
from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import points
from strata.planning.belief import Belief
from strata.planning.mcts import MCTSPlanner
from strata.planning.objectives import aggregate
from strata.planning.rollout_planner import RolloutPlanner


def _belief(laws, domain, prior=None):
    prior = np.ones(len(laws)) if prior is None else np.asarray(prior, dtype=float)
    hset = HypothesisSet(
        [Hypothesis(law, 0.0, float(p), i) for i, (law, p) in enumerate(zip(laws, prior))],
        np.zeros((len(laws), len(laws)), dtype=np.int64),
    )
    return Belief(hset, domain)


def _dblp(n=300, seed=101):
    sampler = im_graph.dataset_sampler("dblp", n, method="rw", seed=seed)
    return im_graph.build_graph(sampler(), name="dblp%d" % n)


def _play(planner, graph, law, belief, T, B, seed):
    rng = np.random.default_rng(seed)
    state = im_env.make_state(graph, M=1, budget_left=B)
    total = 0.0
    for t in range(T):
        macro = planner.choose(graph, state, belief, t)
        mask = planner.regions[macro.region][1]
        k = min(int(macro.k), int(state.budget_left))
        action = planner._operator(macro.op).refine(graph, law, state, k, mask)
        _, r, _ = im_env.step(graph, law, state, action, rng)
        total += float(r[0])
    return total


def test_root_sampling_fills_every_hypothesis_and_maxmin_is_respected():
    graph = _dblp(200)
    laws = [im_laws.make_law("WC", graph), im_laws.make_law("CC2", graph)]
    belief = _belief(laws, "im", [0.5, 0.5])
    planner = MCTSPlanner(OperatorLibrary.standard("im"), base="even", n_sims=80,
                          mode="maxmin", seed=3, T=5)
    state = im_env.make_state(graph, M=1, budget_left=10)
    macro = planner.choose(graph, state, belief, 0)
    info = planner.last_info
    assert macro in info["macros"]
    V = info["per_law_values"]
    assert V.shape[1] == 2 and np.all(np.isfinite(V))
    # the chosen macro has the best worst-hypothesis value among root children
    worst = V.min(axis=1)
    assert worst[info["macros"].index(macro)] == pytest.approx(worst.max())
    np.testing.assert_allclose(info["scores"], aggregate(V, belief.weights, "maxmin"))


def test_mcts_not_worse_than_rollout_under_cc3():
    graph = _dblp(300)
    law = im_laws.make_law("CC3", graph)
    belief = _belief([law], "im")
    lib = OperatorLibrary.standard("im")
    T, B = 6, 18
    roll = RolloutPlanner(lib, base="even", M=8, seed=1, T=T)
    mcts = MCTSPlanner(lib, base="even", n_sims=120, seed=1, T=T)
    v_roll = _play(roll, graph, law, belief, T, B, seed=7)
    v_mcts = _play(mcts, graph, law, belief, T, B, seed=7)
    assert v_mcts >= 0.95 * v_roll


def test_mcts_runs_on_rpp_and_matches_flat_harvest():
    loc, initial, factor = points.random_instance(40, seed=4)
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, budget=12, T=6,
                         profit_factor=factor, empty_day_advances=True)
    law = rpp_laws.make_law("TREND", seed=0)
    belief = _belief([law], "rpp")
    lib = OperatorLibrary.standard("rpp")
    mcts = MCTSPlanner(lib, n_sims=40, seed=2)
    totals, flat = [], []
    for ep in range(2):
        for planner, out in ((mcts, totals), (None, flat)):
            rng = np.random.default_rng(100 + ep)
            state = env.reset(M=1)
            total = 0.0
            while int(state.t) <= env.T:
                if planner is None:
                    rem = int(state.budget_left[0])
                    k = rem if int(state.t) == env.T else int(np.ceil(rem / (env.T - int(state.t) + 1)))
                    action = lib.get("harvest").refine(env, law, state, k, np.ones(env.n, bool))
                else:
                    macro = planner.choose(env, state, belief, int(state.t))
                    mask = planner.regions[macro.region][1]
                    k = min(int(macro.k), int(state.budget_left[0]))
                    action = planner._operator(macro.op).refine(env, law, state, k, mask)
                _, r, done, _ = rpp_env.step(env, state, action, law=law, rng=rng, return_info=True)
                total += float(r[0])
                if done:
                    break
            out.append(total)
    assert np.mean(totals) >= np.mean(flat) - 0.25 * abs(np.mean(flat)) - 1e-9


def test_root_tree_values_match_flat_continuation_and_do_not_front_load():
    # The full open-loop tree undervalued macros that keep budget for later and,
    # with enough simulations, spent the whole budget in round 0.  The root tree
    # values "macro, then base" without that bias.
    graph = _dblp(200)
    law = im_laws.make_law("WC", graph)
    belief = _belief([law], "im")
    lib = OperatorLibrary.standard("im")
    T, B = 6, 12
    planner = MCTSPlanner(lib, base="even", n_sims=400, seed=0, T=T, tree="root")
    state = im_env.make_state(graph, M=1, budget_left=B)
    macro = planner.choose(graph, state, belief, 0)
    assert int(macro.k) < B
    info = planner.last_info
    scores = dict(zip(info["macros"], np.asarray(info["scores"]).ravel()))
    dump = [s for m, s in scores.items() if m.op == "hub" and m.region == 0 and int(m.k) == B]
    if dump:
        assert scores[macro] >= dump[0]
