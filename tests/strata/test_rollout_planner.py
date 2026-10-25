"""M3a: rollout planner acceptance tests (KT-b reproduction and RPP).

The IM tests rebuild the exact instance of ``experiments/kt_law/ktb_hier.py``
(dblp n=500, graph seed 101, the same Louvain regions) and check that one-step
lookahead with a flat continuation reproduces the published kill-test totals
within three percent.  The RPP test checks that the planner is not worse than
the flat harvest baseline on a small TREND instance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from strata.abstraction.library import OperatorLibrary
from strata.abstraction.operators import Gate, Harvest, Hedge, Hub
from strata.core.types import Hypothesis, HypothesisSet
from strata.domains.im import env as im_env
from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import points
from strata.planning.belief import Belief
from strata.planning.rollout_planner import RolloutPlanner, allocation_k

REPO = Path(__file__).resolve().parents[2]
KT_LAW = REPO / "experiments" / "kt_law"
if str(KT_LAW) not in sys.path:
    sys.path.insert(0, str(KT_LAW))

from common import load_subgraph  # noqa: E402  (verified KT-b instance loader)


def _single_law_belief(law, domain):
    hset = HypothesisSet(
        [Hypothesis(law, 0.0, 1.0, 0)], np.zeros((1, 1), dtype=np.int64)
    )
    return Belief(hset, domain)


def _ktb_regions(G, n=500):
    """``ktb_hier.regions_of``: whole graph plus the 3 largest communities."""
    comms = nx.community.louvain_communities(G.to_undirected(), seed=0)
    comms = sorted(comms, key=len, reverse=True)[:3]
    regions = [("all", np.ones(n, dtype=bool))]
    for i, community in enumerate(comms):
        mask = np.zeros(n, dtype=bool)
        mask[list(community)] = True
        regions.append(("c%d" % i, mask))
    return regions


def _flush_best_flat(graph, law, episodes=(1000, 1001), budget=30, T=10):
    """Best of the three fixed hub schedules (the KT-b ``flat best`` column)."""
    hub = Hub()
    full = np.ones(graph.adj.shape[0], dtype=bool)
    scores = {}
    for alloc in ("even", "normal", "static"):
        values = []
        for ep_seed in episodes:
            state = im_env.make_state(graph, M=1, budget_left=budget)
            rng = np.random.default_rng(ep_seed)
            total = 0.0
            for t in range(T):
                k = allocation_k(alloc, t, T, int(state.budget_left))
                seeds = hub.refine(graph, law, state, k, full)
                _, reward, _ = im_env.step(graph, law, state, seeds, rng)
                total += float(reward[0])
            values.append(total)
        scores[alloc] = float(np.mean(values))
    return max(scores.values()), scores


@pytest.mark.slow
@pytest.mark.parametrize(
    "law_name, expected, flat_expected",
    [("CC3", 92.0, 60.0), ("CC2", 181.0, 169.0)],
)
def test_rollout_reproduces_ktb_with_true_law(law_name, expected, flat_expected):
    G = load_subgraph("dblp", 500, seed=101)
    graph = im_graph.build_graph(G, name="ktb-dblp500-seed101")
    regions = _ktb_regions(G, 500)
    law = im_laws.make_law(law_name, graph)
    belief = _single_law_belief(law, "im")

    library = OperatorLibrary([Hub(), Gate(2), Gate(3), Hedge(2)])
    planner = RolloutPlanner(
        library, base=None, M=48, mode="mean", seed=1000, T=10, regions=regions
    )
    values = []
    for ep_seed in (1000, 1001):
        planner.seed = ep_seed
        values.append(
            planner.run_episode(
                graph, belief, true_law=law, T=10, budget=30, seed=ep_seed
            )
        )
    mean_value = float(np.mean(values))
    assert mean_value == pytest.approx(expected, rel=0.03)

    flat_best, _ = _flush_best_flat(graph, law)
    assert flat_best == pytest.approx(flat_expected, rel=0.03)
    assert mean_value > flat_best


def test_rollout_not_worse_than_flat_harvest_on_rpp_trend():
    loc, initial, factor = points.random_instance(40, seed=3)
    env = rpp_env.RPPEnv(
        loc=loc,
        initial_profit=initial,
        profit_factor=factor,
        budget=12,
        T=6,
        empty_day_advances=True,
    )
    law = rpp_laws.get_law("TREND", factor=factor, seed=0)
    belief = _single_law_belief(law, "rpp")

    library = OperatorLibrary.standard("rpp")
    planner = RolloutPlanner(library, base="even", M=32, mode="mean", seed=0)
    planner_values = [
        planner.run_episode(env, belief, true_law=law, seed=ep_seed)
        for ep_seed in range(4)
    ]

    harvest = Harvest()
    full = np.ones(env.n, dtype=bool)
    flat_values = []
    for ep_seed in range(4):
        state = env.reset(M=1)
        rng = np.random.default_rng(ep_seed)
        total = 0.0
        while int(state.t) <= env.T:
            k = allocation_k("even", int(state.t) - 1, env.T, int(state.budget_left[0]))
            action = harvest.refine(env, law, state, k, full)
            _, reward, done, _ = rpp_env.step(
                env, state, action, law=law, rng=rng, return_info=True
            )
            total += float(reward[0])
            if done:
                break
        flat_values.append(total)

    planner_mean = float(np.mean(planner_values))
    flat_mean = float(np.mean(flat_values))
    flat_se = float(np.std(flat_values, ddof=1) / np.sqrt(len(flat_values)))
    assert planner_mean >= flat_mean - flat_se
