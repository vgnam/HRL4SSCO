"""Diagnosis and decision-gap refinement tests."""

from __future__ import annotations

import numpy as np

from strata.abstraction.refinement import decision_gap, propose_split
from strata.agent import StrataAgent
from strata.diagnosis.calibration import im_calibration, rpp_calibration
from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.domains.im import logs as im_logs
from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import logs as rpp_logs
from strata.domains.rpp import points
from strata.eval import protocol


def test_im_calibration_flags_the_wrong_law_only():
    sampler = im_graph.dataset_sampler("dblp", 150, method="rw", seed=101)
    graph = im_graph.build_graph(sampler(), name="dblp150")
    eps = im_logs.collect_logs(graph, im_laws.make_law("CC2", graph), T=6, B=12, n_episodes=30, seed=2)
    assert im_calibration(im_laws.make_law("CC2", graph), eps, graph)["p_value"] > 0.05
    assert im_calibration(im_laws.make_law("WC", graph), eps, graph)["p_value"] < 0.01


def test_rpp_calibration_flags_a_missing_trend():
    loc, initial, factor = points.random_instance(40, seed=2)
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, budget=12, T=6, profit_factor=factor,
                         empty_day_advances=True)
    trend = rpp_laws.make_law("TREND", trend_strength=0.05, seed=0)
    eps = rpp_logs.collect_logs(env, trend, T=6, budget=12, n_episodes=40, seed=2)
    from strata.laws import enumerate as en
    from strata.laws import fit

    iid = fit.fit_drift_program(en.drift_structure_by_name("uniform", 1), eps, env)
    assert rpp_calibration(iid, eps, env)["p_value"] < 0.01


def test_agent_records_bands_and_attributes():
    inst = protocol.make_instance("im", dataset="dblp", n=150, graph_seed=101, law="CC2", T=4, B=8)
    logs = protocol.make_logs(inst, n_episodes=30, seed=1)
    agent = StrataAgent(logs, law_source="dsl", operator_source="hand", n_sims=20, diagnose=True)
    protocol.evaluate(agent, inst, episodes=1, seed=0)
    assert all("band" in r for r in agent.records)
    out = agent.diagnose(logs)
    assert set(out["summary"]) == {"law", "abstraction", "ok"}
    assert out["summary"]["law"] == 0  # the MAP law is the true gate law


def test_decision_gap_and_split():
    # two kinds of states: operator 0 best when x < 5, operator 1 best otherwise
    x = np.arange(10, dtype=float)
    values = np.stack([np.where(x < 5, 10.0, 1.0), np.where(x < 5, 1.0, 10.0)], axis=1)
    gap = decision_gap(values)
    assert gap["split"] and gap["n_best_ops"] == 2
    prop = propose_split({"x": x, "noise": np.zeros(10)}, values)
    assert prop["feature"] == "x" and 4.0 < prop["cut"] < 5.0
    same = np.tile([10.0, 1.0], (6, 1)) + np.random.default_rng(0).normal(0, 0.01, (6, 2))
    assert not decision_gap(same)["split"]
