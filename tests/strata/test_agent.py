"""End-to-end tests of the STRATA agent on IM and RPP (fake chat, offline)."""

from __future__ import annotations

import numpy as np
import pytest

from strata.agent import StrataAgent, make_strata_methods
from strata.baselines import model_based
from strata.eval import protocol

from test_llm_laws import CROWD_OK, FRAC_OK, FakeChat  # noqa: E402 (same test directory)


def _im_instance(law="CC2", n=200, T=6, B=12):
    return protocol.make_instance("im", dataset="dblp", n=n, graph_seed=101, law=law, T=T, B=B)


def _rpp_instance(law="TREND", n=40, T=6, B=12):
    return protocol.make_instance("rpp", dataset=None, n=n, graph_seed=4, law=law, T=T, B=B)


def _regret(policy, instance, oracle_values, episodes=2):
    values = protocol.evaluate(policy, instance, episodes=episodes, seed=3)
    return protocol.regret(values, oracle_values).mean, values


@pytest.mark.slow
def test_im_end_to_end_cc2_matches_libB():
    inst = _im_instance("CC2")
    logs = protocol.make_logs(inst, n_episodes=80, seed=1)
    oracle = protocol.evaluate(model_based.Oracle(logs), inst, episodes=2, seed=3)
    r_lib, _ = _regret(model_based.LibB(logs), inst, oracle)
    agent = StrataAgent(logs, law_source="dsl", operator_source="hand", n_sims=60)
    r_agent, _ = _regret(agent, inst, oracle)
    assert r_agent <= r_lib + 0.05
    assert agent.report["hypotheses"][0][0].startswith("gate")


def test_im_llm_path_on_frac_uses_llm_hypothesis():
    inst = _im_instance("FRAC")
    logs = protocol.make_logs(inst, n_episodes=80, seed=1)
    chat = FakeChat([FRAC_OK])
    agent = StrataAgent(logs, chat=chat, n_sims=40, llm_rounds=0, llm_budget=1,
                        operator_source="hand")
    values = protocol.evaluate(agent, inst, episodes=1, seed=3)
    assert np.all(np.isfinite(values))
    assert agent.report["hypotheses"][0][0].startswith("llm:")
    # the belief stays on the LLM law while the true law plays out
    assert agent.records[-1]["weights_after"][0] > 0.5


def test_rpp_end_to_end_crowd_with_llm_law():
    inst = _rpp_instance("CROWD")
    logs = protocol.make_logs(inst, n_episodes=40, seed=1)
    chat = FakeChat([CROWD_OK])
    agent = StrataAgent(logs, chat=chat, n_sims=30, llm_rounds=0, llm_budget=1,
                        operator_source="hand")
    values = protocol.evaluate(agent, inst, episodes=2, seed=3)
    flat = protocol.evaluate(model_based.WCDefault(logs), inst, episodes=2, seed=3)
    assert np.all(np.isfinite(values))
    assert np.mean(values) >= np.mean(flat) - 0.1 * abs(np.mean(flat))


@pytest.mark.parametrize("name", sorted(make_strata_methods()))
def test_every_ablation_runs(name):
    inst = _im_instance("CC2", n=120, T=4, B=8)
    logs = protocol.make_logs(inst, n_episodes=30, seed=1)
    overrides = dict(make_strata_methods()[name])
    overrides.setdefault("law_source", "dsl")
    overrides.setdefault("operator_source", "hand")
    agent = StrataAgent(logs, n_sims=20, M=4, **overrides)
    values = protocol.evaluate(agent, inst, episodes=1, seed=0)
    assert np.isfinite(values).all()
