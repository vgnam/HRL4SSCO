"""L1 by LLM: synthesis loop with a fake chat (offline) and an optional real-LLM test."""

from __future__ import annotations

import os

import numpy as np
import pytest

from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.domains.im import logs as im_logs
from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import logs as rpp_logs
from strata.domains.rpp import points
from strata.laws import hypotheses as hyp
from strata.laws import llm_propose as lp

FRAC_OK = '''```python
class Law:
    """Fractional threshold: active neighbours must reach a share of the in-degree."""
    params = {"theta": 0.15}
    bounds = {"theta": (0.05, 0.9)}
    rho = 0.0

    def prob(self, f):
        need = np.maximum(np.ceil(self.params["theta"] * f.indeg), 1.0)
        return (f.a >= need[None, :]).astype(float)
```'''
BROKEN = '''```python
class Law:
    params = {"w": 0.1}
    bounds = {"w": (0.0, 1.0)}
    def prob(self, f)
        return f.a
```'''
WRONG_SHAPE = '''```python
class Law:
    params = {"w": 0.1}
    bounds = {"w": (0.0, 1.0)}
    def prob(self, f):
        return np.zeros(3)
```'''
CROWD_OK = '''```python
class Drift:
    """Profits fall with the number of already visited POIs nearby."""
    family = "uniform"
    params = {"width": 0.05, "c": 0.005}
    bounds = {"width": (0.001, 0.3), "c": (0.0, 0.1)}

    def mean(self, f):
        near = (f.kernel >= np.exp(-1.0)).astype(float)
        return -self.params["c"] * (f.visited.astype(float) @ near)

    def scale(self, f):
        return np.full_like(f.profit, self.params["width"])
```'''


class FakeChat:
    """Returns canned answers in order; records the prompts."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, messages, sample=0):
        self.prompts.append(messages[-1]["content"])
        return self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]

    def usage(self):
        return {"calls": len(self.prompts)}


def _im_frac(n=200, episodes=120):
    sampler = im_graph.dataset_sampler("dblp", n, method="rw", seed=101)
    graph = im_graph.build_graph(sampler(), name="dblp%d" % n)
    law = im_laws.make_law("FRAC", graph)
    eps = im_logs.collect_logs(graph, law, T=8, B=16, n_episodes=episodes, seed=5)
    k = int(0.3 * len(eps))
    return graph, eps[k:], eps[:k]


def _rpp_crowd(n=40, episodes=60):
    loc, initial, factor = points.random_instance(n, seed=3)
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, budget=12, T=6,
                         profit_factor=factor, empty_day_advances=True)
    law = rpp_laws.make_law("CROWD", c=0.02)
    eps = rpp_logs.collect_logs(env, law, T=6, budget=12, n_episodes=episodes, seed=5)
    k = int(0.3 * len(eps))
    return env, eps[k:], eps[:k]


def test_statistics_and_templates_render():
    graph, train, _ = _im_frac(n=120, episodes=20)
    text = lp.statistics_tables("im", train, graph)
    assert "Activation rate by active in-neighbours" in text and "| a |" in text
    env, rtrain, _ = _rpp_crowd(n=30, episodes=10)
    text = lp.statistics_tables("rpp", rtrain, env)
    assert "Profit change by region" in text and "past visits nearby" in text


@pytest.mark.slow
def test_im_synthesis_rejects_bad_programs_and_beats_dsl_on_frac():
    graph, train, val = _im_frac()
    chat = FakeChat(["\n".join([FRAC_OK, BROKEN, WRONG_SHAPE]), FRAC_OK, FRAC_OK])
    cands, report = lp.propose("im", train, val, graph, chat=chat, source="llm+dsl",
                               level=2, rounds=1, k=3, budget_calls=4)
    assert report["llm"]["rejected"] >= 2
    assert any("SyntaxError" in e or "failed to load" in e for e in report["llm"]["errors"])
    llm = [c for c in cands if getattr(c, "source", "")]
    assert llm, "the correct FRAC program must be accepted"
    # the prompt carried the statistics and, after errors, the error messages
    assert "Activation rate" in chat.prompts[0]
    assert any("failed with these errors" in p for p in chat.prompts[1:])
    hset = hyp.build_hypothesis_set(cands, val, eps=5.0, cap=8, domain="im",
                                    indeg=graph.indeg, graph=graph)
    top = hset.hyps[0].law
    assert getattr(top, "source", "") and top.name.startswith("llm:")
    dsl_best = max(h.loglik for h in hset.hyps if not getattr(h.law, "source", ""))  \
        if any(not getattr(h.law, "source", "") for h in hset.hyps) else -np.inf
    assert hset.hyps[0].loglik > dsl_best


def test_rpp_synthesis_finds_crowding():
    env, train, val = _rpp_crowd()
    chat = FakeChat([CROWD_OK])
    cands, report = lp.propose("rpp", train, val, env, chat=chat, source="llm+dsl",
                               level=3, rounds=1, k=1, budget_calls=2)
    llm = [c for c in cands if getattr(c, "source", "")]
    assert llm
    assert llm[0].effect.params["c"] > 0.005  # fitted towards the true 0.02
    from strata.laws import fit
    ll_llm = fit.loglik_drift(llm[0], val, env)
    ll_dsl = max(fit.loglik_drift(c, val, env) for c in cands if not getattr(c, "source", ""))
    assert ll_llm > ll_dsl


def test_no_chat_falls_back_to_dsl():
    graph, train, val = _im_frac(n=120, episodes=20)
    cands, report = lp.propose("im", train, val, graph, chat=None, source="llm", level=1)
    assert report["fallback"] and cands


@pytest.mark.skipif(os.environ.get("STRATA_LLM") != "1", reason="set STRATA_LLM=1 to call the real LLM")
def test_real_llm_beats_dsl_on_frac():
    from strata.llm import Chat

    graph, train, val = _im_frac()
    cands, report = lp.propose("im", train, val, graph, chat=Chat(), source="llm+dsl",
                               level=3, rounds=2, k=3, budget_calls=6)
    hset = hyp.build_hypothesis_set(cands, val, eps=5.0, cap=8, domain="im",
                                    indeg=graph.indeg, graph=graph)
    assert getattr(hset.hyps[0].law, "source", ""), report
