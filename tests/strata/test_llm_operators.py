"""L2 by LLM: operator synthesis with a fake chat (offline) and an optional real-LLM test."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from strata.abstraction import llm_operators as lo
from strata.abstraction.library import OperatorLibrary
from strata.abstraction.operators import Gate, Harvest, Hub
from strata.domains.im import env as im_env
from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import points

GATE2_EQUIV = '''```python
NAME = "gate2-copy"

def objective(f, state, region_mask, chosen, graph):
    el = f.eligible
    W = (el & (f.a == 1)) + 0.3 * (el & (f.a < 1))
    return np.asarray(W.astype(np.float32) @ graph.adjT) + 1e-3 * el
```'''
SHARED3 = '''```python
NAME = "shared-3"

def objective(f, state, region_mask, chosen, graph):
    # push inactive nodes towards three active in-neighbours, completion first
    el = f.eligible
    w = np.where(el & (f.a == 2), 1.0, np.where(el & (f.a == 1), 0.4, np.where(el, 0.1, 0.0)))
    return np.asarray(w @ graph.adjT)
```'''
HUB_EQUIV = '''```python
NAME = "hub-copy"

def objective(f, state, region_mask, chosen, graph, prob):
    P0, P1 = prob(f.a), prob(f.a + 1)
    D = np.where(f.eligible, P1 - P0, 0.0)
    return np.asarray(D @ graph.adjT) + f.eligible * (1.0 - P0)
```'''
BROKEN = '''```python
NAME = "broken"
def objective(f, state, region_mask, chosen, graph)
    return 0
```'''
HARVEST_EQUIV = '''```python
NAME = "harvest-copy"

def score(env, state, region_mask):
    return state.profit.copy()
```'''


class FakeChat:
    def __init__(self, answers):
        self.answers, self.n = list(answers), 0

    def __call__(self, messages, sample=0):
        self.n += 1
        return self.answers[min(self.n - 1, len(self.answers) - 1)]


def _dblp(n=300):
    sampler = im_graph.dataset_sampler("dblp", n, method="rw", seed=101)
    return im_graph.build_graph(sampler(), name="dblp%d" % n)


def _with_adj(code, graph, name):
    """Canned objectives need the adjacency transpose; inject it as a constant."""
    import scipy.sparse as sp  # noqa: F401

    return code.replace(name, "_ADJ_T") .replace("def objective", "_ADJ_T = None\ndef objective")


def test_llm_objective_reproduces_gate2_refine():
    graph = _dblp(200)
    law = im_laws.make_law("CC2", graph)
    op = lo.operator_from_answer("im", lo.extract_code_blocks(GATE2_EQUIV)[0])
    state = im_env.make_state(graph, M=3, budget_left=6)
    im_env.step(graph, law, state, np.asarray([1, 5, 9]), np.random.default_rng(0))
    mask = np.ones(graph.adj.shape[0], bool)
    np.testing.assert_array_equal(op.refine(graph, law, state, 4, mask),
                                  Gate(2).refine(graph, law, state, 4, mask))


@pytest.mark.parametrize("law_name", ["WC", "CC2"])
def test_llm_objective_with_prob_reproduces_hub_refine(law_name):
    graph = _dblp(200)
    law = im_laws.make_law(law_name, graph)
    op = lo.operator_from_answer("im", lo.extract_code_blocks(HUB_EQUIV)[0])
    assert op._takes_prob
    state = im_env.make_state(graph, M=3, budget_left=6)
    im_env.step(graph, law, state, np.asarray([1, 5, 9]), np.random.default_rng(0))
    mask = np.ones(graph.adj.shape[0], bool)
    np.testing.assert_array_equal(op.refine(graph, law, state, 4, mask),
                                  Hub().refine(graph, law, state, 4, mask))


def test_rpp_score_reproduces_harvest():
    loc, initial, factor = points.random_instance(30, seed=1)
    env = rpp_env.RPPEnv(loc=loc, initial_profit=initial, budget=10, T=5, profit_factor=factor)
    state = env.reset(M=2)
    op = lo.operator_from_answer("rpp", lo.extract_code_blocks(HARVEST_EQUIV)[0])
    mask = np.ones(env.n, bool)
    np.testing.assert_array_equal(op.refine(env, None, state, 4, mask),
                                  Harvest().refine(env, None, state, 4, mask))


def test_synthesis_rejects_broken_and_accepts_useful_operator_under_cc3():
    graph = _dblp(300)
    law = im_laws.make_law("CC3", graph)
    chat = FakeChat(["\n".join([BROKEN, SHARED3])])
    lib = OperatorLibrary.standard("im", gates=(2,))
    new_lib, report = lo.synthesize_operators("im", [law], lib, graph, chat, T=6, budget=18,
                                              rounds=1, k=2, delta=0.5, episodes=1)
    assert any("SyntaxError" in e or "failed to load" in e for e in report["errors"])
    assert report["accepted"] == ["llm-shared-3"]
    assert "llm-shared-3" in new_lib.names()


def test_library_with_llm_operator_roundtrips_json():
    code = lo.extract_code_blocks(HARVEST_EQUIV)[0]
    lib = OperatorLibrary([Harvest(), lo.operator_from_answer("rpp", code)])
    back = OperatorLibrary.from_dict(json.loads(json.dumps(lib.to_dict())))
    assert back.names() == lib.names()
    assert back.get("llm-harvest-copy").source == code


@pytest.mark.skipif(os.environ.get("STRATA_LLM") != "1", reason="set STRATA_LLM=1 to call the real LLM")
def test_real_llm_proposes_an_accepted_operator_under_cc3():
    from strata.llm import Chat

    graph = _dblp(300)
    law = im_laws.make_law("CC3", graph)
    lib = OperatorLibrary.standard("im", gates=(2,))
    _, report = lo.synthesize_operators("im", [law], lib, graph, Chat(), T=6, budget=18,
                                        rounds=2, k=3, delta=0.5, episodes=1)
    assert report["accepted"], report
