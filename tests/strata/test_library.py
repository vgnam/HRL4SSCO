"""M2a acceptance tests for the operator library (macros, validation, I/O)."""

import json
import os
import tempfile

import networkx as nx
import numpy as np
import pytest

from strata.abstraction import OperatorLibrary, Scenario, make_operator
from strata.abstraction.library import even_allocation, k_options
from strata.abstraction.operators import Gate, Harvest, Hedge, Hub, Wait
from strata.abstraction.regions import make_spatial_regions
from strata.domains.im import env as im_env
from strata.domains.im import graph as im_graph
from strata.domains.im import hidden_laws as im_laws
from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import points


def _graph():
    return im_graph.build_graph(
        nx.gnp_random_graph(36, 0.18, seed=7, directed=True), name="validate"
    )


def test_even_schedule_and_k_options():
    assert even_allocation(10, 2, 6) == 3
    assert even_allocation(10, 5, 6) == 10
    assert k_options(10, 3, 6) == [0, 4, 8, 10]
    assert k_options(0, 0, 10) == [0]


def test_macros_expand_every_operator_and_region():
    lib = OperatorLibrary.standard("im")
    z = (2, 12, 0, 1)  # two regions -> two bucket ids
    macros = lib.macros(z, T=8)
    ks = {m.k for m in macros}
    assert 0 in ks
    assert max(ks) == 12
    wait = [m for m in macros if m.k == 0]
    assert len(wait) == 1 and wait[0].op == "wait"
    for m in macros:
        if m.op != "wait":
            assert m.region in (0, 1)
    names = {m.op for m in macros if m.k > 0}
    assert "hub" in names and "gate-2" in names


def test_macros_of_rpp_library():
    lib = OperatorLibrary.standard("rpp")
    macros = lib.macros((0, 6, 0, 1), T=5)
    names = {m.op for m in macros}
    assert {"harvest", "sweep", "reposition", "wait"} <= names


def test_save_load_roundtrip():
    lib = OperatorLibrary([Hub(), Gate(3), Hedge(2), Wait()])
    path = os.path.join(tempfile.gettempdir(), "strata_m2a_library.json")
    lib.save(path)
    loaded = OperatorLibrary.load(path)
    assert loaded.names() == lib.names()
    gate = loaded.get("gate-3")
    assert isinstance(gate, Gate) and gate.gate == 3
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == 1


def test_make_operator_by_macro_name():
    op = make_operator("hedge-3")
    assert isinstance(op, Hedge) and op.gate == 3
    assert make_operator("harvest").name == "harvest"


def test_validate_keeps_hub_against_wait():
    G = _graph()
    state = im_env.make_state(G, M=3, budget_left=20)
    law = im_laws.make_law("CC2", G)
    scenario = Scenario("im", G, state, np.ones(G.adj.shape[0], bool), 6)
    lib = OperatorLibrary([Hub(), Wait()])
    assert lib.validate(lib.get("hub"), [law], [scenario], tau=0.0, seed=1)
    assert lib.validate(lib.get("wait"), [law], [scenario], tau=0.0, seed=1)


def test_validate_rejects_dominated_operator():
    G = _graph()
    state = im_env.make_state(G, M=2, budget_left=20)
    law = im_laws.make_law("CC2", G)
    scenario = Scenario("im", G, state, np.ones(G.adj.shape[0], bool), 6)
    lib = OperatorLibrary([Wait(), Wait()])
    # Two identical operators cannot be distinguished, so the second is dropped.
    assert not lib.validate(lib[1], [law], [scenario], tau=0.0)


def test_validate_rpp_harvest():
    loc, profit, _ = points.random_instance(40, seed=11)
    env = rpp_env.RPPEnv(loc, profit, budget=20, T=6, empty_day_advances=True)
    state = env.reset(M=2)
    regs = make_spatial_regions(env, k_top=2)
    scenario = Scenario("rpp", env, state, regs[1][1], 5)
    lib = OperatorLibrary([Harvest(), Wait()])
    law = rpp_laws.IID(seed=0)
    assert lib.validate(lib.get("harvest"), [law], [scenario], tau=0.0, seed=2)


def test_from_dict_unknown_type_raises():
    lib = OperatorLibrary()
    with pytest.raises(ValueError):
        OperatorLibrary.from_dict(
            {"version": 1, "operators": [{"type": "Nope", "name": "x", "params": {}}]}
        )
    assert lib.names() == []


def test_macros_explicit_regions_and_rpp_day_index():
    lib = OperatorLibrary.standard("im")
    # several bucketed features per region: the count must come from n_regions
    z = (0, 12, 0, 1, 2, 3, 0, 1)
    regions = {m.region for m in lib.macros(z, T=6, n_regions=2) if m.op != "wait"}
    assert regions == {0, 1}
    rpp = OperatorLibrary.standard("rpp")
    # RPP days are one-based: on day T the even share is the whole remaining budget
    last = {m.k for m in rpp.macros((5, 6, 0, 1), T=5, n_regions=1)}
    assert last == {0, 6}
    first = {m.k for m in rpp.macros((1, 6, 0, 1), T=5, n_regions=1)}
    assert max(first) == 6 and min(first) == 0 and len(first) > 2
