"""M5 tests for the flat, model-based and WS-Option baselines.

The quantitative acceptance test is the plan's IM cell: under CC2 on dblp with
``n = 100``, LibB (level-2 MLE) must recover the oracle decision (regret ~0)
while the WC default must be worse, the same direction as
``experiments/kt_law/out/t2_fit_CC2_100.out``.
"""

import csv
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from strata.baselines import flat as flat_mod  # noqa: E402
from strata.baselines import model_based  # noqa: E402
from strata.baselines import ws_option  # noqa: E402
from strata.eval import protocol  # noqa: E402
from strata.eval import report as report_mod  # noqa: E402
from strata.eval import run as run_mod  # noqa: E402


def test_allocation_schedule():
    # Remaining is held at 12 to expose the schedule itself.
    assert [flat_mod.allocation_k("even", t, 6, 12) for t in range(6)] == [
        2,
        3,
        3,
        4,
        6,
        12,
    ]
    assert [flat_mod.allocation_k("normal", t, 6, 12) for t in range(6)] == [
        12,
        0,
        0,
        0,
        0,
        0,
    ]
    assert [flat_mod.allocation_k("static", t, 6, 12) for t in range(6)] == [
        4,
        0,
        6,
        0,
        12,
        0,
    ]
    # The actual closed loop spends an even 2 per round at T=6, B=12.
    remaining = 12
    spent = []
    for t in range(6):
        k = flat_mod.allocation_k("even", t, 6, remaining)
        spent.append(k)
        remaining -= k
    assert spent == [2] * 6 and remaining == 0
    assert flat_mod.allocation_k("even", 0, 6, 0) == 0


def test_im_flat_families_and_best_flat():
    instance = protocol.make_instance(
        "im", dataset="dblp", n=80, graph_seed=5, law="CC2", T=5, B=10
    )
    hub = flat_mod.best_flat(instance, ops=flat_mod.hub_ops("im"), episodes=1, seed=3)
    assert hub.op == "hub"
    assert hub.alloc in flat_mod.ALLOCS

    any_op = flat_mod.best_flat(instance, episodes=1, seed=3)
    assert any_op.op in flat_mod.IM_OPS
    assert set(any_op.scores) == {
        (op, alloc) for op in flat_mod.IM_OPS for alloc in flat_mod.ALLOCS
    }
    values = protocol.evaluate(any_op, instance, episodes=1, seed=3)
    assert values.shape == (1,)


def test_rpp_flat_families_and_best_flat():
    instance = protocol.make_instance(
        "rpp", dataset=None, n=35, graph_seed=2, law="IID", T=5, B=10
    )
    hub = flat_mod.best_flat(instance, ops=flat_mod.hub_ops("rpp"), episodes=1, seed=1)
    assert hub.op == "harvest"
    any_op = flat_mod.best_flat(instance, episodes=1, seed=1)
    assert any_op.op in flat_mod.RPP_OPS
    assert protocol.evaluate(any_op, instance, episodes=1, seed=1).shape == (1,)


@pytest.mark.slow
def test_im_libb_recovers_oracle_and_wc_default_is_worse():
    instance = protocol.make_instance(
        "im", dataset="dblp", n=100, graph_seed=101, law="CC2", T=10, B=20
    )
    logs = protocol.make_logs(instance, n_episodes=60, seed=101)

    oracle = model_based.Oracle(logs, selection_episodes=2, seed=7)
    libb = model_based.LibB(logs, selection_episodes=2, seed=7)
    wc = model_based.WCDefault(logs, selection_episodes=2, seed=7)

    oracle_values = protocol.evaluate(oracle, instance, episodes=2, seed=7)
    libb_values = protocol.evaluate(libb, instance, episodes=2, seed=7)
    wc_values = protocol.evaluate(wc, instance, episodes=2, seed=7)

    libb_regret = protocol.regret(libb_values, oracle_values)
    wc_regret = protocol.regret(wc_values, oracle_values)

    # Level 2 can express CC2; the selected gate-2 law gives the oracle value.
    assert abs(libb_regret.mean) <= 0.01
    # Assuming the repo's weighted cascade is clearly worse (T2: +22.4%).
    assert wc_regret.mean > 0.01
    assert libb.law is not None and getattr(libb.law, "two_hop", False) is False


def _require_ws_option():
    """Skip (not xfail) when the interpreter cannot load torch or the agents."""
    try:
        pytest.importorskip("torch")
    except Exception as exc:  # torch is installed but its DLLs fail to load
        pytest.skip("PyTorch cannot be loaded in this interpreter: %s" % (exc,))
    if not ws_option.available():
        pytest.skip("the original WS-Option agents cannot be imported")


@pytest.mark.slow
def test_ws_option_tiny_run_returns_value():
    _require_ws_option()
    instance = protocol.make_instance(
        "im", dataset="dblp", n=50, graph_seed=1, law="CC2", T=3, B=6
    )
    value = ws_option.run_ws_option(
        instance, epochs=1, iterations=1, episodes=1, seed=0
    )
    assert np.isfinite(value)


@pytest.mark.slow
def test_ws_option_rpp_tiny_run_returns_value():
    _require_ws_option()
    instance = protocol.make_instance(
        "rpp", dataset=None, n=20, graph_seed=3, law="IID", T=3, B=6
    )
    value = ws_option.run_ws_option(
        instance, epochs=1, iterations=1, episodes=1, seed=0
    )
    assert np.isfinite(value)


def test_ws_option_im_adapter_matches_reference_under_wc():
    """Under the true WC law the adapter must mirror ``IM/Env.py`` in mean."""
    _require_ws_option()

    import importlib.util

    import networkx as nx

    instance = protocol.make_instance(
        "im", dataset="dblp", n=50, graph_seed=2, law="WC", T=3, B=6
    )
    graph = instance.graph
    nxg = nx.from_scipy_sparse_array(graph.adj, create_using=nx.DiGraph())
    nxg.add_nodes_from(range(graph.adj.shape[0]))

    spec = importlib.util.spec_from_file_location(
        "_im_reference_env", os.path.join(REPO, "IM", "Env.py")
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    plan_rng = np.random.default_rng(123)
    plan = [plan_rng.choice(graph.adj.shape[0], 2, replace=False) for _ in range(3)]

    rollouts = 2000
    ref_values = np.zeros(rollouts)
    env = reference.Env(graph=nxg.copy(), budget=6, T=3)
    for r in range(rollouts):
        import random

        random.seed(10000 + r)
        env.reset()
        total = 0.0
        for seeds in plan:
            state, reward, _, _ = env.step(env.state, seeds)
            env.state = state
            total += reward
        ref_values[r] = total

    adapter = ws_option.IMEnvAdapter(graph, instance.law, budget=6, T=3, seed=0)
    adapter_values = np.zeros(rollouts)
    for r in range(rollouts):
        adapter.rng = np.random.default_rng(10000 + r)
        adapter.reset()
        state = adapter.state
        total = 0.0
        for seeds in plan:
            state, reward, _, _ = adapter.step(state, seeds)
            total += reward
        adapter_values[r] = total

    mean_ref = float(ref_values.mean())
    mean_ad = float(adapter_values.mean())
    se_ref = float(ref_values.std(ddof=1) / np.sqrt(rollouts))
    se_ad = float(adapter_values.std(ddof=1) / np.sqrt(rollouts))
    se_diff = float(np.hypot(se_ref, se_ad))
    assert abs(mean_ad - mean_ref) <= 3.0 * se_diff


@pytest.mark.slow
def test_end_to_end_im_run_and_report(tmp_path, capsys):
    out = str(tmp_path / "strata_im_e2e.csv")
    path = run_mod.main(
        [
            "--domain",
            "im",
            "--datasets",
            "dblp",
            "--n",
            "300",
            "--laws",
            "CC2",
            "--methods",
            "flat-hub*",
            "flat-any*",
            "wc-default",
            "libA",
            "libB",
            "libC",
            "oracle",
            "ws-option",
            "--episodes",
            "2",
            "--log-episodes",
            "40",
            "--T",
            "6",
            "--budget",
            "12",
            "--seed",
            "0",
            "--out",
            out,
        ]
    )
    assert path == out
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    methods = {row["method"] for row in rows}
    assert methods == {
        "flat-hub*",
        "flat-any*",
        "wc-default",
        "libA",
        "libB",
        "libC",
        "oracle",
    }
    assert len(rows) == 7 * 2  # one row per (method, episode)

    table = report_mod.main([path])
    for method in methods:
        assert method in table

    captured = capsys.readouterr()
    if not ws_option.available():
        assert "ws-option" in captured.err


def test_end_to_end_rpp_run_and_report(tmp_path, capsys):
    out = str(tmp_path / "strata_rpp_e2e.csv")
    path = run_mod.main(
        [
            "--domain",
            "rpp",
            "--datasets",
            "random",
            "--n",
            "40",
            "--laws",
            "TREND",
            "--methods",
            "flat-hub*",
            "flat-any*",
            "wc-default",
            "libA",
            "libB",
            "libC",
            "oracle",
            "ws-option",
            "--episodes",
            "2",
            "--log-episodes",
            "20",
            "--T",
            "6",
            "--budget",
            "12",
            "--seed",
            "0",
            "--out",
            out,
        ]
    )
    assert path == out
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7 * 2
    assert all(np.isfinite(float(row["regret"])) for row in rows)

    table = report_mod.main([path])
    assert "TREND" in table and "libB" in table
    captured = capsys.readouterr()
    if not ws_option.available():
        assert "ws-option" in captured.err
