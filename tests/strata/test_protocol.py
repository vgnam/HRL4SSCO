"""M5 tests for the benchmark protocol (instances, logs, evaluation, regret)."""

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from strata.baselines.flat import FlatPolicy  # noqa: E402
from strata.eval import protocol  # noqa: E402
from strata.eval.report import format_table, summarise  # noqa: E402


@pytest.fixture(scope="module")
def im_instance():
    return protocol.make_instance(
        "im",
        dataset="dblp",
        n=60,
        graph_seed=101,
        law="CC2",
        T=5,
        B=10,
    )


@pytest.fixture(scope="module")
def rpp_instance():
    return protocol.make_instance(
        "rpp",
        dataset=None,
        n=40,
        graph_seed=3,
        law="TREND",
        T=6,
        B=12,
    )


def test_evaluate_is_deterministic_given_seed(im_instance):
    policy = FlatPolicy("hub", "even")
    a = protocol.evaluate(policy, im_instance, episodes=3, seed=11)
    b = protocol.evaluate(policy, im_instance, episodes=3, seed=11)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3,)
    assert np.all(np.isfinite(a))


def test_evaluate_rpp_is_deterministic_given_seed(rpp_instance):
    policy = FlatPolicy("harvest", "even")
    a = protocol.evaluate(policy, rpp_instance, episodes=2, seed=4)
    b = protocol.evaluate(policy, rpp_instance, episodes=2, seed=4)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isfinite(a))


def test_instance_rpp_empty_day_flag():
    default = protocol.Instance("rpp", law=None, n=20, graph_seed=1)
    original = protocol.Instance(
        "rpp", law=None, n=20, graph_seed=1, empty_day_advances=False
    )
    # STRATA semantics by default, original RPP/Env.py semantics on request.
    assert default.env.empty_day_advances is True
    assert original.env.empty_day_advances is False


def test_make_logs_use_domain_builders(im_instance, rpp_instance):
    im_logs = protocol.make_logs(im_instance, n_episodes=4, seed=2)
    assert len(im_logs) == 4
    assert all(len(episode) == im_instance.T for episode in im_logs)

    rpp_logs = protocol.make_logs(rpp_instance, n_episodes=3, seed=2)
    assert len(rpp_logs) == 3
    assert all(len(episode) <= rpp_instance.T for episode in rpp_logs)


def test_regret_relative_and_paired_se():
    oracle = np.array([10.0, 10.0, 10.0])
    values = np.array([9.0, 10.0, 11.0])
    reg = protocol.regret(values, oracle)
    # Paired differences are [1, 0, -1] -> mean 0, SE > 0.
    assert reg.mean == pytest.approx(0.0)
    assert reg.se > 0.0
    assert reg.n == 3
    assert reg.mean_oracle == pytest.approx(10.0)

    same = protocol.regret(oracle, oracle)
    assert same.mean == pytest.approx(0.0)
    assert same.se == pytest.approx(0.0)


def test_regret_is_positive_when_policy_is_worse():
    oracle = np.array([10.0, 20.0])
    values = np.array([5.0, 10.0])
    reg = protocol.regret(values, oracle)
    assert reg.mean == pytest.approx(0.5)
    assert reg.mean_value == pytest.approx(7.5)


def test_report_aggregates_per_episode_rows():
    rows = [
        {
            "law": "CC2",
            "dataset": "dblp",
            "method": "libB",
            "value": "1",
            "regret": "0.0",
        },
        {
            "law": "CC2",
            "dataset": "dblp",
            "method": "libB",
            "value": "1",
            "regret": "0.2",
        },
        {
            "law": "CC2",
            "dataset": "dblp",
            "method": "wc-default",
            "value": "1",
            "regret": "0.5",
        },
    ]
    summary = summarise(rows)
    key = ("CC2", "dblp", "libB")
    assert summary[key]["mean"] == pytest.approx(0.1)
    assert summary[key]["max"] == pytest.approx(0.2)
    assert summary[key]["se"] == pytest.approx(0.1)
    table = format_table(summary)
    assert "libB" in table and "wc-default" in table
