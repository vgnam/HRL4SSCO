"""Flat DQN baseline (strata.baselines.flat_dqn)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from strata.baselines.flat_dqn import FlatDQNPolicy  # noqa: E402
from strata.eval import protocol  # noqa: E402


@pytest.mark.parametrize("domain", ["im", "rpp"])
def test_flat_dqn_trains_and_plays_valid_rounds(domain):
    if domain == "im":
        inst = protocol.make_instance("im", dataset="dblp", n=80, graph_seed=0, law="WC", T=4, B=8)
    else:
        inst = protocol.make_instance("rpp", dataset=None, n=25, graph_seed=0, law="IID", T=4, B=8)
    policy = FlatDQNPolicy(episodes=6, seed=0)
    values = protocol.evaluate(policy, inst, episodes=2, seed=0)
    assert np.all(np.isfinite(values))
    assert len(policy.train_log) == 6
    # greedy rounds respect the budget and pick distinct, still-available nodes
    view = policy.view
    state = view.reset()
    used = 0
    while not view.done(state):
        action = policy.act(state, 0)
        assert len(set(action.tolist())) == len(action)
        assert np.all(view.valid(state, [])[action])
        used += len(action)
        view.step(state, list(action), np.random.default_rng(0))
    assert used <= inst.B
