"""Problem-shaped mixtures of experts (strata.laws.mixture)."""

from __future__ import annotations

import numpy as np
import pytest

from strata.core.types import Features
from strata.domains.im import hidden_laws as im_laws
from strata.eval import protocol
from strata.laws import fit, mixture
from strata.laws.llm_propose import wrap_llm_law


@pytest.fixture(scope="module")
def het():
    inst = protocol.make_instance("im", dataset="dblp", n=150, graph_seed=0, law="HET", T=8, B=16)
    logs = list(protocol.make_logs(inst, policies=None, n_episodes=80, seed=0))
    return inst, logs[:25], logs[25:]


def test_gated_mixture_recovers_a_heterogeneous_law_that_a_product_cannot(het):
    inst, val, train = het
    g = inst.graph
    experts = [im_laws.make_law(k, g) for k in ("WC", "CC2", "LT")]
    pool = mixture._Pool("im", experts, train, val, g)
    _, ll1 = pool.fit([0, 1, 2], C=1)
    _, ll2 = pool.fit([0, 1, 2], C=2)
    true_ll = fit.loglik(inst.law, val, g.indeg, g)
    assert ll2 > ll1 + 50.0  # node types need gating, not a product
    assert ll2 > true_ll - 0.02 * abs(true_ll)


def test_built_mixture_matches_fit_loglik_and_alternatives_stay_within_eps(het):
    inst, val, train = het
    g = inst.graph
    experts = [im_laws.make_law(k, g) for k in ("WC", "CC2", "LT", "UNI")]
    contexts, report = mixture.build_mixtures("im", experts, train, val, g, components=(1, 2))
    ctx = contexts[0]
    assert np.isclose(fit.loglik(ctx.law, val, g.indeg, g), ctx.val_ll, rtol=1e-6)
    assert report["groups"][0]["components"] == 2
    for alt in ctx.alternatives(eps=5.0, n_boot=2):
        assert alt.ll_val >= ctx.val_ll - 5.0


def test_gate_updates_online_and_resets(het):
    inst, val, train = het
    g = inst.graph
    experts = [im_laws.make_law(k, g) for k in ("WC", "CC2")]
    pool = mixture._Pool("im", experts, train, val, g)
    params, _ = pool.fit([0, 1], C=2)
    law = pool.make([0, 1], params)
    n = g.indeg.size
    v = int(np.argmax(g.indeg))
    a = np.zeros((1, n), np.float32)
    a[0, v] = 1.0  # one active in-neighbour: CC2 says "no", WC says "maybe"
    feats = Features(a=a, E=a.copy(), a2=None, eligible=np.ones((1, n), bool),
                     indeg=g.indeg, community=g.community)
    pc = law.component_probs(feats)[:, 0, v]
    before = law.gate[v].copy()
    y = np.zeros(n, bool)
    y[v] = True  # it activated anyway: the component with the higher p gains weight
    law.observe_round(feats, y, np.ones(n, bool))
    hi = int(np.argmax(pc))
    assert law.gate[v, hi] >= before[hi]
    assert np.isclose(law.gate[v].sum(), 1.0)
    law.reset_gate()
    assert np.allclose(law.gate, law.gate0)


class _Mean:
    family = "gaussian"
    params = {"s": 0.05}
    bounds = {"s": (1e-4, 0.5)}

    def __init__(self, kind):
        self.kind = kind

    def mean(self, f):
        if self.kind == "region":
            return np.broadcast_to((f.region % 2 == 0) * 0.02 - 0.01, f.profit.shape).astype(float)
        return np.zeros_like(f.profit)

    def scale(self, f):
        return np.full_like(f.profit, self.params["s"])


def test_rpp_additive_mixture_picks_uniform_noise_and_matches_loglik():
    inst = protocol.make_instance("rpp", dataset=None, n=40, graph_seed=0, law="IID", T=6, B=12)
    logs = list(protocol.make_logs(inst, policies=None, n_episodes=30, seed=0))
    val, train = logs[:10], logs[10:]
    env = inst.env
    experts = [wrap_llm_law(_Mean(k), "rpp", source="%s" % k) for k in ("region", "zero")]
    contexts, _ = mixture.build_mixtures("rpp", experts, train, val, env)
    ctx = contexts[0]
    assert ctx.law.effect.family == "uniform"  # the benchmark's base noise is uniform
    assert np.isclose(fit.loglik_drift(ctx.law, val, env), ctx.val_ll, rtol=1e-6)
    feats = mixture.dsl.drift_features(env, env.reset(M=2), np.zeros((2, env.n), bool))
    sample = ctx.law.effect.sample(feats, np.random.default_rng(0))
    assert sample.shape == (2, env.n) and np.all(np.isfinite(sample))
