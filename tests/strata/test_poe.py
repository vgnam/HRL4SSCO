"""Products of programmatic experts (strata.laws.poe) for IM and RPP."""

from __future__ import annotations

import numpy as np

from strata.domains.im import hidden_laws as im_laws
from strata.eval import protocol
from strata.laws import fit, poe
from strata.laws.llm_propose import wrap_llm_law


def _im(law, n=150, episodes=40):
    inst = protocol.make_instance("im", dataset="dblp", n=n, graph_seed=0, law=law, T=6, B=12)
    logs = protocol.make_logs(inst, policies=None, n_episodes=episodes, seed=0)
    return inst.graph, list(logs)


def test_im_poe_loglik_matches_fit_and_beats_its_experts_on_a_mixture_law():
    graph, logs = _im("HET")
    experts = [im_laws.make_law(k, graph) for k in ("WC", "CC2", "LT")]
    lls = [fit.loglik(e, logs, graph.indeg, graph) for e in experts]
    law = poe.fit_poe_im(experts, logs, graph)
    # the product's likelihood is the one fit.loglik computes for the product law
    assert np.isclose(law.loglik, fit.loglik(law, logs, graph.indeg, graph))
    assert law.loglik > max(lls)
    assert np.all(law.theta >= 0)


def test_im_poe_weights_the_true_expert_under_wc():
    graph, logs = _im("WC")
    experts = [im_laws.make_law(k, graph) for k in ("WC", "CC2")]
    law = poe.fit_poe_im(experts, logs, graph)
    names = [e.name for e in law.experts]
    assert "WC" in names
    w = dict(zip(names, law.theta))
    assert w["WC"] > w.get("CC2", 0.0)


def test_im_poe_groups_by_exposure_memory():
    graph, logs = _im("WC", episodes=20)
    experts = [im_laws.make_law(k, graph) for k in ("WC", "UNI", "REINF")]  # REINF has rho = 1
    laws, report = poe.build_poe_laws("im", experts, logs, graph)
    assert len(laws) == 1 and laws[0].rho == 0.0  # REINF alone in its group: no product
    assert report["groups"][0]["experts"] == 2


class _Mean:
    family = "gaussian"
    params = {"s": 0.05}
    bounds = {"s": (1e-4, 0.5)}

    def __init__(self, c):
        self.c = c

    def mean(self, f):
        return np.full_like(f.profit, self.c)

    def scale(self, f):
        return np.full_like(f.profit, self.params["s"])


def test_rpp_poe_fits_and_does_not_lose_to_its_experts():
    inst = protocol.make_instance("rpp", dataset=None, n=40, graph_seed=0, law="TREND", T=6, B=12)
    logs = list(protocol.make_logs(inst, policies=None, n_episodes=20, seed=0))
    env = inst.env
    programs = [wrap_llm_law(_Mean(c), "rpp", source="mean %g" % c) for c in (0.0, 0.02, -0.02)]
    lls = [fit.loglik_drift(p, logs, env) for p in programs]
    laws, report = poe.build_poe_laws("rpp", programs, logs, env)
    assert len(laws) == 1
    assert laws[0].loglik >= max(lls) - 1e-6
    sample = laws[0].effect.sample(
        poe.dsl.drift_features(env, env.reset(M=2), np.zeros((2, env.n), bool)),
        np.random.default_rng(0))
    assert sample.shape == (2, env.n) and np.all(np.isfinite(sample))
