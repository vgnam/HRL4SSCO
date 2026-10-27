"""Products of programmatic experts (PoE-World style) for the L1 law layer.

Following PoE-World (Piriyakulkij et al., NeurIPS 2025), a law is an
exponentially weighted product of many small programs, each of which only has
to be right about one mechanism:

    p_theta(o) ∝ prod_i p_i(o) ** theta_i,   theta_i >= 0,

normalised per predicted variable.  The weights are fitted by maximum
likelihood with L-BFGS-B, experts with ``theta < prune`` are dropped and the
rest refitted.

* IM: each expert gives the activation probability ``p_i`` of an eligible
  node.  Normalising the product over {activate, stay} gives
  ``p = sigmoid(beta + sum_i theta_i * logit(p_i))`` -- a non-negative
  logistic regression on the experts' logits (``beta`` is a constant expert).
  The environment keeps one exposure array per state, so all experts of one
  product share the same exposure memory (``rho``) and neighbourhood
  (``two_hop``); :func:`build_poe_laws` fits one product per such group.
* RPP: each expert gives a mean and scale for a POI's daily profit change and
  is read as a Gaussian (a uniform of half-width ``s`` as std ``s / sqrt 3``).
  A weighted product of Gaussians is the Gaussian with precision
  ``sum_i theta_i / s_i^2`` and the precision-weighted mean.

The experts' own parameters stay at their individually fitted values; only
the product weights are learned here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from strata.core.types import STATUS_INACTIVE
from strata.laws import dsl
from strata.laws import fit

P_CLIP = 1e-4  # deterministic experts get finite logits (PoE-World adds noise likewise)
SQRT3 = math.sqrt(3.0)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), P_CLIP, 1.0 - P_CLIP)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


# =========================================================================
# IM
# =========================================================================


class PoELaw:
    """``p = sigmoid(beta + sum_i theta_i logit p_i)`` over IM law experts."""

    def __init__(self, experts, theta, beta=0.0, name="poe"):
        self.experts = list(experts)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.beta = float(beta)
        self.rho = float(getattr(self.experts[0], "rho", 0.0) or 0.0)
        self.two_hop = bool(getattr(self.experts[0], "two_hop", False))
        self.name = "%s[%d]" % (name, len(self.experts))
        self.loglik = None
        self.source = "\n\n".join(
            "# expert %d, weight %.3f\n%s" % (i, w, getattr(e, "source", "") or getattr(e, "name", repr(e)))
            for i, (e, w) in enumerate(zip(self.experts, self.theta)))

    def accumulate(self, E, a):
        return a + self.rho * E if self.rho else a

    def prob(self, feats):
        z = np.full(np.shape(feats.a), self.beta, dtype=np.float64)
        for w, e in zip(self.theta, self.experts):
            z = z + w * _logit(e.prob(feats))
        return _sigmoid(z)

    def n_params(self):
        return len(self.experts) + 1

    def __repr__(self):
        return "<%s %s>" % (self.name, np.round(self.theta, 3).tolist())


def _im_events(law, episodes, graph):
    """Expert probabilities and outcomes on every eligible node of every logged round.

    Mirrors :func:`strata.laws.fit.loglik` (same exposure recursion), so the
    product's training likelihood equals ``fit.loglik`` of the fitted product.
    """
    indeg = np.asarray(graph.indeg, dtype=np.float64)
    community = np.asarray(graph.community) if graph.community is not None else np.zeros(indeg.size, int)
    ps, ys = [], []
    for rounds in episodes:
        E = np.zeros(indeg.size, dtype=np.float32)
        for round_ in rounds:
            eligible = np.asarray(fit._field(round_, "eligible"), dtype=bool)
            if not eligible.any():
                continue
            feats, E, _ = fit._features_for_round(law, round_, E, indeg, community, graph)
            ps.append(np.asarray(law.prob(feats)[0], dtype=np.float64)[eligible])
            y = np.asarray(fit._field(round_, "y"), dtype=bool)
            ys.append(y[eligible])
            E = np.where(y, 0.0, E)
            after = np.asarray(fit._field(round_, "after"))
            E[after != STATUS_INACTIVE] = 0.0
    if not ps:
        return np.zeros(0), np.zeros(0, dtype=bool)
    return np.concatenate(ps), np.concatenate(ys)


def _fit_logistic(L, y, l1=0.0):
    """Non-negative logistic regression ``sigmoid(beta + L @ theta)`` (convex)."""
    K = L.shape[1]
    yf = y.astype(np.float64)

    def nll(x):
        beta, theta = x[0], x[1:]
        z = beta + L @ theta
        # log(1 + e^z) - y z, computed stably
        val = np.sum(np.logaddexp(0.0, z) - yf * z) + l1 * theta.sum()
        r = _sigmoid(z) - yf
        grad = np.concatenate([[r.sum()], L.T @ r + l1])
        return val, grad

    x0 = np.concatenate([[0.0], np.full(K, 1.0 / max(K, 1))])
    bounds = [(-20.0, 20.0)] + [(0.0, 20.0)] * K
    res = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})
    return float(res.x[0]), np.asarray(res.x[1:]), -float(res.fun)


def _group_key(law):
    return (round(float(getattr(law, "rho", 0.0) or 0.0), 4), bool(getattr(law, "two_hop", False)))


def fit_poe_im(experts, episodes, graph, prune=0.05, l1=0.0):
    """Fit one product over ``experts`` (all with the same exposure memory)."""
    events = [_im_events(e, episodes, graph) for e in experts]
    y = events[0][1]
    L = np.stack([_logit(p) for p, _ in events], axis=1)
    keep = np.arange(len(experts))
    beta, theta, _ = _fit_logistic(L, y, l1)
    for _ in range(3):  # prune and refit (PoE-World's synthesize-fit-prune step)
        alive = theta >= prune
        if alive.all() or not alive.any():
            break
        keep, L = keep[alive], L[:, alive]
        beta, theta, _ = _fit_logistic(L, y, l1)
    law = PoELaw([experts[i] for i in keep], theta, beta)
    law.loglik = float(fit.loglik(law, episodes, graph.indeg, graph))
    return law


# =========================================================================
# RPP
# =========================================================================


class PoEDriftEffect(dsl.DriftEffect):
    """Precision-weighted product of Gaussian drift experts."""

    def __init__(self, experts, theta, name="poe"):
        self.experts = list(experts)  # LLMDriftEffect-like objects with ``_ms``
        self.theta = np.asarray(theta, dtype=np.float64)
        self.name = "%s[%d]" % (name, len(self.experts))
        self.param_names = tuple("theta%d" % i for i in range(len(self.experts)))
        self.source = "\n\n".join(
            "# expert %d, weight %.3f\n%s" % (i, w, getattr(e, "source", ""))
            for i, (e, w) in enumerate(zip(self.experts, self.theta)))

    def _ms(self, feats):
        prec = np.zeros(np.shape(feats.profit), dtype=np.float64)
        num = np.zeros_like(prec)
        for w, e in zip(self.theta, self.experts):
            mu, s = e._ms(feats)
            sd = s / SQRT3 if getattr(e, "family", "gaussian") == "uniform" else s
            p = w / np.maximum(sd, 1e-6) ** 2
            prec, num = prec + p, num + p * mu
        prec = np.maximum(prec, 1e-12)
        return num / prec, 1.0 / np.sqrt(prec)

    def mean(self, feats):
        return self._ms(feats)[0]

    def logpdf(self, delta, feats):
        mu, s = self._ms(feats)
        d = np.asarray(delta, dtype=np.float64) - mu
        return -0.5 * (d / s) ** 2 - np.log(s * math.sqrt(2.0 * math.pi))

    def sample(self, feats, rng):
        mu, s = self._ms(feats)
        return mu + s * rng.standard_normal(mu.shape)


def _rpp_events(effect, episodes, env, n_regions=4, seed=0):
    """Expert (mean, std) and observed change on every scored POI-day (as loglik_drift)."""
    regions = fit._drift_regions(env, n_regions=n_regions, seed=seed)
    kernel = dsl.spill_kernel(env)
    initial = np.asarray(getattr(env, "initial_profit", np.zeros(env.n)), dtype=np.float64)
    mus, sds, ds = [], [], []
    for episode in episodes:
        for round_ in episode:
            before = np.asarray(fit._field(round_, "profit_before"), dtype=np.float64)
            after = np.asarray(fit._field(round_, "profit_after"), dtype=np.float64)
            vb = np.asarray(fit._field(round_, "visited_before"), dtype=bool)
            va = np.asarray(fit._field(round_, "visited_after"), dtype=bool)
            mask = (~va) & (before > 0.0) & (after > env.profit_min + 1e-9)
            if not mask.any():
                continue
            feats = dsl.DriftFeatures(
                profit=before[None, :], visited=va[None, :],
                t=np.asarray([int(fit._field(round_, "t"))], dtype=np.int64),
                distance=fit._distance_to_action(env, fit._field(round_, "action"))[None, :],
                region=regions, initial=initial[None, :], visited_now=(va & ~vb)[None, :],
                kernel=kernel)
            mu, s = effect._ms(feats)
            sd = s / SQRT3 if getattr(effect, "family", "gaussian") == "uniform" else s
            mus.append(mu[0][mask]); sds.append(np.maximum(sd[0][mask], 1e-6))
            ds.append((after - before)[mask])
    return np.concatenate(mus), np.concatenate(sds), np.concatenate(ds)


def fit_poe_rpp(effects, episodes, env, prune=0.05):
    events = [_rpp_events(e, episodes, env) for e in effects]
    d = events[0][2]
    MU = np.stack([m for m, _, _ in events], axis=1)
    IV = np.stack([1.0 / s ** 2 for _, s, _ in events], axis=1)

    def solve(MU, IV):
        def nll(theta):
            prec = np.maximum(IV @ theta, 1e-12)
            mu = (MU * IV) @ theta / prec
            r = d - mu
            val = np.sum(0.5 * prec * r ** 2 - 0.5 * np.log(prec))
            # gradients of 0.5 prec r^2 - 0.5 log prec w.r.t. theta
            dmu = (IV * (MU - mu[:, None])) / prec[:, None]
            grad = (0.5 * r[:, None] ** 2 * IV - (prec * r)[:, None] * dmu
                    - 0.5 * IV / prec[:, None]).sum(axis=0)
            return val, grad

        K = MU.shape[1]
        res = minimize(nll, np.full(K, 1.0 / K), jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 50.0)] * K, options={"maxiter": 500})
        return np.asarray(res.x)

    keep = np.arange(len(effects))
    theta = solve(MU, IV)
    for _ in range(3):
        alive = theta >= prune * theta.max()
        if alive.all() or not alive.any():
            break
        keep, MU, IV = keep[alive], MU[:, alive], IV[:, alive]
        theta = solve(MU, IV)
    effect = PoEDriftEffect([effects[i] for i in keep], theta)
    program = dsl.DriftProgram(effect)
    program.source = effect.source
    program.params = {}
    program.loglik = float(fit.loglik_drift(program, episodes, env))
    return program


# =========================================================================
# Entry point
# =========================================================================


def build_poe_laws(domain, candidates, episodes, obj, max_experts=12, prune=0.05):
    """Products of experts over ``candidates``; returns ``(laws, report)``.

    IM: one product per exposure-memory group with at least two experts (the
    ``max_experts`` best by training likelihood).  RPP: one product over the
    LLM drift effects.
    """
    domain = str(domain).lower()
    report: Dict[str, Any] = {"groups": []}
    out: List[Any] = []
    ranked = sorted(candidates, key=lambda l: -float(getattr(l, "loglik", -np.inf) or -np.inf))
    if domain == "im":
        groups: Dict[Tuple[float, bool], List[Any]] = {}
        for law in ranked:
            groups.setdefault(_group_key(law), []).append(law)
        for key, laws in groups.items():
            laws = laws[:max_experts]
            if len(laws) < 2:
                continue
            try:
                poe = fit_poe_im(laws, episodes, obj, prune=prune)
            except (ValueError, FloatingPointError) as e:
                report["groups"].append({"key": key, "error": str(e)})
                continue
            out.append(poe)
            report["groups"].append({"key": key, "experts": len(laws), "kept": len(poe.experts),
                                     "theta": np.round(poe.theta, 3).tolist(),
                                     "beta": round(poe.beta, 3), "loglik": poe.loglik})
        return out, report
    effects = [getattr(l, "effect", None) for l in ranked]
    effects = [e for e in effects if e is not None and hasattr(e, "_ms")][:max_experts]
    if len(effects) >= 2:
        try:
            poe = fit_poe_rpp(effects, episodes, obj, prune=prune)
            out.append(poe)
            report["groups"].append({"experts": len(effects), "kept": len(poe.effect.experts),
                                     "theta": np.round(poe.effect.theta, 3).tolist(),
                                     "loglik": poe.loglik})
        except (ValueError, FloatingPointError) as e:
            report["groups"].append({"error": str(e)})
    return out, report
