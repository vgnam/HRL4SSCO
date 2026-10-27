"""Mixtures of programmatic experts shaped by the SSCO problems (L1 law layer).

PoE-World multiplies every expert into one distribution.  Our two benchmarks
have different structure, so the mixing follows it instead:

* IM -- *gated mixture of logit-sum components*.  Mechanisms that act on the
  same node at once (degree normalisation, thresholds, fatigue) add up in
  log-odds inside a component, ``z_c = beta_c + sum_i theta_ci logit p_i``
  (``theta >= 0``).  Heterogeneous populations (nodes following different
  laws) are separate components with a per-node gate ``r_vc``:
  ``p_v = sum_c r_vc sigmoid(z_c(v))``.  Fitted by EM on the logs (node
  posteriors in the E-step, weighted non-negative logistic regressions in the
  M-step); the gate is a per-node posterior that :meth:`GatedMixLaw.observe_round`
  keeps updating online from the activations the agent observes.  All experts
  of one mixture share an exposure memory (the environment stores a single
  exposure array), so one mixture is fitted per ``(rho, two_hop)`` group.
* RPP -- *additive effects*.  Drift mechanisms (regional trends, spill-over,
  congestion, reversion) add up: ``delta = b + sum_i theta_i mu_i`` with
  ``theta >= 0``, plus a noise family (uniform or Gaussian) and width fitted
  separately -- the benchmark's base noise is uniform.

Experts are pruned by held-out likelihood (backward elimination: an expert
stays only if removing it costs more than ``delta`` validation log-likelihood).
:class:`MixContext` also produces *alternative* mixtures (bootstrap refits and
leave-one-expert-out) that fit the held-out data almost as well; the agent keeps
the ones that change the planner's first decision as extra hypotheses.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import lsq_linear, minimize, minimize_scalar

from strata.core.types import STATUS_INACTIVE
from strata.laws import dsl
from strata.laws import fit

P_CLIP = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), P_CLIP, 1.0 - P_CLIP)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _summary(law):
    """One line per expert for prompts and reports (its docstring if it has one)."""
    src = getattr(law, "source", "") or ""
    m = re.search(r'"""(.*?)"""', src, flags=re.S)
    if m:
        return m.group(1).strip().splitlines()[0][:120]
    return str(getattr(law, "name", type(law).__name__))[:120]


# =========================================================================
# IM: gated mixture of logit-sum components
# =========================================================================


class GatedMixLaw:
    """``p_v = sum_c r_vc sigmoid(beta_c + sum_i theta_ci logit p_i(v))``."""

    def __init__(self, experts, theta, beta, gate, name="mix"):
        self.experts = list(experts)
        self.theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))  # (C, K)
        self.beta = np.asarray(beta, dtype=np.float64).reshape(-1)  # (C,)
        self.gate0 = np.asarray(gate, dtype=np.float64)  # (n, C) posterior after the logs
        self.gate = self.gate0.copy()
        self.rho = float(getattr(self.experts[0], "rho", 0.0) or 0.0)
        self.two_hop = bool(getattr(self.experts[0], "two_hop", False))
        C, K = self.theta.shape
        self.name = "%s[C=%d,K=%d]" % (name, C, K)
        self.loglik = None
        self.ll_val = None
        lines = []
        for i, e in enumerate(self.experts):
            w = ", ".join("%.2f" % self.theta[c, i] for c in range(C))
            lines.append("# expert %d (weight per component: %s): %s" % (i, w, _summary(e)))
        share = self.gate0.mean(axis=0)
        lines.append("# components: %d, share of nodes %s, bias %s" % (
            C, np.round(share, 2).tolist(), np.round(self.beta, 2).tolist()))
        self.source = "\n".join(lines)

    # ------------------------------------------------------------------ law
    def accumulate(self, E, a):
        return a + self.rho * E if self.rho else a

    def _logits(self, feats):
        return np.stack([_logit(e.prob(feats)) for e in self.experts], axis=0)  # (K, M, n)

    def component_probs(self, feats):
        Lx = self._logits(feats)
        z = self.beta[:, None, None] + np.tensordot(self.theta, Lx, axes=([1], [0]))
        return _sigmoid(z)  # (C, M, n)

    def prob(self, feats):
        pc = self.component_probs(feats)
        return np.einsum("cmn,nc->mn", pc, self.gate)

    def n_params(self):
        C, K = self.theta.shape
        return C * (K + 1) + (C - 1)

    # --------------------------------------------------------- online gates
    def reset_gate(self):
        self.gate = self.gate0.copy()

    def observe_round(self, feats, y, eligible):
        """Bayes update of the node gates from one observed round (path 0)."""
        if self.gate.shape[1] == 1:
            return
        pc = np.clip(self.component_probs(feats)[:, 0, :], P_CLIP, 1.0 - P_CLIP)  # (C, n)
        el = np.asarray(eligible, dtype=bool).reshape(-1)
        yy = np.asarray(y, dtype=bool).reshape(-1)
        like = np.where(yy[None, :], pc, 1.0 - pc).T  # (n, C)
        post = self.gate * like
        post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-300)
        self.gate = np.where(el[:, None], post, self.gate)

    def __repr__(self):
        return "<%s>" % self.name


def im_expert_events(law, episodes, graph):
    """Expert logit, outcome and node id of every eligible node of every round.

    Uses the expert's own exposure recursion, exactly as :func:`fit.loglik`.
    """
    indeg = np.asarray(graph.indeg, dtype=np.float64)
    n = indeg.size
    community = np.asarray(graph.community) if graph.community is not None else np.zeros(n, int)
    Ls, ys, vs = [], [], []
    for rounds in episodes:
        E = np.zeros(n, dtype=np.float32)
        for round_ in rounds:
            eligible = np.asarray(fit._field(round_, "eligible"), dtype=bool)
            if not eligible.any():
                continue  # as fit.loglik: such rounds leave the exposure untouched
            y = np.asarray(fit._field(round_, "y"), dtype=bool)
            feats, E, _ = fit._features_for_round(law, round_, E, indeg, community, graph)
            idx = np.flatnonzero(eligible)
            Ls.append(_logit(np.asarray(law.prob(feats)[0], dtype=np.float64)[idx]))
            ys.append(y[idx])
            vs.append(idx)
            E = np.where(y & eligible, 0.0, E)
            after = np.asarray(fit._field(round_, "after"))
            E[after != STATUS_INACTIVE] = 0.0
    if not Ls:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, np.int64)
    return np.concatenate(Ls), np.concatenate(ys), np.concatenate(vs)


def _fit_logistic(L, y, w=None, x0=None):
    """Weighted non-negative logistic regression ``sigmoid(beta + L theta)`` (convex)."""
    K = L.shape[1]
    yf = y.astype(np.float64)
    w = np.ones_like(yf) if w is None else np.asarray(w, dtype=np.float64)

    def nll(x):
        z = x[0] + L @ x[1:]
        r = w * (_sigmoid(z) - yf)
        return float(np.sum(w * (np.logaddexp(0.0, z) - yf * z))), np.concatenate([[r.sum()], L.T @ r])

    if x0 is None:
        x0 = np.concatenate([[0.0], np.full(K, 1.0 / max(K, 1))])
    res = minimize(nll, x0, jac=True, method="L-BFGS-B",
                   bounds=[(-20.0, 20.0)] + [(0.0, 20.0)] * K, options={"maxiter": 300})
    return np.asarray(res.x)


def _bern_ll(p, y):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.where(y, np.log(p), np.log1p(-p))


def _node_loglik(pc, y, v, n):
    """(n, C) summed log-likelihood of each node's events under each component."""
    return np.stack([np.bincount(v, weights=_bern_ll(pc[c], y), minlength=n)
                     for c in range(pc.shape[0])], axis=1)


def fit_im_gated(L, y, v, n, C=1, restarts=2, iters=25, seed=0):
    """EM for the gated mixture; returns ``(theta (C,K), beta (C,), gate (n,C), train_ll)``."""
    K = L.shape[1]
    if C == 1:
        x = _fit_logistic(L, y)
        pc = _sigmoid(x[0] + L @ x[1:])[None, :]
        return x[1:][None, :], x[:1], np.ones((n, 1)), float(_bern_ll(pc[0], y).sum())
    rng = np.random.default_rng(seed)
    base = _fit_logistic(L, y)
    best = None
    for _ in range(int(restarts)):
        gate = rng.dirichlet(np.ones(C), size=n)
        xs = [base + np.concatenate([[rng.normal(0, 0.5)], rng.normal(0, 0.2, K)]).clip(-20, 20)
              for _ in range(C)]
        xs = [np.concatenate([[x[0]], np.maximum(x[1:], 0.0)]) for x in xs]
        for _ in range(int(iters)):
            xs = [_fit_logistic(L, y, w=gate[v, c], x0=xs[c]) for c in range(C)]
            pc = np.stack([_sigmoid(x[0] + L @ x[1:]) for x in xs])
            pi = np.maximum(gate.mean(axis=0), 1e-6)
            logpost = np.log(pi)[None, :] + _node_loglik(pc, y, v, n)
            mx = logpost.max(axis=1, keepdims=True)
            gate = np.exp(logpost - mx)
            gate /= gate.sum(axis=1, keepdims=True)
        ll = float(np.sum(mx[:, 0] + np.log(np.exp(logpost - mx).sum(axis=1))))
        if best is None or ll > best[3]:
            best = (np.stack([x[1:] for x in xs]), np.asarray([x[0] for x in xs]), gate, ll)
    return best


def _im_val_ll(theta, beta, gate, Lv, yv, vv):
    pc = _sigmoid(beta[:, None] + theta @ Lv.T)  # (C, N)
    p = np.einsum("cn,nc->n", pc, gate[vv])
    return float(_bern_ll(p, yv).sum())


# =========================================================================
# RPP: additive effects + fitted noise family
# =========================================================================


def _effect_mean(effect, feats):
    if hasattr(effect, "_ms"):
        return effect._ms(feats)[0]
    return np.asarray(effect.mean(feats), dtype=np.float64)


class SumDriftEffect(dsl.DriftEffect):
    """``delta = b + sum_i theta_i mu_i`` with uniform or Gaussian noise."""

    def __init__(self, experts, theta, b, family, width, name="mix"):
        self.experts = list(experts)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.b = float(b)
        self.family = str(family)
        self.width = float(width)
        self.name = "%s[K=%d,%s]" % (name, len(self.experts), self.family)
        self.param_names = tuple("theta%d" % i for i in range(len(self.experts))) + ("b", "width")
        lines = ["# expert %d (weight %.2f): %s" % (i, w, _summary(e))
                 for i, (e, w) in enumerate(zip(self.experts, self.theta))]
        lines.append("# intercept %.4f, noise %s width %.4f" % (self.b, self.family, self.width))
        self.source = "\n".join(lines)

    def mean(self, feats):
        mu = np.full(np.shape(feats.profit), self.b, dtype=np.float64)
        for w, e in zip(self.theta, self.experts):
            if w:
                mu = mu + w * _effect_mean(e, feats)
        return mu

    def logpdf(self, delta, feats):
        return _noise_logpdf(np.asarray(delta, dtype=np.float64) - self.mean(feats),
                             self.family, self.width)

    def sample(self, feats, rng):
        mu = self.mean(feats)
        if self.family == "gaussian":
            return mu + self.width * rng.standard_normal(mu.shape)
        return mu + self.width * rng.uniform(-1.0, 1.0, mu.shape)


def _noise_logpdf(r, family, width):
    width = max(float(width), 1e-6)
    if family == "gaussian":
        return -0.5 * (r / width) ** 2 - math.log(width * math.sqrt(2.0 * math.pi))
    excess = np.maximum(np.abs(r) - width, 0.0)  # soft tail as dsl.UniformDrift
    return -math.log(2.0 * width) - 0.5 * (excess / (0.1 * width)) ** 2


def _fit_noise(r):
    """Best of a Gaussian and a (soft-tailed) uniform for residuals ``r``."""
    sd = max(float(np.sqrt(np.mean(r ** 2))), 1e-6)
    out = [("gaussian", sd, float(_noise_logpdf(r, "gaussian", sd).sum()))]
    hi = max(float(np.max(np.abs(r))), 1e-5) * 1.5
    res = minimize_scalar(lambda h: -float(_noise_logpdf(r, "uniform", h).sum()),
                          bounds=(1e-6, hi), method="bounded")
    out.append(("uniform", float(res.x), -float(res.fun)))
    return max(out, key=lambda t: t[2])


def rpp_expert_events(effect, episodes, env, n_regions=4, seed=0):
    """Expert mean and observed change on every scored POI-day (as fit.loglik_drift)."""
    regions = fit._drift_regions(env, n_regions=n_regions, seed=seed)
    kernel = dsl.spill_kernel(env)
    initial = np.asarray(getattr(env, "initial_profit", np.zeros(env.n)), dtype=np.float64)
    mus, ds = [], []
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
            mus.append(np.asarray(_effect_mean(effect, feats), dtype=np.float64)[0][mask])
            ds.append((after - before)[mask])
    if not mus:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(mus), np.concatenate(ds)


def fit_rpp_sum(MU, d):
    """Least squares ``b + MU theta`` (theta >= 0), then the noise family on the residuals."""
    K = MU.shape[1]
    A = np.concatenate([np.ones((MU.shape[0], 1)), MU], axis=1)
    res = lsq_linear(A, d, bounds=([-np.inf] + [0.0] * K, [np.inf] * (K + 1)))
    b, theta = float(res.x[0]), np.asarray(res.x[1:])
    family, width, ll = _fit_noise(d - A @ res.x)
    return theta, b, family, width, ll


def _rpp_val_ll(theta, b, family, width, MUv, dv):
    return float(_noise_logpdf(dv - (b + MUv @ theta), family, width).sum())


# =========================================================================
# Fitting a mixture over a pool of experts (shared by IM and RPP)
# =========================================================================


def _group_key(law):
    return (round(float(getattr(law, "rho", 0.0) or 0.0), 4), bool(getattr(law, "two_hop", False)))


class _Pool:
    """Cached expert features on the training and validation logs."""

    def __init__(self, domain, experts, train, val, obj, restarts=2, iters=25):
        self.domain, self.experts, self.train, self.val, self.obj = domain, list(experts), train, val, obj
        self.restarts, self.iters = int(restarts), int(iters)
        if domain == "im":
            tr = [im_expert_events(e, train, obj) for e in self.experts]
            va = [im_expert_events(e, val, obj) for e in self.experts]
            self.Ltr = np.stack([t[0] for t in tr], axis=1)
            self.ytr, self.vtr = tr[0][1], tr[0][2]
            self.Lva = np.stack([t[0] for t in va], axis=1)
            self.yva, self.vva = va[0][1], va[0][2]
            self.n = int(obj.indeg.size)
        else:
            effects = [getattr(e, "effect", e) for e in self.experts]
            tr = [rpp_expert_events(e, train, obj) for e in effects]
            va = [rpp_expert_events(e, val, obj) for e in effects]
            self.Ltr = np.stack([t[0] for t in tr], axis=1)
            self.dtr = tr[0][1]
            self.Lva = np.stack([t[0] for t in va], axis=1)
            self.dva = va[0][1]

    def fit(self, cols, C=1, rows=None, seed=0):
        """Fit on the experts ``cols`` (training rows ``rows``); returns (params, val_ll)."""
        cols = np.asarray(cols, dtype=np.int64)
        if self.domain == "im":
            L, y, v = self.Ltr[:, cols], self.ytr, self.vtr
            if rows is not None:
                L, y, v = L[rows], y[rows], v[rows]
            theta, beta, gate, _ = fit_im_gated(L, y, v, self.n, C=C, restarts=self.restarts,
                                                 iters=self.iters, seed=seed)
            return (theta, beta, gate), _im_val_ll(theta, beta, gate, self.Lva[:, cols], self.yva, self.vva)
        MU, d = self.Ltr[:, cols], self.dtr
        if rows is not None:
            MU, d = MU[rows], d[rows]
        theta, b, family, width, _ = fit_rpp_sum(MU, d)
        return (theta, b, family, width), _rpp_val_ll(theta, b, family, width, self.Lva[:, cols], self.dva)

    def make(self, cols, params, name="mix"):
        experts = [self.experts[i] for i in cols]
        if self.domain == "im":
            theta, beta, gate = params
            law = GatedMixLaw(experts, theta, beta, gate, name=name)
            law.loglik = float(fit.loglik(law, self.train, self.obj.indeg, self.obj))
            return law
        theta, b, family, width = params
        effect = SumDriftEffect([getattr(e, "effect", e) for e in experts], theta, b, family, width,
                                name=name)
        program = dsl.DriftProgram(effect)
        program.source = effect.source
        program.params = {}
        program.loglik = float(fit.loglik_drift(program, self.train, self.obj))
        return program

    def prune(self, cols, C=1, delta=1.0, seed=0):
        """One backward pass on validation log-likelihood, lightest experts first.

        An expert is dropped when refitting without it loses less than
        ``delta``; O(K) refits at the chosen number of components.
        """
        cols = list(cols)
        params, best = self.fit(cols, C=C, seed=seed)
        theta = np.atleast_2d(params[0])
        order = [cols[i] for i in np.argsort(theta.max(axis=0))]
        for j in order:
            if len(cols) <= 1:
                break
            rest = [c for c in cols if c != j]
            p2, ll = self.fit(rest, C=C, seed=seed)
            if ll >= best - delta:
                cols, params, best = rest, p2, max(best, ll)
        return cols, params, best


class MixContext:
    """The fitted mixture of one expert group, able to produce alternatives."""

    def __init__(self, pool, cols, C, params, val_ll, law):
        self.pool, self.cols, self.C, self.params, self.val_ll, self.law = pool, cols, C, params, val_ll, law
        law.ll_val = val_ll

    def alternatives(self, eps=5.0, n_boot=6, seed=0):
        """Mixtures within ``eps`` held-out log-likelihood: bootstrap refits and leave-one-out."""
        pool, out = self.pool, []
        rng = np.random.default_rng(seed)
        ep_rows = _episode_rows(pool)
        tol = float(eps)  # total held-out log-likelihood, for both domains
        for b in range(int(n_boot)):
            pick = rng.integers(len(ep_rows), size=len(ep_rows))
            rows = np.concatenate([ep_rows[i] for i in pick]) if len(ep_rows) else None
            params, ll = pool.fit(self.cols, C=self.C, rows=rows, seed=seed + 1 + b)
            if ll >= self.val_ll - tol:
                law = pool.make(self.cols, params, name="mix-boot%d" % b)
                law.ll_val = ll
                out.append(law)
        if len(self.cols) > 1:
            for j in range(len(self.cols)):
                rest = self.cols[:j] + self.cols[j + 1:]
                params, ll = pool.fit(rest, C=self.C, seed=seed)
                if ll >= self.val_ll - tol:
                    law = pool.make(rest, params, name="mix-drop%d" % j)
                    law.ll_val = ll
                    out.append(law)
        return out


def _episode_rows(pool):
    """Training-event row indices of each logged episode (for the episode bootstrap)."""
    rows, start = [], 0
    for episode in pool.train:
        if pool.domain == "im":
            count = 0
            for round_ in episode:
                count += int(np.asarray(fit._field(round_, "eligible"), dtype=bool).sum())
        else:
            count = 0
            for round_ in episode:
                before = np.asarray(fit._field(round_, "profit_before"), dtype=np.float64)
                after = np.asarray(fit._field(round_, "profit_after"), dtype=np.float64)
                va = np.asarray(fit._field(round_, "visited_after"), dtype=bool)
                count += int(((~va) & (before > 0.0) & (after > pool.obj.profit_min + 1e-9)).sum())
        rows.append(np.arange(start, start + count))
        start += count
    return rows


def build_mixtures(domain, candidates, train, val, obj, max_experts=12, components=(1, 2, 3),
                   prune_delta=1.0, seed=0):
    """Fit, prune and size the mixtures over ``candidates``.

    Returns ``(contexts, report)``; ``contexts[0]`` holds the best mixture by
    held-out likelihood.  IM fits one mixture per exposure-memory group.
    """
    domain = str(domain).lower()
    ranked = sorted(candidates, key=lambda l: -float(getattr(l, "loglik", -np.inf) or -np.inf))
    if domain == "im":
        groups: Dict[Tuple[float, bool], List[Any]] = {}
        for law in ranked:
            groups.setdefault(_group_key(law), []).append(law)
        group_list = [g[:max_experts] for g in groups.values()]
    else:
        effects = [l for l in ranked if getattr(getattr(l, "effect", None), "mean", None) is not None]
        group_list = [effects[:max_experts]]
    contexts, report = [], {"groups": []}
    for experts in group_list:
        if len(experts) < 2:
            continue
        try:
            pool = _Pool(domain, experts, train, val, obj)
            allc = list(range(len(experts)))
            best = None  # fewest components unless more gain over ``prune_delta``
            for C in (components if domain == "im" else (1,)):
                _, ll = pool.fit(allc, C=C, seed=seed)
                if best is None or ll > best[1] + prune_delta:
                    best = (C, ll)
            C = best[0]
            cols, params, ll = pool.prune(allc, C=C, delta=prune_delta, seed=seed)
            law = pool.make(cols, params)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            report["groups"].append({"error": "%s: %s" % (type(e).__name__, e)})
            continue
        contexts.append(MixContext(pool, cols, C, params, ll, law))
        report["groups"].append({"experts": len(experts), "kept": len(cols), "components": C,
                                 "val_ll": round(ll, 2), "train_ll": round(law.loglik, 2)})
    contexts.sort(key=lambda c: -c.val_ll)
    return contexts, report


def quick_mixture(domain, candidates, train, val, obj, max_experts=12, seed=0):
    """Cheap mixture for the synthesis loop: best group, C in {1, 2}, no pruning."""
    domain = str(domain).lower()
    ranked = sorted(candidates, key=lambda l: -float(getattr(l, "loglik", -np.inf) or -np.inf))
    if domain == "im":
        groups: Dict[Tuple[float, bool], List[Any]] = {}
        for law in ranked:
            groups.setdefault(_group_key(law), []).append(law)
        group_list = [g[:max_experts] for g in groups.values() if len(g) >= 1]
    else:
        group_list = [[l for l in ranked if getattr(getattr(l, "effect", None), "mean", None)
                       is not None][:max_experts]]
    best = None
    for experts in group_list:
        if not experts:
            continue
        try:
            pool = _Pool(domain, experts, train, val, obj, restarts=1, iters=12)
            cols = list(range(len(experts)))
            for C in ((1, 2) if domain == "im" else (1,)):
                params, ll = pool.fit(cols, C=C, seed=seed)
                if best is None or ll > best[0]:
                    best = (ll, pool, cols, params)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    if best is None:
        return None
    ll, pool, cols, params = best
    law = pool.make(cols, params)
    law.ll_val = ll
    return law


def community_residuals(law, episodes, graph, top=8):
    """IM: observed vs predicted activation rate per community (node-type gaps)."""
    from strata.laws.llm_propose import _im_events, _table

    _, _, c, _, y = _im_events(episodes, graph)
    p = np.concatenate([np.asarray(q) for q in _im_predictions(law, episodes, graph)])
    rows = []
    for cc in np.unique(c):
        sel = c == cc
        if sel.sum() >= 5:
            rows.append((abs(y[sel].mean() - p[sel].mean()) * math.sqrt(sel.sum()), int(cc),
                         int(sel.sum()), "%.4f" % y[sel].mean(), "%.4f" % p[sel].mean()))
    rows.sort(key=lambda r: -r[0])
    return _table(["community", "events", "observed rate", "predicted"], [r[1:] for r in rows[:top]])


# =========================================================================
# Prompt material
# =========================================================================


def disagreement_table(domain, base, alts, episodes, obj, top=10):
    """Where decision-changing alternatives predict differently from ``base`` on the logs."""
    if domain == "im":
        from strata.laws.llm_propose import _im_events, _table

        a, d, c, _, y = _im_events(episodes, obj)
        q = np.quantile(d, [0.25, 0.5, 0.75]) if d.size else [1, 2, 3]
        preds = [np.concatenate([np.asarray(p) for p in _im_predictions(m, episodes, obj)])
                 for m in [base] + list(alts)]
        rows = []
        for av in range(7):
            for b in range(4):
                for cc in np.unique(c):
                    sel = (np.minimum(a, 6) == av) & (np.digitize(d, q) == b) & (c == cc)
                    if sel.sum() < 5:
                        continue
                    pb = preds[0][sel].mean()
                    gap = max(abs(p[sel].mean() - pb) for p in preds[1:])
                    rows.append((gap * math.sqrt(sel.sum()), av, "q%d" % (b + 1), int(cc), int(sel.sum()),
                                 "%.3f" % y[sel].mean(), "%.3f" % pb,
                                 " / ".join("%.3f" % p[sel].mean() for p in preds[1:])))
        rows.sort(key=lambda r: -r[0])
        return _table(["a", "indeg quartile", "community", "events", "observed", "best model",
                       "alternatives"], [r[1:] for r in rows[:top]])
    from strata.laws.llm_propose import _rpp_obs, _table

    datas = [_rpp_obs(episodes, obj, law=m) for m in [base] + list(alts)]
    rows = []
    for key in ("region", "t", "crowd"):
        k = np.asarray(datas[0][key])
        k = np.minimum(k, 5) if key == "crowd" else k
        for val in np.unique(k):
            sel = k == val
            if sel.sum() < 5:
                continue
            pb = datas[0]["pred"][sel].mean()
            gap = max(abs(dd["pred"][sel].mean() - pb) for dd in datas[1:])
            rows.append((gap * math.sqrt(sel.sum()), "%s=%s" % (key, val), int(sel.sum()),
                         "%.4f" % datas[0]["delta"][sel].mean(), "%.4f" % pb,
                         " / ".join("%.4f" % dd["pred"][sel].mean() for dd in datas[1:])))
    rows.sort(key=lambda r: -r[0])
    return _table(["group", "obs", "observed mean change", "best model", "alternatives"],
                  [r[1:] for r in rows[:top]])


def _im_predictions(law, episodes, graph):
    indeg = np.asarray(graph.indeg, dtype=np.float64)
    n = indeg.size
    community = np.asarray(graph.community) if graph.community is not None else np.zeros(n, int)
    for rounds in episodes:
        E = np.zeros(n, dtype=np.float32)
        for r in rounds:
            el = np.asarray(fit._field(r, "eligible"), dtype=bool).reshape(-1)
            feats, E, _ = fit._features_for_round(law, r, E, indeg, community, graph)
            yield np.asarray(law.prob(feats))[0][el]
            yy = np.asarray(fit._field(r, "y"), dtype=bool).reshape(-1) & el
            E = np.where(yy, 0.0, E)
            E[np.asarray(fit._field(r, "after")).reshape(-1) != STATUS_INACTIVE] = 0.0
