"""L1 of STRATA, main path: the LLM writes law programs from logged statistics.

Loop (WorldCoder / REx style):

1. ``statistics_tables`` summarises the behaviour logs (IM: activation rate by
   active in-neighbours, in-degree, community and exposure history; RPP: mean
   and spread of profit changes by region, day, distance to the POIs visited
   that day, profit level and crowding).
2. The LLM answers with Python classes that follow a template
   (``IM_TEMPLATE`` / ``RPP_TEMPLATE``): parameters with bounds plus
   ``prob(f)`` (IM) or ``mean(f)``/``scale(f)`` (RPP).
3. Each class is loaded in a sandboxed child process
   (:mod:`strata.llm.synthesis`), wrapped so it satisfies the same interfaces as
   the DSL programs (``LLMLaw`` for IM, ``LLMDriftEffect`` inside a
   ``DriftProgram`` for RPP), numerically checked, and its parameters are fitted
   within the declared bounds by maximum likelihood. Errors go back to the LLM
   once.
4. The pool is refined: a bandit score (held-out log-likelihood plus an
   exploration bonus that decays with the number of refinements) picks the
   program to improve; the LLM receives its code and its ``residual_tables``.
   The loop stops after ``patience`` rounds without a held-out gain larger than
   ``delta`` or when the call budget is spent.

``propose`` returns the candidate laws for
:func:`strata.laws.hypotheses.build_hypothesis_set`: ``source="llm"`` (only LLM
programs), ``"dsl"`` (the enumerated structures, the safety net and ablation)
or ``"llm+dsl"`` (both; the default of the STRATA agent).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from strata.core.types import STATUS_INACTIVE
from strata.laws import dsl
from strata.laws import enumerate as en
from strata.laws import fit
from strata.llm.synthesis import SynthesisError, extract_code_blocks, load_definitions

PROB_EPS = 1e-9

IM_ENV_TEXT = (
    "Influence maximisation on a directed graph over T rounds. Each round the agent seeds "
    "a set of nodes; every node that is still inactive and was not seeded becomes active "
    "with some probability that depends on its incoming neighbours that are active this "
    "round (the previous round's newly active nodes plus this round's seeds). Active nodes "
    "become removed after one round. The activation law is unknown."
)
RPP_ENV_TEXT = (
    "Route planning with drifting profits. Each day the agent visits an ordered set of "
    "POIs and collects their current profit. Visited POIs are removed. The profits of the "
    "remaining POIs change every day by a random amount (floored at profit_min); the law "
    "of these changes is unknown and may depend on space, time and the agent's visits."
)

IM_TEMPLATE = '''```python
class Law:
    """One-line description of the mechanism."""
    params = {"w": 0.1}            # initial values of the free parameters
    bounds = {"w": (1e-4, 1.0)}    # box bounds used for maximum-likelihood fitting
    rho = 0.0                      # exposure memory: f.E = a + rho * E_previous_round

    def prob(self, f):
        # f.a: (M, n) number of active in-neighbours this round
        # f.E: (M, n) accumulated exposure (equals f.a when rho == 0)
        # f.eligible: (M, n) bool, f.indeg: (n,), f.community: (n,) int ids
        # return the (M, n) activation probability in [0, 1]
        w = self.params["w"]
        return 1.0 - np.power(1.0 - w, f.E)
```'''

RPP_TEMPLATE = '''```python
class Drift:
    """One-line description of the mechanism."""
    family = "uniform"             # "uniform" (half-width = scale) or "gaussian" (std = scale)
    params = {"width": 0.05}
    bounds = {"width": (1e-4, 0.5)}

    def mean(self, f):
        # f.profit: (M, n) profit before the change, f.visited: (M, n) bool,
        # f.t: (M,) 1-based day, f.distance: (M, n) distance to the POI visited today (0 if none),
        # f.region: (n,) spatial region id, f.initial: (M, n) initial profit,
        # f.visited_now: (M, n) bool visited today, f.kernel: (n, n) spatial proximity (rows = source)
        return np.zeros_like(f.profit)

    def scale(self, f):
        return np.full_like(f.profit, self.params["width"])
```'''


# =========================================================================
# Wrappers: LLM classes -> Law / DriftProgram
# =========================================================================


def _check_bounds(obj):
    params = dict(getattr(obj, "params", {}) or {})
    bounds = dict(getattr(obj, "bounds", {}) or {})
    for k, v in params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise SynthesisError("parameter %r must be a number" % k)
        lo, hi = bounds.get(k, (None, None))
        if lo is None or hi is None or not lo < hi:
            raise SynthesisError("parameter %r needs bounds (lo, hi) with lo < hi" % k)
    return params, bounds


class LLMLaw:
    """An LLM-written IM law that satisfies the Law protocol."""

    two_hop = False

    def __init__(self, obj, source="", name=None):
        self.obj = obj
        self.source = source
        self.params, self.bounds = _check_bounds(obj)
        self.rho = float(getattr(obj, "rho", 0.0) or 0.0)
        self.name = name or "llm:%s" % type(obj).__name__
        self.loglik = None

    def accumulate(self, E, a):
        return a + self.rho * E if self.rho else a

    def with_params(self, params):
        clone = LLMLaw(self.obj, self.source, self.name)
        clone.params = dict(params)
        return clone

    def prob(self, feats):
        self.obj.params = dict(self.params)
        p = np.asarray(self.obj.prob(feats), dtype=np.float64)
        p = np.broadcast_to(p, np.shape(feats.a))
        return np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0)

    def n_params(self):
        return len(self.params)

    def __repr__(self):
        return "<%s %s>" % (self.name, {k: round(v, 4) for k, v in self.params.items()})


class LLMDriftEffect(dsl.DriftEffect):
    """An LLM-written RPP drift (mean + scale of a uniform or Gaussian change)."""

    def __init__(self, obj, source="", name=None):
        self.obj = obj
        self.source = source
        self.params, self.bounds = _check_bounds(obj)
        self.family = str(getattr(obj, "family", "gaussian")).lower()
        if self.family not in ("gaussian", "uniform"):
            raise SynthesisError("family must be 'gaussian' or 'uniform'")
        self.name = name or "llm:%s" % type(obj).__name__
        self.param_names = tuple(self.params)

    def with_params(self, params):
        clone = LLMDriftEffect(self.obj, self.source, self.name)
        clone.params = dict(params)
        return clone

    def _ms(self, feats):
        self.obj.params = dict(self.params)
        shape = np.shape(feats.profit)
        mu = np.broadcast_to(np.asarray(self.obj.mean(feats), dtype=np.float64), shape)
        s = np.broadcast_to(np.asarray(self.obj.scale(feats), dtype=np.float64), shape)
        return np.nan_to_num(mu), np.maximum(np.nan_to_num(s, nan=1e-3), 1e-6)

    def mean(self, feats):
        return self._ms(feats)[0]

    def logpdf(self, delta, feats):
        mu, s = self._ms(feats)
        d = np.asarray(delta, dtype=np.float64) - mu
        if self.family == "gaussian":
            return -0.5 * (d / s) ** 2 - np.log(s * math.sqrt(2.0 * math.pi))
        inside = -np.log(2.0 * s)
        excess = np.maximum(np.abs(d) - s, 0.0)
        # finite soft tail outside the support (as dsl.UniformDrift)
        return inside - 0.5 * (excess / (0.1 * s)) ** 2

    def sample(self, feats, rng):
        mu, s = self._ms(feats)
        if self.family == "gaussian":
            return mu + s * rng.standard_normal(mu.shape)
        return mu + s * rng.uniform(-1.0, 1.0, mu.shape)


def wrap_llm_law(obj, domain, source=""):
    """Wrap a loaded LLM class into an IM law or an RPP drift program."""
    if str(domain).lower() == "im":
        return LLMLaw(obj, source)
    effect = LLMDriftEffect(obj, source)
    program = dsl.DriftProgram(effect)
    program.source = source
    program.params = effect.params
    return program


# =========================================================================
# Fitting
# =========================================================================


def _loglik(domain, law, episodes, obj):
    if domain == "im":
        return fit.loglik(law, episodes, obj.indeg, obj)
    return fit.loglik_drift(law, episodes, obj)


def _with(domain, law, params):
    if domain == "im":
        return law.with_params(params)
    effect = law.effect.with_params(params)
    program = dsl.DriftProgram(effect)
    program.source = law.source if hasattr(law, "source") else effect.source
    program.params = effect.params
    return program


def fit_llm_law(domain, law, episodes, obj, n_random=8, seed=0):
    """Maximum likelihood of the declared parameters within their bounds.

    Derivative-free (random starts + bounded Powell), because LLM laws are
    often piecewise constant in their parameters (thresholds, gates).
    """
    params = law.params if domain == "im" else law.effect.params
    bounds = law.bounds if domain == "im" else law.effect.bounds
    keys = list(params)
    if not keys:
        out = _with(domain, law, params)
        out.loglik = float(_loglik(domain, out, episodes, obj))
        return out
    lo = np.asarray([bounds[k][0] for k in keys], dtype=np.float64)
    hi = np.asarray([bounds[k][1] for k in keys], dtype=np.float64)

    def value(x):
        p = dict(zip(keys, np.clip(x, lo, hi).tolist()))
        ll = _loglik(domain, _with(domain, law, p), episodes, obj)
        return -ll if np.isfinite(ll) else 1e30

    rng = np.random.default_rng(seed)
    starts = [np.clip(np.asarray([params[k] for k in keys], dtype=np.float64), lo, hi)]
    starts += [lo + (hi - lo) * rng.random(len(keys)) for _ in range(n_random)]
    scored = sorted(((value(x), x) for x in starts), key=lambda t: t[0])
    best_v, best_x = scored[0]
    try:
        res = minimize(value, best_x, method="Powell", bounds=list(zip(lo, hi)),
                       options={"maxiter": 200, "xtol": 1e-3, "ftol": 1e-4})
        if res.fun < best_v:
            best_v, best_x = float(res.fun), np.asarray(res.x)
    except (ValueError, FloatingPointError):
        pass
    out = _with(domain, law, dict(zip(keys, np.clip(best_x, lo, hi).tolist())))
    out.loglik = -float(best_v)
    return out


# =========================================================================
# Statistics and residuals for the prompt
# =========================================================================


def _field(round_, key):
    return round_[key] if isinstance(round_, dict) else getattr(round_, key)


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _im_events(episodes, graph):
    """Per eligible (node, round): a, indeg, community, exposure count, y."""
    indeg = np.asarray(graph.indeg)
    comm = np.asarray(graph.community)
    A, D, C, X, Y = [], [], [], [], []
    for rounds in episodes:
        exposed = np.zeros(indeg.size, dtype=np.int64)
        for r in rounds:
            a = np.asarray(_field(r, "a")).reshape(-1)
            el = np.asarray(_field(r, "eligible"), dtype=bool).reshape(-1)
            y = np.asarray(_field(r, "y"), dtype=bool).reshape(-1)
            exposed = exposed + (a > 0)
            idx = np.flatnonzero(el)
            A.append(a[idx]); D.append(indeg[idx]); C.append(comm[idx])
            X.append(exposed[idx]); Y.append(y[idx])
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
    return cat(A, np.float64), cat(D, np.float64), cat(C, np.int64), cat(X, np.int64), cat(Y, bool)


def _rate_rows(keys, y, labels):
    rows = []
    for lab in labels:
        sel = keys == lab if not callable(lab) else lab(keys)
        n = int(sel.sum())
        if n:
            rows.append((lab if not callable(lab) else lab.__doc__, n, "%.4f" % y[sel].mean()))
    return rows


def _rpp_obs(episodes, env, n_regions=4, seed=0, law=None):
    """Observed changes (same rows as ``fit.drift_dataset``) plus crowding and,
    when ``law`` is given, the law's predicted mean change of every row."""
    regions = fit._drift_regions(env, n_regions=n_regions, seed=seed)
    kernel = dsl.spill_kernel(env)
    near = np.asarray(env.dist) <= float(getattr(env, "max_dist", 1.0)) / 4.0
    np.fill_diagonal(near, False)
    initial = np.asarray(getattr(env, "initial_profit", np.zeros(env.n)), dtype=np.float64)
    cols = {k: [] for k in ("delta", "profit", "t", "distance", "region", "crowd", "pred")}
    for episode in episodes:
        visits = np.zeros(env.n, dtype=np.int64)
        for round_ in episode:
            before = np.asarray(_field(round_, "profit_before"), dtype=np.float64)
            after = np.asarray(_field(round_, "profit_after"), dtype=np.float64)
            vb = np.asarray(_field(round_, "visited_before"), dtype=bool)
            va = np.asarray(_field(round_, "visited_after"), dtype=bool)
            visited_now = va & ~vb
            visits = visits + visited_now
            mask = (~va) & (before > 0.0) & (after > env.profit_min + 1e-9)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            dist = fit._distance_to_action(env, _field(round_, "action"))
            cols["delta"].append((after - before)[idx])
            cols["profit"].append(before[idx])
            cols["t"].append(np.full(idx.size, int(_field(round_, "t"))))
            cols["distance"].append(dist[idx])
            cols["region"].append(regions[idx])
            cols["crowd"].append((near[:, idx] * visits[:, None]).sum(axis=0))
            if law is not None:
                feats = dsl.DriftFeatures(
                    profit=before[None, :], visited=va[None, :],
                    t=np.asarray([int(_field(round_, "t"))]), distance=dist[None, :],
                    region=regions, initial=initial[None, :],
                    visited_now=visited_now[None, :], kernel=kernel)
                effect = getattr(law, "effect", law)
                cols["pred"].append(np.asarray(effect.mean(feats))[0][idx])
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    if law is None:
        out.pop("pred")
    return out


def statistics_tables(domain, episodes, obj, n_regions=4):
    """Compact markdown tables of the logged transitions for the prompt."""
    if domain == "im":
        a, d, c, x, y = _im_events(episodes, obj)
        a_lab = np.minimum(a, 6).astype(int)
        parts = ["Activation rate by active in-neighbours a (a>=6 pooled):",
                 _table(["a", "events", "rate"], _rate_rows(a_lab, y, range(7)))]
        q = np.quantile(d, [0.25, 0.5, 0.75]) if d.size else [1, 2, 3]
        d_lab = np.digitize(d, q)
        rows = []
        for b in range(4):
            for av in (1, 2, 3):
                sel = (d_lab == b) & (np.minimum(a, 3) == av)
                if sel.sum():
                    rows.append(("q%d" % (b + 1), av if av < 3 else "3+", int(sel.sum()),
                                 "%.4f" % y[sel].mean()))
        parts += ["Activation rate by in-degree quartile (cuts %s) and a:" % np.round(q, 1),
                  _table(["indeg quartile", "a", "events", "rate"], rows)]
        top = [int(k) for k, _ in sorted(zip(*np.unique(c, return_counts=True)),
                                         key=lambda t: -t[1])[:5]]
        rows = [(k, int(((c == k) & (a > 0)).sum()), "%.4f" % y[(c == k) & (a > 0)].mean())
                for k in top if ((c == k) & (a > 0)).any()]
        parts += ["Activation rate among exposed nodes (a>0) by community (5 largest):",
                  _table(["community", "events", "rate"], rows)]
        x_lab = np.minimum(x, 5)
        rows = [(v, int(((x_lab == v) & (a > 0)).sum()), "%.4f" % y[(x_lab == v) & (a > 0)].mean())
                for v in range(1, 6) if ((x_lab == v) & (a > 0)).any()]
        parts += ["Activation rate among exposed nodes by number of rounds exposed so far (5+ pooled):",
                  _table(["rounds exposed", "events", "rate"], rows)]
        return "\n\n".join(parts)

    data = _rpp_obs(episodes, obj, n_regions=n_regions)
    delta = np.asarray(data["delta"])
    parts = []

    def group(name, key, labels=None):
        key = np.asarray(key)
        labels = np.unique(key) if labels is None else labels
        rows = []
        for v in labels:
            sel = key == v
            if sel.sum():
                rows.append((v, int(sel.sum()), "%.4f" % delta[sel].mean(), "%.4f" % delta[sel].std()))
        parts.append("Profit change by %s:" % name)
        parts.append(_table([name, "obs", "mean", "std"], rows))

    group("region", data["region"])
    group("day", data["t"])
    dist = np.asarray(data["distance"])
    dq = np.digitize(dist, [1e-9] + list(np.quantile(dist[dist > 0], [0.33, 0.66])) if (dist > 0).any() else [1e-9])
    group("distance bucket to today's visits (0 = no visit)", dq)
    pq = np.digitize(np.asarray(data["profit"]), np.quantile(np.asarray(data["profit"]), [0.25, 0.5, 0.75]))
    group("profit quartile", pq)
    if "crowd" in data:
        group("past visits nearby (5+ pooled)", np.minimum(np.asarray(data["crowd"]), 5))
    return "\n\n".join(parts)


def residual_tables(domain, law, episodes, obj, n_regions=4):
    """Observed vs predicted, sorted by the largest discrepancy."""
    if domain == "im":
        a, d, c, x, y = _im_events(episodes, obj)
        pred = []
        for rounds in episodes:
            E = np.zeros(obj.indeg.size, dtype=np.float32)
            for r in rounds:
                el = np.asarray(_field(r, "eligible"), dtype=bool).reshape(-1)
                feats, E, _ = fit._features_for_round(law, r, E, obj.indeg, obj.community, obj)
                p = np.asarray(law.prob(feats))[0]
                pred.append(p[el])
                yy = np.asarray(_field(r, "y"), dtype=bool).reshape(-1) & el
                E = np.where(yy, 0.0, E)
                after = np.asarray(_field(r, "after")).reshape(-1)
                E[after != STATUS_INACTIVE] = 0.0
        p = np.concatenate(pred) if pred else np.zeros(0)
        q = np.quantile(d, [0.25, 0.5, 0.75]) if d.size else [1, 2, 3]
        rows = []
        for av in range(7):
            for b in range(4):
                sel = (np.minimum(a, 6) == av) & (np.digitize(d, q) == b)
                if sel.sum() >= 5:
                    rows.append((av, "q%d" % (b + 1), int(sel.sum()), y[sel].mean(), p[sel].mean()))
        rows.sort(key=lambda r: -abs(r[3] - r[4]) * math.sqrt(r[2]))
        rows = [(r[0], r[1], r[2], "%.4f" % r[3], "%.4f" % r[4]) for r in rows[:12]]
        return _table(["a", "indeg quartile", "events", "observed rate", "predicted"], rows)

    data = _rpp_obs(episodes, obj, n_regions=n_regions, law=law)
    delta, mu = data["delta"], data["pred"]
    rows = []
    for key in ("region", "t", "crowd"):
        k = np.minimum(np.asarray(data[key]), 5) if key == "crowd" else np.asarray(data[key])
        for v in np.unique(k):
            sel = k == v
            if sel.sum() >= 5:
                rows.append(("%s=%s" % (key, v), int(sel.sum()), delta[sel].mean(), mu[sel].mean()))
    rows.sort(key=lambda r: -abs(r[2] - r[3]) * math.sqrt(r[1]))
    rows = [(r[0], r[1], "%.4f" % r[2], "%.4f" % r[3]) for r in rows[:12]]
    return _table(["group", "obs", "observed mean change", "predicted mean"], rows)


# =========================================================================
# Synthesis loop
# =========================================================================


def _describe(law):
    src = getattr(law, "source", "") or ""
    if src:
        return src
    return "%s (DSL program)" % getattr(law, "name", type(law).__name__)


def _prompt(domain, stats, pool, k, focus=None, residuals=None, errors=None):
    env_text = IM_ENV_TEXT if domain == "im" else RPP_ENV_TEXT
    template = IM_TEMPLATE if domain == "im" else RPP_TEMPLATE
    cls = "Law" if domain == "im" else "Drift"
    lines = [
        "You are reverse-engineering the hidden stochastic law of an environment from logged data.",
        env_text, "", "Statistics of the logged transitions:", stats, "",
    ]
    if pool:
        lines.append("Current hypotheses with their held-out log-likelihood (higher is better):")
        for item in pool[:5]:
            lines.append("- held-out LL %.1f:\n%s" % (item["ll_val"], _describe(item["law"])))
        lines.append("")
    if focus is not None:
        lines += ["Improve or extend this program:", _describe(focus["law"]), "",
                  "Where its predictions disagree most with the data:", residuals or "", ""]
    if errors:
        lines += ["Your previous programs failed with these errors; fix them:"] + ["- " + e for e in errors] + [""]
    lines += [
        "Write %d different candidate programs, each as its own ```python block defining one class "
        "named %s that follows exactly this template (numpy is available as np, math as math; no "
        "other imports, no I/O):" % (k, cls),
        template,
        "Only parameters declared in params/bounds are fitted; keep each program small and "
        "vectorised over the (M, n) arrays.",
        "Do not resubmit a program equivalent to a current hypothesis (the same predictions for "
        "some parameter values): each candidate must be a structurally different mechanism, "
        "ideally one that agrees with the data but predicts differently where the data is thin.",
    ]
    return [{"role": "user", "content": "\n".join(lines)}]


def _probe_features(domain, obj):
    rng = np.random.default_rng(0)
    if domain == "im":
        from strata.core.types import Features

        n = obj.indeg.size
        a = rng.integers(0, 4, (2, n)).astype(np.float32)
        return Features(a=a, E=a.copy(), a2=None, eligible=np.ones((2, n), bool),
                        indeg=obj.indeg, community=obj.community)
    n = obj.n
    return dsl.DriftFeatures(
        profit=rng.uniform(0.3, 1.0, (2, n)), visited=np.zeros((2, n), bool),
        t=np.asarray([1, 2]), distance=rng.uniform(0, 1, (2, n)),
        region=np.zeros(n, dtype=np.int64), initial=rng.uniform(0.5, 1.0, (2, n)),
        visited_now=np.zeros((2, n), bool), kernel=dsl.spill_kernel(obj))


def _candidates_from_answer(domain, answer, obj):
    cls = "Law" if domain == "im" else "Drift"
    ok, errors = [], []
    feats = _probe_features(domain, obj)
    for code in extract_code_blocks(answer):
        try:
            ns = load_definitions(code, [cls])
            law = wrap_llm_law(ns[cls](), domain, source=code)
            if domain == "im":
                p = law.prob(feats)
                if p.shape != feats.a.shape or not np.all(np.isfinite(p)):
                    raise SynthesisError("prob must return a finite (M, n) array")
            else:
                lp = law.logpdf(np.zeros_like(feats.profit), feats)
                if np.shape(lp) != feats.profit.shape or not np.all(np.isfinite(lp)):
                    raise SynthesisError("mean/scale must return finite (M, n) arrays")
            ok.append(law)
        except Exception as e:  # noqa: BLE001 - every failure is reported to the LLM
            errors.append("%s: %s" % (type(e).__name__, str(e)[:200]))
    return ok, errors


def synthesize_laws(domain, episodes_train, episodes_val, obj, chat, rounds=3, k=3,
                    budget_calls=8, delta=1.0, patience=2, seed_laws=(), seed=0, c_explore=1.0):
    """LLM synthesis of law programs; returns ``(fitted_laws, report)``."""
    domain = str(domain).lower()
    stats = statistics_tables(domain, episodes_train, obj)
    pool: List[Dict[str, Any]] = []
    for law in seed_laws:
        pool.append({"law": law, "ll_val": float(_loglik(domain, law, episodes_val, obj)),
                     "refined": 0, "llm": False})
    pool.sort(key=lambda it: -it["ll_val"])
    report = {"calls": 0, "accepted": 0, "rejected": 0, "errors": [], "rounds": []}
    best = pool[0]["ll_val"] if pool else -np.inf
    stale, sample = 0, 0

    def ask(messages):
        nonlocal sample
        report["calls"] += 1
        sample += 1
        return chat(messages, sample=sample - 1)

    def evaluate(laws):
        added = []
        for law in laws:
            try:
                fitted = fit_llm_law(domain, law, episodes_train, obj, seed=seed)
                ll_val = float(_loglik(domain, fitted, episodes_val, obj))
            except Exception as e:  # noqa: BLE001
                report["errors"].append("fit: %s" % e)
                report["rejected"] += 1
                continue
            if not np.isfinite(ll_val):
                report["rejected"] += 1
                continue
            item = {"law": fitted, "ll_val": ll_val, "refined": 0, "llm": True}
            pool.append(item)
            added.append(item)
            report["accepted"] += 1
        pool.sort(key=lambda it: -it["ll_val"])
        return added

    for r in range(int(rounds) + 1):
        if report["calls"] >= budget_calls:
            break
        focus = None
        if r > 0 and pool:
            total = sum(it["refined"] for it in pool) + 1
            scale = max(1.0, abs(pool[0]["ll_val"]) * 0.01)
            scores = [it["ll_val"] + c_explore * scale * math.sqrt(math.log(total + 1) / (1 + it["refined"]))
                      for it in pool]
            focus = pool[int(np.argmax(scores))]
            focus["refined"] += 1
        residuals = residual_tables(domain, focus["law"], episodes_train, obj) if focus else None
        answer = ask(_prompt(domain, stats, pool, k, focus, residuals))
        laws, errors = _candidates_from_answer(domain, answer, obj)
        report["rejected"] += len(errors)
        report["errors"] += errors
        if errors and report["calls"] < budget_calls:
            fixed, errors2 = _candidates_from_answer(
                domain, ask(_prompt(domain, stats, pool, k, focus, residuals, errors)), obj)
            laws += fixed
            report["rejected"] += len(errors2)
            report["errors"] += errors2
        evaluate(laws)
        new_best = pool[0]["ll_val"] if pool else -np.inf
        report["rounds"].append({"round": r, "best_ll_val": new_best, "pool": len(pool)})
        if new_best > best + delta:
            best, stale = new_best, 0
        else:
            stale += 1
            if r > 0 and stale >= patience:
                break
    usage = chat.usage() if hasattr(chat, "usage") else {}
    report.update({"usage": usage})
    return [it["law"] for it in pool if it["llm"]], report


# =========================================================================
# Entry point used by the agent
# =========================================================================


def dsl_candidates(domain, episodes, obj, level=2):
    """The enumerated structures fitted by maximum likelihood (safety net)."""
    if domain == "im":
        return [fit.fit_program(s, episodes, obj.indeg, obj, polish=False)
                for s in en.enumerate_structures(level)]
    return [fit.fit_drift_program(s, episodes, obj) for s in en.enumerate_drift_structures(level)]


def propose(domain, episodes_train, episodes_val, obj, chat=None, source="llm+dsl",
            level=2, **llm_kwargs):
    """Candidate laws for the hypothesis set, plus a report.

    With ``source`` containing ``"llm"`` and ``chat=None`` the call falls back to
    the DSL and records it in the report.
    """
    domain = str(domain).lower()
    source = str(source).lower()
    report: Dict[str, Any] = {"source": source, "fallback": False}
    dsl_laws = dsl_candidates(domain, episodes_train, obj, level) if ("dsl" in source or chat is None) else []
    llm_laws: List[Any] = []
    if "llm" in source:
        if chat is None:
            report["fallback"] = True
        else:
            seeds = sorted(dsl_laws, key=lambda l: -float(getattr(l, "loglik", -np.inf) or -np.inf))[:3]
            llm_laws, rep = synthesize_laws(domain, episodes_train, episodes_val, obj, chat,
                                            seed_laws=seeds, **llm_kwargs)
            report["llm"] = rep
    out = llm_laws + (dsl_laws if ("dsl" in source or not llm_laws) else [])
    report["n_llm"], report["n_dsl"] = len(llm_laws), len(out) - len(llm_laws)
    return out, report
