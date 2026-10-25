"""Maximum-likelihood fitting of L1 structures to logged transitions.

IM
--
``loglik(law, episodes, indeg, graph)`` is a port of
``experiments/kt_law/t2_fit.loglik``: only eligible nodes contribute, the
candidate's own ``rho`` drives the exposure recursion, and probabilities are
clipped to ``[1e-12, 1 - 1e-12]``.  ``fit_program`` first scans the structure's
parameter grid and then polishes the best point with L-BFGS-B.  ``fit_free``
uses weighted pool-adjacent-violators so the fitted table is monotone, and
``fit_product`` optimises the OneLife weights ``theta``.

RPP
---
``fit_drift_program`` fits a continuous drift distribution to the observed
profit changes of POIs that stayed unvisited.  Location covariates (time,
distance to the POI just visited, current profit, spatial region) enter the
mean linearly; ``GaussianDrift``/``UniformDrift`` are fitted in closed form and
``MixtureDrift`` with a small EM.
"""

import numpy as np
from scipy.optimize import minimize

from strata.core.types import STATUS_ACTIVE, STATUS_INACTIVE, Features
from strata.laws import dsl
from strata.laws.dsl import DriftProgram, LawProgram, ProductLaw

EPS = 1e-12


# =========================================================================
# IM: likelihood
# =========================================================================


def _field(round_, key):
    if isinstance(round_, dict):
        return round_[key]
    return getattr(round_, key)


def _infer_n(episodes):
    for rounds in episodes:
        for r in rounds:
            return int(np.asarray(_field(r, "a")).size)
    return 0


def _accumulate(law, E, a):
    """Use the law's own recursion when it has one (LawProgram/ProductLaw)."""
    fn = getattr(law, "accumulate", None)
    if fn is not None:
        return fn(E, a)
    rho = float(getattr(law, "rho", 0.0))
    return a + rho * E if rho else a


def _features_for_round(law, round_, E, indeg, community, graph):
    """Features of one round, optionally recomputing two-hop statistics."""
    a = np.asarray(_field(round_, "a"), dtype=np.float32)
    eligible = np.asarray(_field(round_, "eligible"), dtype=bool)
    two_hop = bool(getattr(law, "two_hop", False))
    a2 = None
    if two_hop and graph is not None:
        before = np.asarray(_field(round_, "before"))
        seeded = np.asarray(_field(round_, "seeded"), dtype=bool)
        act = (before == STATUS_ACTIVE) | seeded
        a = (act.astype(np.float32) @ graph.adj).reshape(-1)
        a2 = (act.astype(np.float32) @ (graph.adj @ graph.adj)).reshape(-1)
    E_now = _accumulate(law, E[None, :], a[None, :])[0]
    feats = Features(
        a=a[None, :],
        E=E_now[None, :],
        a2=None if a2 is None else a2[None, :],
        eligible=eligible[None, :],
        indeg=indeg,
        community=community,
    )
    return feats, E_now, eligible


def loglik(law, episodes, indeg=None, graph=None):
    """Summed transition log-likelihood of ``law`` on ``episodes``."""
    if graph is not None and indeg is None:
        indeg = graph.indeg
    if indeg is None:
        n = _infer_n(episodes)
        indeg = np.ones(n, dtype=np.float64)
    indeg = np.asarray(indeg, dtype=np.float64)
    n = indeg.size
    community = (
        np.asarray(graph.community)
        if graph is not None and graph.community is not None
        else np.zeros(n, dtype=np.int64)
    )

    total = 0.0
    for rounds in episodes:
        E = np.zeros(n, dtype=np.float32)
        for round_ in rounds:
            eligible = np.asarray(_field(round_, "eligible"), dtype=bool)
            if not eligible.any():
                continue
            feats, E, _ = _features_for_round(law, round_, E, indeg, community, graph)
            p_full = law.prob(feats)[0]
            p_full = np.where(eligible, p_full, 0.0)
            p = np.clip(p_full.astype(np.float64), EPS, 1.0 - EPS)
            y = np.asarray(_field(round_, "y"), dtype=bool) & eligible
            total += float(np.sum(np.where(y, np.log(p), np.log(1.0 - p))))
            # Exposure reset convention of t2_fit.loglik.
            E = np.where(y, 0.0, E)
            after = np.asarray(_field(round_, "after"))
            E[after != STATUS_INACTIVE] = 0.0
    return total


# =========================================================================
# IM: fitting
# =========================================================================

_CONTINUOUS_BOUNDS = {
    "w": (1e-6, 1.0),
    "c": (1e-6, 1.0),
    "alpha": (1e-4, 20.0),
    "threshold": (0.5, 5.0),
    "beta": (0.1, 20.0),
    "lo": (-5.0, 5.0),
}


def _continuous_keys(params):
    return [
        k
        for k, v in params.items()
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and k in _CONTINUOUS_BOUNDS
    ]


def _polish(structure, params, episodes, indeg, graph):
    keys = _continuous_keys(params)
    if not keys:
        return params
    x0 = np.asarray([float(params[k]) for k in keys], dtype=np.float64)
    bounds = [_CONTINUOUS_BOUNDS[k] for k in keys]

    def objective(x):
        trial = dict(params)
        for k, v in zip(keys, x):
            trial[k] = float(v)
        law = structure.build(trial)
        return -loglik(law, episodes, indeg=indeg, graph=graph)

    try:
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-10},
        )
    except Exception:
        return params
    if not res.success or not np.isfinite(res.fun):
        return params
    best = dict(params)
    for k, v in zip(keys, res.x):
        best[k] = float(v)
    if -res.fun >= -objective(x0) - 1e-12:
        return best
    return params


def fit_program(structure, episodes, indeg=None, graph=None, grid=None, polish=True):
    """Fit one :class:`Structure`; returns a :class:`LawProgram`.

    The chosen parameters and training likelihood are attached as
    ``law.params``, ``law.structure`` and ``law.loglik``.
    """
    if isinstance(structure, str):
        from strata.laws import enumerate as _enumerate

        structure = _enumerate.structure_by_name(structure)
    if structure.kind == "free":
        return fit_free(structure, episodes, indeg=indeg, graph=graph)
    if grid is None:
        grid = structure.param_grid()

    best_params = None
    best_ll = -np.inf
    for params in grid:
        law = structure.build(params)
        ll = loglik(law, episodes, indeg=indeg, graph=graph)
        if ll > best_ll:
            best_ll, best_params = ll, dict(params)

    if best_params is None:
        raise ValueError("empty parameter grid for %s" % structure.name)

    if polish:
        polished = _polish(structure, best_params, episodes, indeg, graph)
        law = structure.build(polished)
        ll = loglik(law, episodes, indeg=indeg, graph=graph)
        if ll >= best_ll:
            best_params, best_ll = polished, ll

    law = structure.build(best_params)
    law.params = dict(best_params)
    law.structure = structure
    law.loglik = float(best_ll)
    return law


def fit_library(
    structures, episodes_train, episodes_test, indeg=None, graph=None, polish=True
):
    """Fit every structure on train, select the best train LL, score on test.

    Returns ``(law, train_ll, test_ll)``.
    """
    best = None
    for structure in structures:
        law = fit_program(
            structure, episodes_train, indeg=indeg, graph=graph, polish=polish
        )
        ll = loglik(law, episodes_train, indeg=indeg, graph=graph)
        if best is None or ll > best[0]:
            best = (ll, law)
    ll_train, law = best
    ll_test = loglik(law, episodes_test, indeg=indeg, graph=graph)
    law.train_loglik = float(ll_train)
    law.test_loglik = float(ll_test)
    return law, float(ll_train), float(ll_test)


# ---------------------------------------------------------------------- free


def _pav(values, weights):
    """Weighted isotonic regression; returns a non-decreasing array."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = []
    wsum = []
    counts = []
    for v, w in zip(values, weights):
        means.append(float(v))
        wsum.append(float(w))
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1] + 1e-15:
            w_new = wsum[-2] + wsum[-1]
            m_new = (means[-2] * wsum[-2] + means[-1] * wsum[-1]) / max(w_new, EPS)
            c_new = counts[-2] + counts[-1]
            means[-2:] = [m_new]
            wsum[-2:] = [w_new]
            counts[-2:] = [c_new]
    out = []
    for m, c in zip(means, counts):
        out.extend([m] * c)
    return np.asarray(out, dtype=np.float64)


def _bucket_counts(episodes):
    max_a = 0
    for rounds in episodes:
        for r in rounds:
            a = np.asarray(_field(r, "a"))
            if a.size:
                max_a = max(max_a, int(np.rint(a.max())))
    n = np.zeros(max_a + 1, dtype=np.float64)
    y = np.zeros(max_a + 1, dtype=np.float64)
    for rounds in episodes:
        for r in rounds:
            a = np.rint(np.asarray(_field(r, "a"))).astype(np.int64)
            eligible = np.asarray(_field(r, "eligible"), dtype=bool)
            yy = np.asarray(_field(r, "y"), dtype=bool)
            for bucket in range(max_a + 1):
                sel = eligible & (a == bucket)
                n[bucket] += float(sel.sum())
                y[bucket] += float((yy & sel).sum())
    return n, y


def fit_free(structure, episodes=None, indeg=None, graph=None, **kwargs):
    """Fit a monotone ``free`` table with weighted PAV.

    Accepts either ``fit_free(structure, episodes, indeg=...)`` or the shorter
    ``fit_free(episodes, indeg)`` form.
    """
    if not hasattr(structure, "kind"):
        # Called as fit_free(episodes, indeg).
        if episodes is not None and indeg is None:
            indeg = episodes
        episodes = structure
        structure = None
    if episodes is None:
        raise ValueError("fit_free needs episodes")
    n, y = _bucket_counts(episodes)
    rate = np.divide(y, n, out=np.zeros_like(y), where=n > 0)
    table = _pav(rate, np.maximum(n, 1.0))
    # Buckets with no evidence inherit the nearest fitted value.
    known = n > 0
    if known.any():
        idx = np.flatnonzero(known)
        for bucket in range(table.size):
            if not known[bucket]:
                table[bucket] = table[idx[np.argmin(np.abs(idx - bucket))]]
    else:
        table[:] = 0.0
    law = LawProgram(dsl.Free(table))
    law.params = {"table": table.copy()}
    law.structure = structure
    law.loglik = float(loglik(law, episodes, indeg=indeg, graph=graph))
    return law


def fit_product(
    programs, episodes, indeg=None, graph=None, theta0=None, theta_bounds=(1e-4, 50.0)
):
    """Fit the OneLife weights of ``ProductLaw`` by L-BFGS-B."""
    programs = [LawProgram(p) if isinstance(p, str) else p for p in programs]
    m = len(programs)
    if theta0 is None:
        theta0 = np.ones(m, dtype=np.float64)
    bounds = [theta_bounds] * m

    def objective(theta):
        law = ProductLaw(programs, theta)
        return -loglik(law, episodes, indeg=indeg, graph=graph)

    res = minimize(
        objective,
        np.asarray(theta0, dtype=np.float64),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    theta = res.x if res.success and np.isfinite(res.fun) else np.asarray(theta0)
    law = ProductLaw(programs, theta)
    law.loglik = float(loglik(law, episodes, indeg=indeg, graph=graph))
    return law


# =========================================================================
# RPP: drift observations and fitting
# =========================================================================


def _drift_regions(env, n_regions=4, seed=0):
    existing = getattr(env, "regions", None)
    if existing is not None and np.asarray(existing).size == env.n:
        return np.asarray(existing, dtype=np.int64)
    from strata.domains.rpp import points

    regions = points.spatial_clusters(env.loc, n_regions, seed=seed)
    env.regions = regions
    return regions


def _distance_to_action(env, action):
    action = np.asarray(action, dtype=np.int64)
    action = action[action >= 0]
    if action.size == 0:
        return np.zeros(env.n, dtype=np.float64)
    return np.min(env.dist[:, action], axis=1)


def drift_dataset(episodes, env, n_regions=4, seed=0):
    """Flatten observed profit changes plus the covariates needed to model them.

    Returns a dict with one entry per observed ``(round, unvisited POI)``:
    ``delta, profit, initial, t, distance, region, node, visited_now, kernel``.
    Nodes that hit the ``profit_min`` floor are censored (their true change is
    below the threshold); like the reference fit they are excluded from the
    likelihood rather than treated as observed zeros.
    """
    regions = _drift_regions(env, n_regions=n_regions, seed=seed)
    n_regions = int(max(regions.max() + 1, 1)) if regions.size else 1
    kernel = dsl.spill_kernel(env)
    n = env.n
    deltas, profits, initials, times, dists, regs, nodes, vnow = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for episode in episodes:
        for round_ in episode:
            before = np.asarray(_field(round_, "profit_before"), dtype=np.float64)
            after = np.asarray(_field(round_, "profit_after"), dtype=np.float64)
            visited_before = np.asarray(_field(round_, "visited_before"), dtype=bool)
            visited_after = np.asarray(_field(round_, "visited_after"), dtype=bool)
            mask = (~visited_after) & (before > 0.0) & (after > env.profit_min + 1e-9)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            visited_now = visited_after & ~visited_before
            deltas.append((after - before)[idx])
            profits.append(before[idx])
            initials.append(np.asarray(env.initial_profit, np.float64)[idx])
            times.append(np.full(idx.size, int(_field(round_, "t")), np.float64))
            dists.append(_distance_to_action(env, _field(round_, "action"))[idx])
            regs.append(regions[idx])
            nodes.append(idx)
            vnow.append(np.broadcast_to(visited_now[None, :], (idx.size, n)).copy())
    if not deltas:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "delta": empty,
            "profit": empty,
            "initial": empty,
            "t": empty,
            "distance": empty,
            "region": np.zeros(0, dtype=np.int64),
            "node": np.zeros(0, dtype=np.int64),
            "visited_now": np.zeros((0, n), dtype=bool),
            "kernel": kernel,
            "n_regions": n_regions,
            "n": n,
        }
    return {
        "delta": np.concatenate(deltas),
        "profit": np.concatenate(profits),
        "initial": np.concatenate(initials),
        "t": np.concatenate(times),
        "distance": np.concatenate(dists),
        "region": np.concatenate(regs).astype(np.int64),
        "node": np.concatenate(nodes).astype(np.int64),
        "visited_now": np.concatenate(vnow),
        "kernel": kernel,
        "n_regions": n_regions,
        "n": n,
    }


def drift_observations(episodes, env, n_regions=4, seed=0):
    """Flatten observed unvisited-POI profit changes into arrays.

    Returns ``(delta, profit, t, distance, region)``.  Nodes that hit the
    ``profit_min`` floor are censored and therefore excluded.
    """
    data = drift_dataset(episodes, env, n_regions=n_regions, seed=seed)
    return (
        data["delta"],
        data["profit"],
        data["t"],
        data["distance"],
        data["region"],
    )


def _design_matrix(data, covariates, distance_scale, season_period):
    """Design matrix plus a ``(parameter, index)`` description per column."""
    delta = data["delta"]
    n = int(data["n"])
    n_rows = delta.size
    columns = [np.ones(n_rows, dtype=np.float64)]
    spec = [("mu", -1)]
    if "t" in covariates:
        columns.append(data["t"] - 1.0)
        spec.append(("time_coef", -1))
    if "reversion" in covariates:
        columns.append(data["initial"] - data["profit"])
        spec.append(("initial_coef", -1))
    if "distance" in covariates:
        columns.append(data["distance"] / max(distance_scale, EPS))
        spec.append(("distance_coef", -1))
    if "profit" in covariates:
        columns.append(data["profit"] - 1.0)
        spec.append(("profit_coef", -1))
    if "region" in covariates:
        for r in range(1, int(data["n_regions"])):
            columns.append((data["region"] == r).astype(np.float64))
            spec.append(("region_mu", r))
    if "season" in covariates:
        period = float(season_period) or 10.0
        omega = 2.0 * np.pi * data["t"] / period
        node = data["node"]
        sin_t = np.sin(omega)
        cos_t = np.cos(omega)
        for j in range(n):
            columns.append(sin_t * (node == j))
            spec.append(("season_sin", j))
        for j in range(n):
            columns.append(cos_t * (node == j))
            spec.append(("season_cos", j))
    if "spill" in covariates:
        kernel = data["kernel"]
        vnow = data["visited_now"]
        node = data["node"]
        for v in range(n):
            columns.append(vnow[:, v].astype(np.float64) * kernel[v, node])
            spec.append(("spill_coef", v))
    X = np.column_stack(columns) if columns else np.zeros((n_rows, 0))
    return X, spec


def _fit_location(data, covariates, distance_scale, season_period):
    """Least-squares location fit; returns the DSL parameter dict and residuals."""
    delta = data["delta"]
    n = int(data["n"])
    X, spec = _design_matrix(data, covariates, distance_scale, season_period)
    if X.shape[1] == 1:
        mu = float(delta.mean()) if delta.size else 0.0
        return {"mu": mu}, delta - mu, 1
    coef, *_ = np.linalg.lstsq(X, delta, rcond=None)
    params = {"mu": float(coef[0])}
    arrays = {}
    for (name, index), value in zip(spec[1:], coef[1:]):
        if index < 0:
            params[name] = float(value)
            continue
        if name == "region_mu":
            params.setdefault("region_mu", [0.0] * int(data["n_regions"]))
            params["region_mu"][index] = float(value)
        else:
            arrays.setdefault(name, np.zeros(n, dtype=np.float64))
            arrays[name][index] = float(value)
    params.update(arrays)
    return params, delta - X @ coef, int(X.shape[1])


def _fit_gaussian(structure, data, distance_scale):
    delta = data["delta"]
    if delta.size == 0:
        return dsl.GaussianDrift()
    params, resid, n_par = _fit_location(
        data, structure.covariates, distance_scale, structure.season_period
    )
    dof = max(resid.size - n_par, 1)
    sigma = float(np.sqrt(np.sum(resid**2) / dof)) if resid.size else 0.05
    params["sigma"] = max(sigma, EPS)
    return dsl.GaussianDrift(distance_scale=distance_scale, **params)


def _fit_uniform(structure, data, distance_scale):
    delta = data["delta"]
    if delta.size == 0:
        return dsl.UniformDrift()
    params, resid, n_par = _fit_location(
        data, structure.covariates, distance_scale, structure.season_period
    )
    # Method-of-moments support estimate.  ``max|residual|`` is the exact MLE
    # for a clean uniform sample, but with covariates it is inflated by the
    # (small) coefficient-estimation error of every point, which lowers the
    # density of the whole held-out set.  ``sqrt(3) * std`` is far more stable.
    # The degrees-of-freedom correction removes the in-sample advantage of
    # higher-dimensional structures, so a flexible model cannot shrink its
    # support just by fitting noise.  ``dof = N - 3p`` is deliberately
    # conservative: it estimates the predictive residual scale (noise variance
    # plus the error of the fitted location on unseen rounds), which is what
    # matters when the likelihood is evaluated on held-out episodes.
    dof = max(resid.size - 3 * n_par, 1)
    if resid.size:
        half = np.sqrt(3.0 * np.sum(resid**2) / dof)
    else:
        half = 0.05
    params["half_width"] = max(half, EPS)
    return dsl.UniformDrift(distance_scale=distance_scale, **params)


def _fit_mixture(delta, k=2, n_iter=100):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.size < k:
        return dsl.MixtureDrift(
            [float(delta.mean()) if delta.size else 0.0],
            [float(delta.std()) if delta.size else 0.05],
            [1.0],
        )
    qs = np.linspace(0.1, 0.9, k)
    means = np.quantile(delta, qs)
    sigma = max(float(delta.std()), EPS)
    sigmas = np.full(k, sigma)
    weights = np.full(k, 1.0 / k)
    for _ in range(n_iter):
        comp = [
            weights[j]
            * np.exp(-0.5 * ((delta - means[j]) / sigmas[j]) ** 2)
            / max(sigmas[j], EPS)
            for j in range(k)
        ]
        comp = np.stack(comp, axis=1)
        denom = comp.sum(axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        resp = comp / denom
        nk = resp.sum(axis=0) + EPS
        weights = nk / delta.size
        means = (resp * delta[:, None]).sum(axis=0) / nk
        sigmas = np.sqrt(
            (resp * (delta[:, None] - means[None, :]) ** 2).sum(axis=0) / nk
        )
        sigmas = np.maximum(sigmas, 1e-6)
    order = np.argsort(means)
    return dsl.MixtureDrift(means[order], sigmas[order], weights[order])


def fit_drift_program(structure, episodes, env, n_regions=4, seed=0):
    """Fit one RPP drift structure and return a :class:`DriftProgram`."""
    data = drift_dataset(episodes, env, n_regions=n_regions, seed=seed)
    distance_scale = float(getattr(env, "max_dist", 100.0)) or 100.0
    if structure.kind == "gaussian":
        effect = _fit_gaussian(structure, data, distance_scale)
    elif structure.kind == "uniform":
        effect = _fit_uniform(structure, data, distance_scale)
    elif structure.kind == "mixture":
        effect = _fit_mixture(data["delta"], k=int(structure.mixture))
    else:
        raise KeyError("unknown drift structure %r" % (structure.kind,))
    program = DriftProgram(effect)
    program.structure = structure
    program.loglik = float(loglik_drift(program, episodes, env, n_regions, seed))
    return program


def loglik_drift(program, episodes, env, n_regions=4, seed=0):
    """Summed log-density of observed unvisited-POI profit changes."""
    regions = _drift_regions(env, n_regions=n_regions, seed=seed)
    kernel = dsl.spill_kernel(env)
    initial = np.asarray(
        getattr(env, "initial_profit", np.zeros(env.n)), dtype=np.float64
    )
    total = 0.0
    for episode in episodes:
        for round_ in episode:
            before = np.asarray(_field(round_, "profit_before"), dtype=np.float64)
            after = np.asarray(_field(round_, "profit_after"), dtype=np.float64)
            visited_before = np.asarray(_field(round_, "visited_before"), dtype=bool)
            visited_after = np.asarray(_field(round_, "visited_after"), dtype=bool)
            mask = (~visited_after) & (before > 0.0) & (after > env.profit_min + 1e-9)
            if not mask.any():
                continue
            visited_now = visited_after & ~visited_before
            distance = _distance_to_action(env, _field(round_, "action"))
            feats = dsl.DriftFeatures(
                profit=before[None, :],
                visited=visited_after[None, :],
                t=np.asarray([int(_field(round_, "t"))], dtype=np.int64),
                distance=distance[None, :],
                region=regions,
                initial=initial[None, :],
                visited_now=visited_now[None, :],
                kernel=kernel,
            )
            ll = program.logpdf((after - before)[None, :], feats)[0]
            finite = np.isfinite(ll) & mask
            total += float(ll[finite].sum())
    return total


def fit_drift_library(structures, episodes, env, n_regions=4, seed=0):
    """Fit all drift structures, returning ``(best_program, programs)``."""
    programs = [
        fit_drift_program(s, episodes, env, n_regions, seed) for s in structures
    ]
    best = max(programs, key=lambda p: p.loglik)
    return best, programs
