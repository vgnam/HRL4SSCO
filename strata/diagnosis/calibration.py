"""Calibration tests of a hypothesis law against observed transitions.

The stochastic counterpart of TheoryCoder's "predicted vs observed" diff: a
law is miscalibrated when, in some bucket of the conditioning statistic, the
observed outcomes are far from what the law predicts.

* IM: per bucket of active in-neighbours ``a`` (0..6+), observed activations
  vs the sum of predicted probabilities; Pearson chi-square over buckets.
* RPP: per region and per day, mean observed profit change vs the law's
  predicted mean; chi-square of the standardised group means.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from strata.core.types import STATUS_INACTIVE
from strata.laws import fit
from strata.laws import llm_propose


def _field(r, k):
    return r[k] if isinstance(r, dict) else getattr(r, k)


def im_calibration(law, episodes, graph, max_a=6):
    obs = np.zeros(max_a + 1)
    exp = np.zeros(max_a + 1)
    var = np.zeros(max_a + 1)
    cnt = np.zeros(max_a + 1)
    for rounds in episodes:
        E = np.zeros(graph.indeg.size, dtype=np.float32)
        for r in rounds:
            el = np.asarray(_field(r, "eligible"), dtype=bool).reshape(-1)
            feats, E, _ = fit._features_for_round(law, r, E, graph.indeg, graph.community, graph)
            p = np.clip(np.asarray(law.prob(feats))[0], 1e-9, 1 - 1e-9)
            y = np.asarray(_field(r, "y"), dtype=bool).reshape(-1) & el
            b = np.minimum(np.asarray(_field(r, "a")).reshape(-1), max_a).astype(int)
            for v in range(max_a + 1):
                sel = el & (b == v)
                obs[v] += y[sel].sum()
                exp[v] += p[sel].sum()
                var[v] += (p[sel] * (1 - p[sel])).sum()
                cnt[v] += sel.sum()
            E = np.where(y, 0.0, E)
            after = np.asarray(_field(r, "after")).reshape(-1)
            E[after != STATUS_INACTIVE] = 0.0
    use = var > 1e-9
    # a deterministic law contradicted by data has zero variance and obs != exp
    contradicted = (~use) & (np.abs(obs - exp) > 0.5) & (cnt > 0)
    chi2 = float((((obs - exp) ** 2)[use] / var[use]).sum())
    df = max(1, int(use.sum()))
    p = 0.0 if contradicted.any() else float(stats.chi2.sf(chi2, df))
    rows = [(v, int(cnt[v]), int(obs[v]), float(exp[v])) for v in range(max_a + 1) if cnt[v]]
    return {"p_value": p, "chi2": chi2, "df": df, "rows": rows, "contradicted": bool(contradicted.any())}


def rpp_calibration(program, episodes, env, n_regions=4):
    data = llm_propose._rpp_obs(episodes, env, n_regions=n_regions, law=program)
    resid = data["delta"] - data["pred"]
    sd = float(np.std(resid)) or 1e-6
    chi2, df, rows = 0.0, 0, []
    for key in ("region", "t"):
        k = np.asarray(data[key])
        for v in np.unique(k):
            sel = k == v
            m = int(sel.sum())
            if m < 5:
                continue
            z = resid[sel].mean() / (sd / np.sqrt(m))
            chi2 += z * z
            df += 1
            rows.append(("%s=%s" % (key, v), m, float(resid[sel].mean()), float(z)))
    p = float(stats.chi2.sf(chi2, max(df, 1)))
    return {"p_value": p, "chi2": chi2, "df": df, "rows": rows}


def calibration(domain, law, episodes, obj):
    return im_calibration(law, episodes, obj) if domain == "im" else rpp_calibration(law, episodes, obj)
