"""L1 hypothesis set with an identifiability certificate.

This module turns a list of maximum-likelihood fitted law programs into the
finite set :math:`\\mathcal{H}_\\varepsilon` used by the belief and planner:

* every candidate whose **validation** log-likelihood is within ``eps`` of the
  best candidate is kept; exact ties are grouped into a single ``tie_group``
  and are never broken (K1: breaking a tie by grid order is the failure mode);
* a validation prior is attached,
  ``exp(LL_val - LL_max) * exp(-0.5 * n_params * log N_obs)``, i.e. a
  likelihood ratio times a BIC complexity penalty.  ``prior_fn`` is a hook for
  the LLM prior of M6 and multiplies in an extra factor;
* pairwise *discriminating events* are counted on the validation log.  A pair
  with zero such events is not identified by the data (``identifiability_report``
  lists those pairs and the tie groups).

Both STRATA domains are supported.  For IM a discriminating event is an
eligible ``(node, round)`` whose activation probability differs materially
between the two laws; deterministic laws are compared by absolute probability
difference and non-deterministic laws by the symmetric Bernoulli KL.  For RPP
it is an observed unvisited ``(POI, day)`` profit change whose log-density
under the two drift programs differs by more than ``tau`` (or where one
program assigns zero mass).

All functions are deterministic given the supplied logs.  No module-level
mutable state is used.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from strata.core.types import (
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    Features,
    Hypothesis,
    HypothesisSet,
)
from strata.laws import dsl
from strata.laws import fit

EPS = 1e-12
TIE_RTOL = 1e-9


# =========================================================================
# Log-access helpers (IM Round / RPPRound objects or plain dicts)
# =========================================================================


def _field(round_: Any, key: str):
    if isinstance(round_, dict):
        return round_[key]
    return getattr(round_, key)


def _infer_n(episodes: Sequence) -> int:
    for rounds in episodes:
        for r in rounds:
            return int(
                np.asarray(_field(r, "a" if _has(r, "a") else "profit_before")).size
            )
    return 0


def _has(round_: Any, key: str) -> bool:
    if isinstance(round_, dict):
        return key in round_
    return hasattr(round_, key)


def _accumulate(law: Any, E: np.ndarray, a: np.ndarray) -> np.ndarray:
    fn = getattr(law, "accumulate", None)
    if fn is not None:
        return fn(E, a)
    rho = float(getattr(law, "rho", 0.0))
    return a + rho * E if rho else a


# =========================================================================
# IM discriminating events
# =========================================================================


def _im_round_events(law: Any, episodes: Sequence, indeg: np.ndarray, graph):
    """Per-round ``(p, eligible)`` pairs of a fitted IM law.

    The exposure recursion mirrors :func:`strata.laws.fit.loglik` so that a
    hypothesis is scored against exactly the observations used for fitting.
    """
    n = int(indeg.size)
    community = (
        np.asarray(graph.community)
        if graph is not None and getattr(graph, "community", None) is not None
        else np.zeros(n, dtype=np.int64)
    )
    out = []
    for rounds in episodes:
        E = np.zeros(n, dtype=np.float32)
        for r in rounds:
            eligible = np.asarray(_field(r, "eligible"), dtype=bool)
            a = np.asarray(_field(r, "a"), dtype=np.float32)
            a2 = None
            if bool(getattr(law, "two_hop", False)) and graph is not None:
                before = np.asarray(_field(r, "before"))
                seeded = np.asarray(_field(r, "seeded"), dtype=bool)
                act = (before == STATUS_ACTIVE) | seeded
                a = (act.astype(np.float32) @ graph.adj).reshape(-1)
                a2 = (act.astype(np.float32) @ (graph.adj @ graph.adj)).reshape(-1)
            E_now = _accumulate(law, E, a)
            feats = Features(
                a=a[None, :],
                E=E_now[None, :],
                a2=None if a2 is None else a2[None, :],
                eligible=eligible[None, :],
                indeg=indeg,
                community=community,
            )
            p = np.asarray(law.prob(feats), dtype=np.float64)
            if p.ndim > 1:
                p = p[0]
            p = np.where(eligible, p, 0.0)

            if eligible.any():
                out.append((p.copy(), eligible.copy()))

            E = E_now
            y = np.asarray(_field(r, "y"), dtype=bool) & eligible
            E = np.where(y, 0.0, E)
            after = np.asarray(_field(r, "after"))
            E[after != STATUS_INACTIVE] = 0.0
    return out


def _deterministic(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    return bool(np.all((values <= 1e-9) | (values >= 1.0 - 1e-9)))


def _bernoulli_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Symmetric Bernoulli KL, clipped away from the deterministic endpoints."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    q = np.clip(np.asarray(q, dtype=np.float64), EPS, 1.0 - EPS)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def _count_im(events_i, events_j, tau: float) -> int:
    count = 0
    for (p_i, elig), (p_j, _) in zip(events_i, events_j):
        if not elig.any():
            continue
        pi = p_i[elig]
        pj = p_j[elig]
        if _deterministic(pi) and _deterministic(pj):
            count += int(np.sum(np.abs(pi - pj) > tau))
        else:
            count += int(np.sum(_bernoulli_kl(pi, pj) > tau))
    return count


# =========================================================================
# RPP discriminating events
# =========================================================================


def _rpp_regions(env, n_regions: int, seed: int) -> np.ndarray:
    existing = getattr(env, "regions", None)
    if existing is not None and np.asarray(existing).size == env.n:
        return np.asarray(existing, dtype=np.int64)
    from strata.domains.rpp import points

    return points.spatial_clusters(env.loc, int(n_regions), seed=int(seed))


def _distance_to_action(env, action) -> np.ndarray:
    action = np.asarray(action, dtype=np.int64)
    action = action[action >= 0]
    if action.size == 0:
        return np.zeros(env.n, dtype=np.float64)
    return np.min(env.dist[:, action], axis=1)


def _rpp_round_events(
    program: Any, episodes: Sequence, env, n_regions: int = 4, seed: int = 0
):
    """Per-round ``(logpdf, mask)`` pairs of an RPP drift program.

    ``mask`` selects the observed unvisited POIs used by
    :func:`strata.laws.fit.loglik_drift` (nodes at the ``profit_min`` floor are
    treated as censored and excluded).
    """
    regions = _rpp_regions(env, n_regions, seed)
    kernel = dsl.spill_kernel(env)
    initial = np.asarray(
        getattr(env, "initial_profit", np.zeros(env.n)), dtype=np.float64
    )
    out = []
    for episode in episodes:
        for r in episode:
            before = np.asarray(_field(r, "profit_before"), dtype=np.float64)
            after = np.asarray(_field(r, "profit_after"), dtype=np.float64)
            visited_before = np.asarray(_field(r, "visited_before"), dtype=bool)
            visited_after = np.asarray(_field(r, "visited_after"), dtype=bool)
            mask = (~visited_after) & (before > 0.0) & (after > env.profit_min + 1e-9)
            if not mask.any():
                continue
            visited_now = visited_after & ~visited_before
            distance = _distance_to_action(env, _field(r, "action"))
            feats = dsl.DriftFeatures(
                profit=before[None, :],
                visited=visited_after[None, :],
                t=np.asarray([int(_field(r, "t"))], dtype=np.int64),
                distance=distance[None, :],
                region=regions,
                initial=initial[None, :],
                visited_now=visited_now[None, :],
                kernel=kernel,
            )
            ll = np.asarray(
                program.logpdf((after - before)[None, :], feats), np.float64
            )
            if ll.ndim > 1:
                ll = ll[0]
            out.append((ll.copy(), mask.copy()))
    return out


def _count_rpp(events_i, events_j, tau: float) -> int:
    count = 0
    for (ll_i, mask), (ll_j, _) in zip(events_i, events_j):
        if not mask.any():
            continue
        li = ll_i[mask]
        lj = ll_j[mask]
        finite = np.isfinite(li) & np.isfinite(lj)
        zero_mass = ~np.isfinite(li) | ~np.isfinite(lj)
        different = np.abs(li - lj) > tau
        count += int(np.sum((different & finite) | zero_mass))
    return count


# =========================================================================
# Public discriminating-event API
# =========================================================================


def _is_drift(law: Any) -> bool:
    return isinstance(law, dsl.DriftProgram) or (
        not hasattr(law, "prob") and hasattr(law, "logpdf")
    )


def _im_setup(episodes, indeg, graph):
    if graph is not None:
        if indeg is None:
            indeg = graph.indeg
    if indeg is None:
        n = _infer_n(episodes)
        indeg = np.ones(n, dtype=np.float64)
    return np.asarray(indeg, dtype=np.float64), graph


def discriminating_events(
    h_i: Any,
    h_j: Any,
    episodes: Sequence,
    tau: float = 1e-3,
    *,
    domain: Optional[str] = None,
    indeg: Optional[np.ndarray] = None,
    graph: Any = None,
    env: Any = None,
    n_regions: int = 4,
    seed: int = 0,
) -> int:
    """Number of logged events on which two hypotheses predict differently.

    ``domain`` is inferred from the law type when omitted (``'rpp'`` for
    :class:`~strata.laws.dsl.DriftProgram`, ``'im'`` otherwise).  RPP needs the
    instance ``env`` (geometry, regions and the spill kernel).
    """
    law_i = getattr(h_i, "law", h_i)
    law_j = getattr(h_j, "law", h_j)
    if domain is None:
        domain = "rpp" if (_is_drift(law_i) or _is_drift(law_j)) else "im"
    domain = str(domain).lower()

    if domain == "rpp":
        if env is None:
            raise ValueError("RPP discriminating events need the instance env")
        ev_i = _rpp_round_events(law_i, episodes, env, n_regions, seed)
        ev_j = _rpp_round_events(law_j, episodes, env, n_regions, seed)
        return _count_rpp(ev_i, ev_j, float(tau))

    if domain != "im":
        raise ValueError("unknown domain %r (expected 'im' or 'rpp')" % (domain,))
    indeg, graph = _im_setup(episodes, indeg, graph)
    ev_i = _im_round_events(law_i, episodes, indeg, graph)
    ev_j = _im_round_events(law_j, episodes, indeg, graph)
    return _count_im(ev_i, ev_j, float(tau))


# =========================================================================
# Validation likelihood and observation counts
# =========================================================================


def _validation_loglik(
    law: Any, domain: str, episodes: Sequence, env, indeg, graph, n_regions, seed
) -> float:
    if domain == "rpp":
        return float(fit.loglik_drift(law, episodes, env, n_regions, seed))
    indeg, graph = _im_setup(episodes, indeg, graph)
    return float(fit.loglik(law, episodes, indeg=indeg, graph=graph))


def _observation_count(domain: str, episodes: Sequence, env, indeg, graph) -> int:
    if domain == "rpp":
        total = 0
        for episode in episodes:
            for r in episode:
                before = np.asarray(_field(r, "profit_before"), dtype=np.float64)
                after = np.asarray(_field(r, "profit_after"), dtype=np.float64)
                visited_after = np.asarray(_field(r, "visited_after"), dtype=bool)
                mask = (
                    (~visited_after) & (before > 0.0) & (after > env.profit_min + 1e-9)
                )
                total += int(mask.sum())
        return total
    total = 0
    for rounds in episodes:
        for r in rounds:
            total += int(np.asarray(_field(r, "eligible"), dtype=bool).sum())
    return total


# =========================================================================
# Hypothesis-set construction
# =========================================================================


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TIE_RTOL * max(abs(a), abs(b), 1.0)


def _group_by_tie(order: Sequence[int], lls: np.ndarray) -> List[List[int]]:
    groups: List[List[int]] = []
    for idx in order:
        if groups and _close(float(lls[idx]), float(lls[groups[-1][0]])):
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def build_hypothesis_set(
    fitted: Sequence[Any],
    episodes_val: Sequence,
    eps: float,
    cap: Optional[int] = None,
    domain: str = "im",
    *,
    prior_fn=None,
    tau: float = 1e-3,
    indeg: Optional[np.ndarray] = None,
    graph: Any = None,
    env: Any = None,
    n_regions: int = 4,
    seed: int = 0,
) -> HypothesisSet:
    """Build :math:`\\mathcal{H}_\\varepsilon` from fitted law programs.

    Parameters
    ----------
    fitted
        Fitted IM ``LawProgram``/``ProductLaw`` objects or RPP
        ``DriftProgram`` objects.
    episodes_val
        Validation log used for the likelihood, tie detection, BIC sample size
        and the discriminating-event counts.
    eps
        Keep every candidate with ``LL_val >= LL_max - eps``.  Whole tie
        groups are kept or dropped together; ``cap`` bounds the size without
        ever splitting a tie group.
    domain
        ``'im'`` or ``'rpp'``.
    prior_fn
        Optional hook ``prior_fn(hypothesis) -> float`` multiplied into the
        prior (M6's LLM prior).
    """
    domain = str(domain).lower()
    if domain not in ("im", "rpp"):
        raise ValueError("unknown domain %r (expected 'im' or 'rpp')" % (domain,))
    if domain == "rpp" and env is None:
        raise ValueError("RPP hypothesis sets need the instance env")

    laws = [getattr(h, "law", h) for h in fitted]
    if not laws:
        return HypothesisSet([], np.zeros((0, 0), dtype=np.int64))

    lls = np.asarray(
        [
            _validation_loglik(
                law, domain, episodes_val, env, indeg, graph, n_regions, seed
            )
            for law in laws
        ],
        dtype=np.float64,
    )

    finite = np.isfinite(lls)
    if not finite.any():
        return HypothesisSet([], np.zeros((0, 0), dtype=np.int64))
    ll_max = float(np.max(lls[finite]))

    # Continuous (RPP) validation likelihoods grow with the number of observed
    # POI-days, so ``eps`` is applied to the mean log-density.  This makes the
    # tolerance comparable to the per-event Bernoulli tolerance of IM (the
    # acceptance test uses the same ``eps=5`` for both the exact IID family and
    # its Gaussian approximation).  The stored ``loglik`` and the prior remain
    # the total validation likelihood, as required by the BIC prior.
    n_obs = _observation_count(domain, episodes_val, env, indeg, graph)
    if domain == "rpp":
        score = lls / float(max(n_obs, 1))
    else:
        score = lls
    score_max = float(np.max(score[finite]))

    order = [int(i) for i in np.argsort(-score, kind="stable") if finite[i]]
    order = [i for i in order if score[i] >= score_max - float(eps)]
    groups = _group_by_tie(order, lls)

    if cap is None:
        kept = [i for group in groups for i in group]
    else:
        cap = int(cap)
        kept = []
        for group in groups:
            if not kept and len(group) > cap:
                # A single tie group that does not fit: never split it.
                kept.extend(group)
                break
            if len(kept) + len(group) <= cap:
                kept.extend(group)
            else:
                break

    if not kept:
        return HypothesisSet([], np.zeros((0, 0), dtype=np.int64))

    bic_scale = math.log(max(int(n_obs), 2))

    hyps: List[Hypothesis] = []
    group_ids: Dict[int, int] = {}
    next_group = 0
    for group in groups:
        for idx in group:
            group_ids[idx] = next_group
        next_group += 1

    for idx in kept:
        law = laws[idx]
        ll = float(lls[idx])
        n_params = int(law.n_params()) if hasattr(law, "n_params") else 0
        prior = math.exp(ll - ll_max) * math.exp(-0.5 * n_params * bic_scale)
        hyp = Hypothesis(
            law=law,
            loglik=ll,
            prior=float(prior),
            tie_group=int(group_ids[idx]),
        )
        if prior_fn is not None:
            factor = float(prior_fn(hyp))
            hyp.prior = float(hyp.prior * factor)
        hyps.append(hyp)

    m = len(hyps)
    disc = np.zeros((m, m), dtype=np.int64)
    if domain == "rpp":
        all_events = [
            _rpp_round_events(h.law, episodes_val, env, n_regions, seed) for h in hyps
        ]
        for i in range(m):
            for j in range(i + 1, m):
                c = _count_rpp(all_events[i], all_events[j], float(tau))
                disc[i, j] = disc[j, i] = c
    else:
        indeg_arr, graph_arr = _im_setup(episodes_val, indeg, graph)
        all_events = [
            _im_round_events(h.law, episodes_val, indeg_arr, graph_arr) for h in hyps
        ]
        for i in range(m):
            for j in range(i + 1, m):
                c = _count_im(all_events[i], all_events[j], float(tau))
                disc[i, j] = disc[j, i] = c

    return HypothesisSet(hyps=hyps, discriminating=disc)


# =========================================================================
# Identifiability certificate
# =========================================================================


def identifiability_report(hset: HypothesisSet) -> Dict[str, Any]:
    """Tie groups and pairs with no discriminating event in the validation log.

    Returns a dict with

    ``tie_groups``
        list of lists of hypothesis indices (ordered by descending LL) that
        share a validation log-likelihood within relative tolerance ``1e-9``;
    ``zero_discriminating``
        list of ``(i, j)`` pairs whose discriminating-event count is zero, i.e.
        the validation log cannot tell them apart;
    ``m``
        number of hypotheses.
    """
    hyps = list(hset.hyps)
    m = len(hyps)
    groups: Dict[int, List[int]] = {}
    for i, h in enumerate(hyps):
        groups.setdefault(int(h.tie_group), []).append(i)
    tie_groups = [groups[key] for key in sorted(groups)]
    disc = np.asarray(hset.discriminating)
    zero_pairs = [
        (i, j) for i in range(m) for j in range(i + 1, m) if int(disc[i, j]) == 0
    ]
    return {
        "tie_groups": tie_groups,
        "zero_discriminating": zero_pairs,
        "m": m,
    }
