"""Finite belief over the L1 hypothesis set (IM and RPP).

``Belief`` starts from the prior weights stored on a
:class:`strata.core.types.HypothesisSet` and updates them by Bayes' rule with
the likelihood of each observed round/day.  The likelihood is computed at the
same granularity as the L1 fit so that a hypothesis is scored exactly as it was
fitted:

IM  node-level Bernoulli likelihood of the newly activated nodes among the
    eligible ones, using the candidate law's own exposure recursion
    (``rho``/``two_hop`` included), as in :func:`strata.laws.fit.loglik`;
RPP log-density of the observed profit changes of POIs that stayed unvisited,
    using the candidate :class:`~strata.laws.dsl.DriftProgram`.  POIs that hit
    the ``profit_min`` floor are censored observations: their contribution is
    the log-CDF ``P(delta <= profit_min - profit_before)`` instead of a density.

Bernoulli probabilities (and every RPP log-density/log-CDF) are clipped away
from zero, so a deterministic law that is contradicted by the data becomes
numerically negligible (~``1e-9``) instead of producing ``-inf``/``NaN`` and
poisoning the whole posterior.  Updates are done in log space to avoid
underflow.  The module has no global mutable state; :meth:`Belief.sample` is
deterministic given the supplied generator.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy.special import log_ndtr, logsumexp

from strata.core.types import (
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    Features,
    Hypothesis,
    HypothesisSet,
)
from strata.laws import dsl

PROB_EPS = 1e-9
LOG_FLOOR = math.log(PROB_EPS)


def _accumulate(law, E, a):
    fn = getattr(law, "accumulate", None)
    if fn is not None:
        return fn(E, a)
    rho = float(getattr(law, "rho", 0.0))
    return a + rho * E if rho else a


def _laws(hyps):
    if hasattr(hyps, "hyps"):
        return list(hyps.hyps)
    return list(hyps)


def _as_hypothesis(item, index):
    if isinstance(item, Hypothesis):
        return item
    return Hypothesis(law=item, loglik=0.0, prior=1.0, tie_group=int(index))


class Belief:
    """Posterior over a finite hypothesis set.

    Parameters
    ----------
    hyps
        A :class:`HypothesisSet` (or a list of hypotheses / laws).
    domain
        ``'im'`` or ``'rpp'``; selects the likelihood model.
    """

    def __init__(self, hyps, domain):
        domain = str(domain).lower()
        if domain not in ("im", "rpp"):
            raise ValueError(
                "unknown belief domain %r (expected 'im' or 'rpp')" % (domain,)
            )
        self.domain = domain
        self.hyps: List[Hypothesis] = [
            _as_hypothesis(h, i) for i, h in enumerate(_laws(hyps))
        ]
        if not self.hyps:
            raise ValueError("a belief needs at least one hypothesis")
        self.laws = [h.law for h in self.hyps]
        if hasattr(hyps, "weights"):
            prior = np.asarray(hyps.weights(), dtype=np.float64)
        else:
            prior = np.asarray([max(float(h.prior), 0.0) for h in self.hyps])
            if prior.sum() <= 0.0:
                prior = np.ones(len(self.hyps), dtype=np.float64)
            prior = prior / prior.sum()
        prior = np.maximum(np.asarray(prior, dtype=np.float64).reshape(-1), 0.0)
        if prior.size != len(self.hyps):
            prior = np.ones(len(self.hyps), dtype=np.float64)
        self._logw = np.log(np.maximum(prior, 1e-300))
        self._logw -= logsumexp(self._logw)
        self._im_E = None
        self._im_shape = None

    # ------------------------------------------------------------- posterior
    @property
    def weights(self) -> np.ndarray:
        """Normalised posterior weights, shape ``(m,)``."""
        return np.exp(self._logw - logsumexp(self._logw))

    def entropy(self) -> float:
        """Shannon entropy of the posterior in nats."""
        w = self.weights
        nz = w > 0.0
        if not nz.any():
            return 0.0
        return float(-np.sum(w[nz] * np.log(w[nz])))

    def map_index(self) -> int:
        """Index of the maximum-a-posteriori hypothesis."""
        return int(np.argmax(self.weights))

    def sample(self, rng) -> int:
        """Draw a hypothesis index from the posterior (BAMCP root sampling)."""
        w = self.weights
        w = w / w.sum()
        return int(rng.choice(w.size, p=w))

    # --------------------------------------------------------------- update
    def update(self, env_or_graph, state_before, action, state_after, info):
        """Incorporate one observed round; returns ``self`` for chaining."""
        if self.domain == "im":
            delta = self._im_loglik(
                env_or_graph, state_before, action, state_after, info
            )
        else:
            delta = self._rpp_loglik(
                env_or_graph, state_before, action, state_after, info
            )
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        delta = np.where(np.isfinite(delta), delta, LOG_FLOOR)
        self._logw = self._logw + delta
        shift = float(np.max(self._logw))
        if not np.isfinite(shift):
            self._logw = np.zeros_like(self._logw)
        else:
            self._logw = self._logw - shift
        self._logw -= logsumexp(self._logw)
        return self

    # ------------------------------------------------------------- IM model
    def _im_loglik(self, graph, before, action, after, info):
        status = np.asarray(before.status)
        M, n = status.shape
        eligible = np.asarray(info["eligible"], dtype=bool)
        y = np.asarray(info["new"], dtype=bool) & eligible
        community = (
            np.asarray(graph.community, dtype=np.int64)
            if getattr(graph, "community", None) is not None
            else np.zeros(n, dtype=np.int64)
        )
        a_info = np.asarray(info["a"], dtype=np.float32)
        two_hop_op = None
        if any(bool(getattr(law, "two_hop", False)) for law in self.laws):
            two_hop_op = graph.adj @ graph.adj

        shape = (M, n)
        if self._im_E is None or self._im_shape != shape or int(before.t) == 0:
            self._im_E = [
                np.asarray(before.E, dtype=np.float32).copy() for _ in self.laws
            ]
            self._im_shape = shape

        out = np.zeros(len(self.laws), dtype=np.float64)
        for i, law in enumerate(self.laws):
            a = a_info
            a2 = None
            if bool(getattr(law, "two_hop", False)) and two_hop_op is not None:
                seeded = np.asarray(info.get("seeded", np.zeros_like(eligible)))
                act = (status == STATUS_ACTIVE) | np.asarray(seeded, dtype=bool)
                a = act.astype(np.float32) @ graph.adj
                a2 = act.astype(np.float32) @ two_hop_op
            E_now = _accumulate(law, self._im_E[i], a)
            feats = Features(
                a=a,
                E=E_now,
                a2=a2,
                eligible=eligible,
                indeg=graph.indeg,
                community=community,
            )
            p = np.asarray(law.prob(feats), dtype=np.float64)
            p = np.clip(p, PROB_EPS, 1.0 - PROB_EPS)
            per = np.where(y, np.log(p), np.log(1.0 - p))
            out[i] = float(np.where(eligible, per, 0.0).sum())
            # laws with latent per-node state (gated mixtures) update it after scoring
            if hasattr(law, "observe_round"):
                law.observe_round(feats, y[0], eligible[0])

            E_next = np.where(y, 0.0, E_now)
            E_next[np.asarray(after.status) != STATUS_INACTIVE] = 0.0
            self._im_E[i] = E_next

        return out

    # ------------------------------------------------------------ RPP model
    def _rpp_regions(self, env):
        existing = getattr(env, "regions", None)
        if existing is not None and np.asarray(existing).size == env.n:
            return np.asarray(existing, dtype=np.int64)
        from strata.domains.rpp import points

        regions = points.spatial_clusters(env.loc, 4, seed=0)
        env.regions = regions
        return np.asarray(regions, dtype=np.int64)

    @staticmethod
    def _distances(env, action, M):
        A = np.asarray(action, dtype=np.int64)
        if A.size == 0:
            return np.zeros((M, env.n), dtype=np.float64)
        if A.ndim == 1:
            A = A[None, :]
        if A.shape[0] == 1 and M > 1:
            A = np.repeat(A, M, axis=0)
        out = np.zeros((M, env.n), dtype=np.float64)
        for m in range(M):
            valid = A[m][A[m] >= 0]
            if valid.size:
                out[m] = np.min(env.dist[:, valid], axis=1)
        return out

    def _rpp_loglik(self, env, before, action, after, info):
        before_profit = np.asarray(before.profit, dtype=np.float64)
        after_profit = np.asarray(after.profit, dtype=np.float64)
        M, n = before_profit.shape
        visited_after = np.asarray(after.visited, dtype=bool)
        if info is not None and "visited_now" in info:
            visited_now = np.asarray(info["visited_now"], dtype=bool)
        else:
            visited_now = visited_after & ~np.asarray(before.visited, dtype=bool)
        regions = self._rpp_regions(env)
        kernel = dsl.spill_kernel(env)
        initial = np.asarray(
            getattr(env, "initial_profit", np.zeros(n)), dtype=np.float64
        )
        distance = self._distances(env, action, M)
        feats = dsl.DriftFeatures(
            profit=before_profit,
            visited=visited_after,
            t=np.full(M, int(before.t), dtype=np.int64),
            distance=distance,
            region=regions,
            initial=np.broadcast_to(initial[None, :], (M, n)).copy(),
            visited_now=visited_now,
            kernel=kernel,
        )

        observed = (~visited_after) & (before_profit > 0.0)
        censored = observed & (after_profit <= env.profit_min + 1e-9)
        uncensored = observed & ~censored
        threshold = env.profit_min - before_profit
        delta_obs = after_profit - before_profit

        out = np.zeros(len(self.laws), dtype=np.float64)
        for i, law in enumerate(self.laws):
            ll = np.asarray(law.logpdf(delta_obs, feats), dtype=np.float64)
            ll = np.where(np.isfinite(ll), ll, LOG_FLOOR)
            total = float(np.maximum(ll[uncensored], LOG_FLOOR).sum())
            if censored.any():
                lc = self._log_cdf(law, threshold, feats)
                lc = np.where(np.isfinite(lc), lc, LOG_FLOOR)
                total += float(np.maximum(lc[censored], LOG_FLOOR).sum())
            out[i] = total
        return out

    @staticmethod
    def _log_cdf(program, threshold, feats):
        """Log ``P(delta <= threshold)`` of a drift program (censoring term)."""
        effect = getattr(program, "effect", program)
        mu = np.asarray(effect.mean(feats), dtype=np.float64)
        name = getattr(effect, "name", "")
        params = getattr(effect, "params", {})

        if name == "gaussian":
            sigma = max(float(params.get("sigma", 1.0)), 1e-12)
            return log_ndtr((threshold - mu) / sigma)
        if name == "uniform":
            h = max(float(params.get("half_width", 1.0)), 1e-12)
            lo = mu - h
            hi = mu + h
            frac = np.clip((threshold - lo) / (2.0 * h), PROB_EPS, 1.0 - PROB_EPS)
            out = np.log(frac)
            out = np.where(threshold >= hi, 0.0, out)
            out = np.where(threshold <= lo, LOG_FLOOR, out)
            return out
        if name == "mixture":
            means = np.asarray(effect.means, dtype=np.float64)
            sigmas = np.asarray(effect.sigmas, dtype=np.float64)
            logw = np.log(
                np.maximum(np.asarray(effect.weights, dtype=np.float64), 1e-300)
            )
            comp = [
                logw[j] + log_ndtr((threshold - means[j]) / sigmas[j])
                for j in range(means.size)
            ]
            return logsumexp(np.stack(comp, axis=0), axis=0)

        cdf = getattr(effect, "cdf", None)
        if cdf is not None:
            return np.log(np.clip(cdf(threshold, feats), PROB_EPS, 1.0 - PROB_EPS))
        # Unknown effect without a CDF: ignore the censored term (as the fit
        # does) rather than inventing a likelihood.
        return np.zeros_like(threshold)
