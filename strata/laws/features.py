"""Per-round sufficient statistics for the L1 law layer.

``Features`` is the interface bundle from ``strata.core.types``.  The helpers
here build it from either a live :class:`~strata.core.types.SimState` or one
logged :class:`~strata.core.types.Round`.

The important detail is the exposure recursion.  A law may have ``rho > 0``
(REINF/DECAY); in that case ``E <- a + rho * E`` and, after the round, the
exposure of every node that activated or is no longer inactive is reset to
zero (the same convention as ``experiments/kt_law/t2_fit.loglik``).  When no
law is supplied the recursion reduces to ``E = a``.
"""

import numpy as np

from strata.core.types import (
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    Features,
)
from strata.domains.im import env as im_env


def _rho(law):
    return float(getattr(law, "rho", 0.0)) if law is not None else 0.0


def _two_hop(law):
    return bool(getattr(law, "two_hop", False)) if law is not None else False


def seed_mask(status, seeds):
    """Boolean ``(M, n)`` mask of the seeds placed in this round."""
    M, n = status.shape
    out = np.zeros((M, n), dtype=bool)
    if seeds is None:
        return out
    s = np.asarray(seeds)
    if s.size == 0:
        return out
    sm = np.repeat(s[None, :], M, axis=0) if s.ndim == 1 else s
    valid = sm >= 0
    rows = np.broadcast_to(np.arange(M)[:, None], sm.shape)
    out[rows[valid], sm[valid]] = True
    return out


def active_after_seeding(status, seeds):
    """Active mask used to propagate this round (old active nodes + seeds)."""
    return (np.asarray(status) == STATUS_ACTIVE) | seed_mask(status, seeds)


def compute_features(graph, state, seeds=None, law=None, op=None):
    """Build the next-round :class:`Features` bundle.

    ``state`` is a :class:`SimState` *before* the round.  ``seeds`` are the
    seeds about to be placed (they become active for propagation but are not
    eligible to activate).  ``law`` supplies ``rho``/``two_hop``; when omitted
    the exposure is instantaneous and influence travels one hop.
    """
    status = state.status
    seeded = seed_mask(status, seeds)
    act = (status == STATUS_ACTIVE) | seeded
    op = im_env.influence_operator(graph, law, op)
    a = act.astype(np.float32) @ op
    E = im_env.accumulate(law, state.E, a)
    eligible = (status == STATUS_INACTIVE) & ~seeded

    a2 = None
    if _two_hop(law):
        two = graph.adj @ graph.adj
        a2 = act.astype(np.float32) @ two
    return Features(
        a=a,
        E=E,
        a2=a2,
        eligible=eligible,
        indeg=graph.indeg,
        community=graph.community,
    )


def _get(round_, key):
    """Read a field from a :class:`Round` or from a plain dict."""
    if isinstance(round_, dict):
        return round_[key]
    return getattr(round_, key)


def round_features(graph, round_, law=None, direct=False, E_prev=None):
    """Features for one logged round (``M = 1``).

    ``direct=True`` recomputes the one-hop active-neighbour count from
    ``round.before``/``round.seeded`` instead of using the stored ``a``.  This
    is needed for two-hop candidates, because the stored ``a`` already
    includes the two-hop operator of the logging law.  Pass ``E_prev`` to
    carry the exposure recursion; otherwise it starts from zero.
    """
    a_stored = np.asarray(_get(round_, "a"), dtype=np.float32)
    before = np.asarray(_get(round_, "before"))
    seeded = np.asarray(_get(round_, "seeded"), dtype=bool)
    after = np.asarray(_get(round_, "after"))
    y = np.asarray(_get(round_, "y"), dtype=bool)
    eligible = np.asarray(_get(round_, "eligible"), dtype=bool)

    act = (before == STATUS_ACTIVE) | seeded
    if direct:
        a = (act.astype(np.float32) @ graph.adj)[None, :]
    else:
        a = a_stored[None, :]

    a2 = None
    if _two_hop(law):
        two = graph.adj @ graph.adj
        a2 = (act.astype(np.float32) @ two)[None, :]

    E_prev = np.zeros_like(a) if E_prev is None else np.asarray(E_prev, np.float32)

    rho = _rho(law)
    E_now = a + rho * E_prev if rho else a
    y_eff = y & eligible
    E_now = np.where(y_eff[None, :], 0.0, E_now)
    dead = after != STATUS_INACTIVE
    E_now[:, dead] = 0.0

    return Features(
        a=a,
        E=E_now.astype(np.float32),
        a2=a2,
        eligible=eligible[None, :],
        indeg=graph.indeg,
        community=graph.community,
    )


def episode_features(graph, rounds, law=None, direct=False):
    """Yield one :class:`Features` bundle per round, carrying exposure forward."""
    rho = _rho(law)
    E = np.zeros(graph.adj.shape[0], dtype=np.float32)
    out = []
    for round_ in rounds:
        a = np.asarray(_get(round_, "a"), dtype=np.float32)
        eligible = np.asarray(_get(round_, "eligible"), dtype=bool)
        before = np.asarray(_get(round_, "before"))
        seeded = np.asarray(_get(round_, "seeded"), dtype=bool)
        after = np.asarray(_get(round_, "after"))
        y = np.asarray(_get(round_, "y"), dtype=bool)

        act = (before == STATUS_ACTIVE) | seeded
        if direct:
            a = (act.astype(np.float32) @ graph.adj)[None, :]
        else:
            a = a[None, :]

        a2 = None
        if _two_hop(law):
            two = graph.adj @ graph.adj
            a2 = (act.astype(np.float32) @ two)[None, :]

        E_now = a + rho * E[None, :] if rho else a
        feats = Features(
            a=a,
            E=E_now.astype(np.float32),
            a2=a2,
            eligible=eligible[None, :],
            indeg=graph.indeg,
            community=graph.community,
        )
        out.append(feats)

        y_eff = y & eligible
        E = np.where(y_eff, 0.0, E_now[0])
        E[after != STATUS_INACTIVE] = 0.0
    return out
