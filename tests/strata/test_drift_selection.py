"""RPP drift-family selection.

Generates behaviour logs under each hidden RPP drift law (``IID``, ``AR``,
``TREND``, ``SPILL``, ``SEASON`` and the real-data ``FITTED`` law), fits every
level-3 drift structure by maximum likelihood on the first 70% of episodes and
selects the structure with the best held-out log-likelihood on the remaining
30%.

The acceptance criterion is that the selected structure belongs to the true
law's family in at least 5 of the 6 cases.  The *family* of each hidden law is
the set of structures that contain its mechanism:

``IID``     unconditional noise (no covariates)
``AR``      mean reversion to the initial popularity
``TREND``   region-dependent location
``SPILL``   signed influence of the POIs visited that day
``SEASON``  per-POI periodic cycle over the day index
``FITTED``  a flexible empirical distribution (Gaussian mixture)
"""

from __future__ import annotations

import numpy as np
import pytest

from strata.domains.rpp import env as rpp_env
from strata.domains.rpp import hidden_laws as rpp_laws
from strata.domains.rpp import logs as rpp_logs
from strata.domains.rpp import points
from strata.laws import enumerate as en
from strata.laws import fit

INSTANCE_SEED = 3
VENUE_SEED = 1
EPISODE_SEED = 11
N_EPISODES = 60
BUDGET = 20
T = 10
TRAIN_FRACTION = 0.7

DRIFT_FAMILIES = {
    "IID": {"uniform", "gaussian", "mixture-2"},
    "AR": {"uniform-reversion", "gaussian-reversion"},
    "TREND": {
        "uniform-region",
        "gaussian-region",
        "uniform-full",
        "gaussian-full",
    },
    "SPILL": {"uniform-spill", "gaussian-spill"},
    "SEASON": {"uniform-season", "gaussian-season"},
    "FITTED": {"mixture-2", "mixture-3"},
}


def _case_instances():
    """The six (law name, env, hidden law) acceptance cases."""
    loc, initial, factor = points.random_instance(40, seed=INSTANCE_SEED)
    env = rpp_env.RPPEnv(
        loc=loc,
        initial_profit=initial,
        profit_factor=factor,
        budget=BUDGET,
        T=T,
        empty_day_advances=True,
    )
    cases = [
        (name, env, rpp_laws.get_law(name, factor=factor, seed=0))
        for name in ("IID", "AR", "TREND", "SPILL", "SEASON")
    ]

    loc, initial, factor, venues = points.make_instance_with_venues(
        "foursquare-NYC", n=30, seed=VENUE_SEED
    )
    fitted_env = rpp_env.RPPEnv(
        loc=loc,
        initial_profit=initial,
        profit_factor=factor,
        budget=BUDGET,
        T=T,
        empty_day_advances=True,
    )
    fitted_law = rpp_laws.make_law(
        "FITTED", dataset="foursquare-NYC", venues=venues, seed=0
    )
    cases.append(("FITTED", fitted_env, fitted_law))
    return cases


def _select(env, law):
    """Fit all level-3 drift structures; return held-out LL sorted descending."""
    episodes = rpp_logs.collect_logs(env, law, n_episodes=N_EPISODES, seed=EPISODE_SEED)
    n_train = int(round(TRAIN_FRACTION * len(episodes)))
    train, test = episodes[:n_train], episodes[n_train:]

    scored = []
    for structure in en.enumerate_drift_structures(3):
        program = fit.fit_drift_program(structure, train, env)
        held_out = fit.loglik_drift(program, test, env)
        scored.append((float(held_out), structure.name))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def test_drift_family_selection():
    print("\n%-8s %-18s %s" % ("law", "selected", "top-3 held-out LL"))
    print("-" * 72)
    passes = 0
    for name, env, law in _case_instances():
        scored = _select(env, law)
        selected = scored[0][1]
        ok = selected in DRIFT_FAMILIES[name]
        passes += int(ok)
        top = "  ".join("%s %.1f" % (s, ll) for ll, s in scored[:3])
        print(
            "%-8s %-18s %s%s"
            % (name, selected, top, "" if ok else "   [not in family]")
        )
    print("-" * 72)
    print("selected family in %d/6 cases" % passes)
    assert passes >= 5
