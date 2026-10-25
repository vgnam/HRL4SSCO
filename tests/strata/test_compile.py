"""M2b acceptance tests for the compiled stochastic model.

The quantitative check is the plan's requirement: for 20 ``(z, macro)`` pairs,
the mean reward produced by :meth:`CompiledModel.sample` must agree with a
direct simulation from the same pooled concrete state within two standard
errors (IM: dblp n=300 under CC2; RPP: random n=40 instance under TREND).
The remaining tests cover the ``KeyError`` fallback signal and cache
bookkeeping.
"""

import os
import sys

import networkx as nx
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from strata.abstraction import regions as regions_mod  # noqa: E402
from strata.abstraction.abstract_state import Abstraction  # noqa: E402
from strata.abstraction.compile import CompiledModel  # noqa: E402
from strata.abstraction.library import OperatorLibrary  # noqa: E402
from strata.core.types import Macro  # noqa: E402
from strata.domains.im import graph as im_graph  # noqa: E402
from strata.domains.im import hidden_laws as im_laws  # noqa: E402
from strata.domains.rpp import env as rpp_env  # noqa: E402
from strata.domains.rpp import hidden_laws as rpp_laws  # noqa: E402
from strata.domains.rpp import points  # noqa: E402


def _pairs(model, n_pairs=20):
    """Twenty ``(z, macro)`` pairs with ``k > 0`` from the populated pool."""
    pairs = []
    for z in sorted(model.pool):
        if len(pairs) >= n_pairs:
            break
        if z[0] >= model.T or z[1] <= 0:
            continue
        options = [m for m in model.macros(z) if m.k > 0]
        if not options:
            continue
        pairs.append((z, options[len(pairs) % len(options)]))
    assert len(pairs) == n_pairs, "pool produced only %d usable pairs" % len(pairs)
    return pairs


def _assert_matches_direct(model, pairs, n_calls=400, n_direct=64):
    for i, (z, macro) in enumerate(pairs):
        rng_sample = np.random.default_rng(1000 + i)
        sampled = np.array(
            [model.sample(z, macro, 0, rng_sample)[1] for _ in range(n_calls)]
        )
        state = model.pool[z][0]
        _, z_next, direct = model.simulate(
            state, macro, 0, np.random.default_rng(5000 + i), n_samples=n_direct
        )
        assert isinstance(z_next, tuple)
        assert len(z_next) == n_direct
        assert direct.shape == (n_direct,)

        var_sampled = float(np.var(sampled, ddof=1))
        var_cached = float(
            np.var(model._cache[model._cache_key(z, macro, 0)][1], ddof=1)
        )
        var_direct = float(np.var(direct, ddof=1))
        se_diff = np.sqrt(
            var_sampled / n_calls + var_cached / model.n_samples + var_direct / n_direct
        )
        diff = abs(float(sampled.mean()) - float(direct.mean()))
        assert diff <= 2.0 * se_diff + 1e-9, (
            "pair %d: sample %.4f vs direct %.4f (2SE %.4f)"
            % (i, sampled.mean(), direct.mean(), 2.0 * se_diff)
        )


# ------------------------------------------------------------------------ IM


@pytest.fixture(scope="module")
def im_model():
    G = im_graph.sample_instance("dblp", 300, seed=101)
    law = im_laws.make_law("CC2", G)
    ab = Abstraction("im", regions_mod.make_regions(G, k_top=3))
    lib = OperatorLibrary.standard("im")
    model = CompiledModel("im", G, [law], ab, lib, n_samples=24, seed=0, pool_cap=1)
    model.build_pool(n_rollouts=12, policies=("base", "random"))
    return model


def test_im_sample_matches_direct_simulation(im_model):
    pairs = _pairs(im_model)
    _assert_matches_direct(im_model, pairs)


# ----------------------------------------------------------------------- RPP


@pytest.fixture(scope="module")
def rpp_model():
    loc, initial, _ = points.random_instance(40, seed=7)
    env = rpp_env.RPPEnv(loc, initial, budget=20, T=8, seed=0, empty_day_advances=True)
    law = rpp_laws.make_law("TREND", seed=0)
    ab = Abstraction("rpp", regions_mod.make_spatial_regions(env, k_top=3))
    lib = OperatorLibrary.standard("rpp")
    model = CompiledModel("rpp", env, [law], ab, lib, n_samples=24, seed=0, pool_cap=1)
    model.build_pool(n_rollouts=12, policies=("base", "random"))
    return model


def test_rpp_sample_matches_direct_simulation(rpp_model):
    pairs = _pairs(rpp_model)
    _assert_matches_direct(rpp_model, pairs)


# ----------------------------------------------------- contract / diagnostics


def _small_model():
    G = im_graph.build_graph(
        nx.gnp_random_graph(30, 0.15, seed=6, directed=True), name="m2b-small"
    )
    law = im_laws.make_law("CC2", G)
    ab = Abstraction("im", regions_mod.make_regions(G, k_top=2))
    model = CompiledModel(
        "im",
        G,
        [law],
        ab,
        OperatorLibrary.standard("im"),
        n_samples=8,
        seed=3,
        T=5,
        budget=10,
        pool_cap=2,
    )
    model.build_pool(n_rollouts=4, policies=("base", "random"))
    return model


def test_sample_raises_keyerror_for_unpooled_state():
    model = _small_model()
    rng = np.random.default_rng(0)
    before = model.stats()
    with pytest.raises(KeyError):
        model.sample(("absent", 3, 0, 1, 2, 3, 4), Macro("hub", 0, 2), 0, rng)
    after = model.stats()
    assert after == before


def test_stats_count_hits_and_misses():
    model = _small_model()
    z, macro = _pairs(model, n_pairs=1)[0]
    rng = np.random.default_rng(11)

    before = model.stats()
    model.sample(z, macro, 0, rng)
    first_miss = model.stats()
    assert first_miss["misses"] == before["misses"] + 1
    assert first_miss["hits"] == before["hits"]
    assert first_miss["cache_size"] == before["cache_size"] + 1

    for _ in range(5):
        model.sample(z, macro, 0, rng)
    after = model.stats()
    assert after["hits"] == before["hits"] + 5
    assert after["misses"] == before["misses"] + 1
    assert after["cache_size"] == before["cache_size"] + 1


def test_expected_reward_and_outcome_distribution():
    model = _small_model()
    z, macro = _pairs(model, n_pairs=1)[0]
    rng = np.random.default_rng(21)
    z_next, reward = model.sample(z, macro, 0, rng)
    assert isinstance(z_next, tuple) and isinstance(reward, float)

    expected = model.expected_reward(z, macro, 0)
    assert np.isfinite(expected)
    dist = model.outcome_distribution(z, macro, 0)
    assert dist and abs(sum(dist.values()) - 1.0) < 1e-12
    assert all(isinstance(k, tuple) for k in dist)

    with pytest.raises(KeyError):
        model.expected_reward(z, Macro("gate-3", 0, 99), 0)


def test_build_pool_is_deterministic():
    a = _small_model()
    b = _small_model()
    assert sorted(a.pool) == sorted(b.pool)
    for z in sorted(a.pool):
        assert len(a.pool[z]) == len(b.pool[z])
        for sa, sb in zip(a.pool[z], b.pool[z]):
            np.testing.assert_array_equal(sa.status, sb.status)
            np.testing.assert_array_equal(sa.E, sb.E)
            assert (sa.t, sa.budget_left) == (sb.t, sb.budget_left)
