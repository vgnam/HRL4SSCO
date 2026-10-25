"""Tests for structure enumeration and structure -> law construction."""

import numpy as np

from strata.core.types import Features
from strata.laws import enumerate as en


def _feats(n=10):
    a = np.tile(np.arange(n, dtype=np.float32), (1, 1))
    return Features(
        a=a,
        E=a.copy(),
        a2=np.ones_like(a),
        eligible=np.ones((1, n), bool),
        indeg=np.full(n, 3, dtype=np.int64),
        community=np.array([0] * 5 + [1] * 5, dtype=np.int64),
    )


def test_levels_are_nested_and_unique():
    l1 = en.enumerate_structures(1)
    l2 = en.enumerate_structures(2)
    l3 = en.enumerate_structures(3)
    assert {s.name for s in l1} <= {s.name for s in l2} <= {s.name for s in l3}
    for level in (l1, l2, l3):
        names = [s.name for s in level]
        assert len(names) == len(set(names))
    assert any(s.kind == "gate" for s in l2)
    assert not any(s.kind == "gate" for s in l1)
    assert any(s.kind == "soft_gate" for s in l3)
    assert any(s.kind == "free" for s in l3)


def test_every_im_structure_builds_and_evaluates():
    f = _feats()
    for structure in en.enumerate_structures(3):
        params = structure.param_grid()[0]
        law = structure.build(params)
        p = law.prob(f)
        assert p.shape == f.a.shape
        assert np.all(np.isfinite(p))
        assert np.all(p >= 0.0) and np.all(p <= 1.0)
        assert law.n_params() >= 0


def test_drift_levels():
    d1 = en.enumerate_drift_structures(1)
    d2 = en.enumerate_drift_structures(2)
    d3 = en.enumerate_drift_structures(3)
    assert {s.name for s in d1} <= {s.name for s in d2} <= {s.name for s in d3}
    assert any(s.region_mu for s in d2)
    assert any("t" in s.covariates for s in d3)
    assert len({s.name for s in d3}) == len(d3)


def test_lookup_helpers():
    assert en.structure_by_name("gate-2", 3).kind == "gate"
    assert en.drift_structure_by_name("gaussian-full", 3).region_mu


def test_named_variants_pin_their_parameter():
    # gate-g must only fit threshold g, otherwise tied gates collapse (K1)
    for s in en.enumerate_structures(2):
        if s.kind == "gate":
            g = int(s.name.split("-")[1])
            assert s.param_grid() == [{"g": g}]
    for s in en.enumerate_structures(3):
        if s.kind == "soft_gate":
            t = float(s.name.split("-")[1])
            assert {p["threshold"] for p in s.param_grid()} == {t}
