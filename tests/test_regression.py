"""Regression tests of method Section 10.3 (must pass before any new experiment).

Values were produced by the reference scripts with Python 3.13 / numpy 2.5.3; they are reproduced
here exactly with numpy 1.26.4.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexlaw.agent import run_episode  # noqa: E402
from lexlaw.baselines.simple import run_fixed  # noqa: E402
from lexlaw.envs.dsr import make_network  # noqa: E402
from lexlaw.envs.text import make_texts  # noqa: E402
from lexlaw.model.belief import Belief  # noqa: E402
from lexlaw.model.schema import ProgramError, Schema, load_program  # noqa: E402

NETS = {0: (460, 1840, 903, 80.470782, "ac10e68a189eec46a7bd083bdfc70766"),
        1: (451, 1804, 922, 74.807786, "02e3baa659df6f861594bc0ad96dd738")}
FIXED = [(0, 0, 986.408087, 492.684857), (0, 13, 832.535903, 516.516856),
         (1, 7919, 875.787169, 445.115720), (1, 7932, 1031.194982, 457.222355)]
LEARN = [(0, 0, 840.565890, 0.187624, 0.167993), (0, 13, 853.676581, 0.177944, 0.150039),
         (1, 7919, 826.589313, 0.144067, 0.138196), (1, 7932, 1017.535203, 0.126149, 0.113176)]


@pytest.mark.parametrize("i", [0, 1])
def test_network(i):
    net = make_network(i)
    sl = net.slots
    stats = (sum(len(x) for x in sl.values()), sum(len(l) for x in sl.values() for l, _ in x),
             sum(int(m.sum()) for x in sl.values() for _, m in x), round(float(net.w.sum()), 6), net.fingerprint())
    assert stats == NETS[i]


@pytest.mark.parametrize("i,seed,ref,b1", FIXED)
def test_reference_planner_and_b1(i, seed, ref, b1):
    net = make_network(i)
    assert round(run_fixed(net, seed)[0], 6) == ref
    assert round(run_fixed(net, seed, policy="reach")[0], 6) == b1


@pytest.mark.parametrize("i,seed,ret,eta,eta_p", LEARN)
def test_channel_em_map(i, seed, ret, eta, eta_p):
    net = make_network(i)
    sup, req = make_texts(net, 3)
    sp = Schema.reference().spaces(net, sup, req)
    b = Belief(net, sp, 0.2)
    assert round(run_episode(net, b, seed), 6) == ret
    assert round(b.eta, 6) == eta
    bp = Belief(net, sp, 0.2, form="prior")
    assert round(run_episode(net, bp, seed), 6) == ret              # variant P gives the same return
    assert round(bp.eta, 6) == eta_p


def test_reference_program_partitions_links():
    net = make_network(0)
    for level in (2, 3):
        sup, req = make_texts(net, level)
        sp = Schema.reference().spaces(net, sup, req)
        assert sp.unresolved == 0 and sp.c2_failures == 0
        for slot in sp.slots:
            g = net.grade[slot.links]
            assert sorted(tuple(m) for m in slot.masks) == sorted(tuple(g == c) for c in sorted(set(g)))


def test_sandboxed_parsing_matches_in_process():
    net = make_network(0)
    sup, req = make_texts(net, 3)
    a = Schema.reference(sandbox=False).spaces(net, sup, req)
    b = Schema.reference(sandbox=True).spaces(net, sup, req)
    assert [s.stated for s in a.slots] == [s.stated for s in b.slots]


@pytest.mark.parametrize("src", [
    "import os\ndef parse_supplier(t): return None\ndef parse_requirement(t): return None",
    "def parse_supplier(t): return open('x')\ndef parse_requirement(t): return None",
    "def parse_supplier(t): return t.__class__\ndef parse_requirement(t): return None",
    "def parse_supplier(t): return None",
])
def test_loader_rejects(src):
    with pytest.raises(ProgramError):
        load_program(src)
