"""Gates of method Section 10.4 that need no language model.

usage: python -m lexlaw.experiments.gates GATE [args]
  G3            revision flags with the ksi conversion removed; escape vs B2        (12 networks)
  G4            DSR-Q: quorum learner over 5 episodes; value of level-3 lines      (12 networks)
  G5 SETTING    rollout budget planner vs static schedules, true model; SETTING in
                C, known15, known30, random                                         (12 x 8)
"""
import math
import sys

import numpy as np

from ..agent import gap, run_episode
from ..envs.dsr import make_network
from ..envs.text import add_quorum_sentence, make_texts
from ..model.belief import Belief, flag_keys
from ..model.schema import Schema
from ..plan.budget import BudgetPlanner
from .tables import blind_belief, static_schedules


def ksi_bug(src):
    out = src.replace("return None if x is None else x / KSI_PER_MPA", "return x")
    assert out != src
    return out


def g3(n_net=12, n_seed=8):
    good, bad = Schema.reference(), Schema.reference(transform=ksi_bug)
    for level in (2, 3):
        flagged_bad, flagged_good, ret = 0, 0, {}
        for i in range(n_net):
            net = make_network(i)
            sup, req = make_texts(net, level)
            sp_bad, sp_good = bad.spaces(net, sup, req), good.spaces(net, sup, req)
            for name, sp in (("bug", sp_bad), ("correct", sp_good)):
                b = Belief(net, sp, 0.2)
                run_episode(net, b, 7919 * i)
                fl = flag_keys(b.key_statistics())
                if name == "bug":
                    flagged_bad += "tensile" in fl
                    print(f"   level {level} network {i:2d} bug:     flagged {fl}  escaped {len(b.escaped)}"
                          f"  unresolved suppliers in slots {sp.unresolved}", flush=True)
                else:
                    flagged_good += len(fl) == 0
                    print(f"   level {level} network {i:2d} correct: flagged {fl}", flush=True)
            for s in range(n_seed):
                seed = 7919 * i + 13 * s
                ret.setdefault("oracle", []).append(run_episode(net, None, seed))
                ret.setdefault("bug + escape", []).append(run_episode(net, Belief(net, sp_bad, 0.2), seed))
                ret.setdefault("bug, no escape", []).append(run_episode(net, Belief(net, sp_bad, 0.2, escape=False), seed))
                ret.setdefault("B2", []).append(run_episode(net, blind_belief(net), seed, em=False))
        print(f"\n== G3 level {level}: tensile flagged with the bug on {flagged_bad}/{n_net} (need >= 10); "
              f"no key flagged with the correct program on {flagged_good}/{n_net} (need >= 11)")
        for k in ("bug + escape", "bug, no escape", "B2"):
            print("   {:<16} first-episode gap {:+6.1f}% ±{:.1f}".format(k, *gap(ret[k], ret["oracle"])), flush=True)


def g4(n_net=12, n_seed=8, n_ep=5):
    schema = Schema.reference()
    chain = {k: np.zeros((n_net, n_ep)) for k in ("oracle", "quorum learner, lines ignored")}
    first = {}
    for i in range(n_net):
        net = make_network(i, "Q")
        sup, req = make_texts(net, 3)
        req_q = add_quorum_sentence(net, req, 3)
        sp_plain = schema.spaces(net, sup, req)
        sp_q = schema.spaces(net, sup, req_q)
        b = Belief(net, sp_plain, quorum=True, use_text=False)
        for e in range(n_ep):
            seed = 7919 * i + 13 * e
            chain["oracle"][i, e] = run_episode(net, None, seed)
            chain["quorum learner, lines ignored"][i, e] = run_episode(net, b, seed, em=False)
        for s in range(n_seed):
            seed = 7919 * i + 13 * s
            first.setdefault("oracle", []).append(run_episode(net, None, seed))
            first.setdefault("lines ignored", []).append(run_episode(net, Belief(net, sp_plain, quorum=True, use_text=False), seed, em=False))
            first.setdefault("level-3 lines + quorum sentence, EM", []).append(
                run_episode(net, Belief(net, sp_q, 0.2, quorum=True), seed))
        print(f"   network {i} done", flush=True)
    o = chain["oracle"]
    k = "quorum learner, lines ignored"
    print(f"\n== G4(a) DSR-Q chains of {n_ep} episodes: " + " | ".join(
        "{:+5.1f}% ±{:.1f}".format(*gap(chain[k][:, e], o[:, e])) for e in range(n_ep)) + "   (need within 5 points by episode 5)")
    print(f"== G4(b) first episode, {n_net} x {n_seed}:")
    for name in ("lines ignored", "level-3 lines + quorum sentence, EM"):
        print("   {:<40} {:+6.1f}% ±{:.1f}".format(name, *gap(first[name], first["oracle"])))
    d = np.mean(first["level-3 lines + quorum sentence, EM"]) - np.mean(first["lines ignored"])
    print(f"   improvement {100 * d / np.mean(first['oracle']):+.1f} points (need >= 5)")


def g5(setting, n_net=12, n_seed=8):
    K, cap, T = 20, 4, 15
    ret = {}
    for i in range(n_net):
        net = make_network(i)
        for s in range(n_seed):
            seed = 7919 * i + 13 * s
            if setting == "C":
                shock, variant, D, f = None, "C", None, 0.0
            elif setting.startswith("known"):
                f = int(setting[5:]) / 100
                shock, variant, D = (7, f), "known", 7
            else:
                D = int(np.random.default_rng(seed + 77).integers(4, 11))
                f, shock, variant = 0.2, (D, 0.2), "random"
            sch = static_schedules(K, cap, T, D if variant == "known" else None)
            ret.setdefault("asap", []).append(run_episode(net, None, seed, K, cap, aftershock=shock))
            names = ["uniform", "half-then-uniform", "keep4"] + (["reserve6", "reserve10"] if variant == "known" else [])
            for nm in names:
                ret.setdefault(nm, []).append(run_episode(net, None, seed, K, cap, aftershock=shock, schedule=sch[nm]))
            bp = BudgetPlanner(net, K, cap, T, variant=variant, D=D if variant == "known" else None, f=f, phi=0.01)
            ret.setdefault("rollout (5.2)", []).append(run_episode(net, None, seed, K, cap, aftershock=shock, budget=bp))
        print(f"   network {i} done", flush=True)
    print(f"\n== G5 setting {setting}: gap relative to asap (return {np.mean(ret['asap']):.1f}), true model")
    for k in ret:
        if k != "asap":
            print("   {:<20} {:+6.1f}% ±{:.1f}".format(k, *gap(ret[k], ret["asap"])))


if __name__ == "__main__":
    g = sys.argv[1]
    if g == "G3":
        g3()
    elif g == "G4":
        g4()
    elif g == "G5":
        g5(sys.argv[2])
