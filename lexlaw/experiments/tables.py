"""Reproduce the tables of method Section 9 with the lexlaw package (development networks only).

usage: python -m lexlaw.experiments.tables PART [n_networks] [n_seeds]
  9.1  Q1, decision sensitivity (30 x 3, K = 45 and K = 20)
  9.2  Q2, learners without text over 5 episodes (30 networks)
  9.4  schema-level model, first episode, levels 2 and 3 (12 x 8)
  9.5  five consecutive episodes, level 3 (12 networks)
  9.6  pacing with the true model and static schedules (12 x 8)
  9.7  DSR-Q, wrong quorum (12 x 8)
"""
import math
import sys

import numpy as np

from ..agent import gap, run_episode
from ..baselines.simple import run_fixed
from ..baselines.versionspace import grades_spaces
from ..envs.dsr import make_network
from ..envs.text import make_texts
from ..model.belief import Belief
from ..model.schema import Schema, build_spaces


def _row(name, vals, ref):
    g, se = gap(vals, ref)
    print(f"   {name:<46} {g:+6.1f}% ±{se:.1f}", flush=True)


def blind_belief(net):
    """B2: no text; every non-empty link subset per slot, prior 1/3 per link."""
    from ..model.belief import blind_space
    sp = grades_spaces(net, kind="none")
    b = Belief(net, sp, use_text=False, escape=False)
    for v in b.cons:
        b.sp[v] = blind_space([len(net.links_of_slot(s)) for s in sp.by_consumer[v]])
    return b


def t91(n_net=30, n_rep=3):
    for K, cap in ((45, 6), (20, 4)):
        ret, spent = {}, {}
        for i in range(n_net):
            net = make_network(i)
            true = net.true_compat()
            rng = net.rng_after
            noisy = np.concatenate([np.where(rng.random(len(m)) < 0.2, ~m, m)
                                    for v in sorted(net.slots) for _, m in net.slots[v]])
            models = {"oracle": true, "noisy20": noisy, "category": np.ones_like(true)}
            for r in range(n_rep):
                seed = 7919 * i + 13 * r
                for p in ("oracle", "noisy20", "category", "reach"):
                    tot, sp = run_fixed(net, seed, K, cap, compat=models.get(p), policy="reach" if p == "reach" else "model")
                    ret.setdefault(p, []).append(tot)
                    spent.setdefault(p, []).append(sp)
        print(f"\n== 9.1 K={K}, cap {cap}: {n_net} networks x {n_rep} episodes; oracle return {np.mean(ret['oracle']):.1f}")
        for p in ("noisy20", "category", "reach"):
            _row(f"{p} (budget used {np.mean(spent[p]):.1f})", ret[p], ret["oracle"])


def t92(n_net=30, n_ep=5):
    for K, cap in ((20, 4), (45, 6)):
        names = ("learn-grades", "learn-blind", "noisy20", "reach")
        ret = {p: np.zeros((n_net, n_ep)) for p in ("oracle",) + names}
        for i in range(n_net):
            net = make_network(i)
            rng = net.rng_after
            noisy = np.concatenate([np.where(rng.random(len(m)) < 0.2, ~m, m)
                                    for v in sorted(net.slots) for _, m in net.slots[v]])
            learners = {"learn-grades": Belief(net, grades_spaces(net, "grades"), use_text=False, escape=False),
                        "learn-blind": blind_belief(net)}
            for e in range(n_ep):
                seed = 7919 * i + 13 * e
                ret["oracle"][i, e] = run_fixed(net, seed, K, cap)[0]
                ret["noisy20"][i, e] = run_fixed(net, seed, K, cap, compat=noisy)[0]
                ret["reach"][i, e] = run_fixed(net, seed, K, cap, policy="reach")[0]
                for p, b in learners.items():
                    ret[p][i, e] = run_episode(net, b, seed, K, cap, em=False)
        print(f"\n== 9.2 K={K}, cap {cap}: {n_net} networks, {n_ep} consecutive episodes")
        for p in names:
            print(f"   {p:<13} " + " | ".join("{:+5.1f}% ±{:.1f}".format(*gap(ret[p][:, e], ret["oracle"][:, e]))
                                            for e in range(n_ep)), flush=True)


def _schema_beliefs(net, level, schema):
    sup, req = make_texts(net, level)
    return schema.spaces(net, sup, req)


def t94(n_net=12, n_seed=8):
    schema = Schema.reference()
    for level in (2, 3):
        ret, eta_hat = {}, []
        for i in range(n_net):
            net = make_network(i)
            sp = _schema_beliefs(net, level, schema)
            variants = {
                "no text, link subsets (B2)": (blind_belief(net), False),
                "class hypotheses, requirement lines ignored": (Belief(net, sp, use_text=False), False),
                "channel, eta by EM": (Belief(net, sp, 0.2), True),
                "channel, eta = 0.05": (Belief(net, sp, 0.05), False),
                "channel, eta = 0.2": (Belief(net, sp, 0.2), False),
                "channel, eta = 0.4": (Belief(net, sp, 0.4), False),
                "variant P, eta = 0.2": (Belief(net, sp, 0.2, form="prior"), False),
                "variant P, eta by EM": (Belief(net, sp, 0.2, form="prior"), True),
            }
            for s in range(n_seed):
                seed = 7919 * i + 13 * s
                ret.setdefault("oracle", []).append(run_episode(net, None, seed))
                for name, (b, em) in variants.items():
                    bb = b.clone()
                    ret.setdefault(name, []).append(run_episode(net, bb, seed, em=em))
                    if name == "channel, eta by EM":
                        eta_hat.append(bb.eta)
        print(f"\n== 9.4 level {level}: {n_net} networks x {n_seed} first episodes, K=20; reference return {np.mean(ret['oracle']):.1f}")
        for k, v in ret.items():
            if k != "oracle":
                _row(k, v, ret["oracle"])
        print(f"   eta after one episode (channel, EM): {np.mean(eta_hat):.3f}", flush=True)


def t95(n_net=12, n_ep=5):
    schema = Schema.reference()
    names = ("no text, link subsets (B2)", "class hypotheses, requirement lines ignored", "class hypotheses + channel, EM",
             "variant P, EM")
    ret = {k: np.zeros((n_net, n_ep)) for k in ("oracle",) + names}
    eta = np.zeros((n_net, n_ep))
    for i in range(n_net):
        net = make_network(i)
        sp = _schema_beliefs(net, 3, schema)
        bs = {names[0]: (blind_belief(net), False), names[1]: (Belief(net, sp, use_text=False), False),
              names[2]: (Belief(net, sp, 0.2), True), names[3]: (Belief(net, sp, 0.2, form="prior"), True)}
        for e in range(n_ep):
            seed = 7919 * i + 13 * e
            ret["oracle"][i, e] = run_episode(net, None, seed)
            for k, (b, em) in bs.items():
                ret[k][i, e] = run_episode(net, b, seed, em=em)
            eta[i, e] = bs[names[2]][0].eta
    print(f"\n== 9.5 level 3, {n_net} networks, {n_ep} consecutive episodes")
    for k in names:
        print(f"   {k:<46} " + " | ".join("{:+5.1f}% ±{:.1f}".format(*gap(ret[k][:, e], ret["oracle"][:, e]))
                                          for e in range(n_ep)), flush=True)
    print("   eta after each episode (channel): " + ", ".join(f"{x:.3f}" for x in eta.mean(0)))


def static_schedules(K, cap, T, D=None):
    return {
        "asap": lambda t, B: min(cap, B),
        "uniform": lambda t, B: min(cap, B, int(math.ceil(B / (T - t)))),
        "half-then-uniform": lambda t, B: min(cap, B) if B > K // 2 else min(cap, B, int(math.ceil(B / (T - t)))),
        "keep4": lambda t, B: min(cap, max(B - 4, 0)) if t < T - 5 else min(cap, B, int(math.ceil(B / (T - t)))),
        "reserve6": lambda t, B: min(cap, max(B - 6, 0)) if (D is not None and t < D) else min(cap, B),
        "reserve10": lambda t, B: min(cap, max(B - 10, 0)) if (D is not None and t < D) else min(cap, B),
    }


def t96(n_net=12, n_seed=8):
    nets = [make_network(i) for i in range(n_net)]
    settings = [("DSR-C, K = 20", 20, 4, 0.01, None), ("DSR-C, K = 45", 45, 6, 0.01, None),
                ("phi = 0.03", 20, 4, 0.03, None), ("phi = 0.05", 20, 4, 0.05, None),
                ("DSR-A, D = 7, f = 0.15", 20, 4, 0.01, (7, 0.15)), ("DSR-A, D = 7, f = 0.30", 20, 4, 0.01, (7, 0.30))]
    for label, K, cap, phi, shock in settings:
        sch = static_schedules(K, cap, 15, shock[0] if shock else None)
        names = ["uniform", "half-then-uniform", "keep4"] + (["reserve6", "reserve10"] if shock else [])
        ret = {}
        for i, net in enumerate(nets):
            for s in range(n_seed):
                seed = 7919 * i + 13 * s
                ret.setdefault("asap", []).append(run_episode(net, None, seed, K, cap, phi=phi, aftershock=shock))
                for nm in names:
                    ret.setdefault(nm, []).append(run_episode(net, None, seed, K, cap, phi=phi, aftershock=shock,
                                                              schedule=sch[nm]))
        print(f"\n== 9.6 {label}: gap relative to asap (return {np.mean(ret['asap']):.1f})")
        for nm in names:
            _row(nm, ret[nm], ret["asap"])


def t97(n_net=12, n_seed=8, p_dual=0.3):
    for K, cap in ((20, 4), (45, 6)):
        ret, used, share = {}, {}, []
        for i in range(n_net):
            net = make_network(i, "Q", p_dual)
            compat = net.true_compat()
            quo = net.true_quorum()
            single = np.ones_like(quo)
            dual = np.array([2 if (net.layer[net.slot_owner[s]] in (1, 2) and
                                   compat[net.slot_start[s]:net.slot_start[s] + net.slot_len[s]].sum() >= 2) else 1
                             for s in range(net.n_slots)])
            mask = np.isin(net.layer[net.slot_owner], (1, 2))
            share.append(np.mean(quo[mask] == 2))
            for s in range(n_seed):
                seed = 7919 * i + 13 * s
                for name, qm in (("oracle", quo), ("believes every slot needs one supplier", single),
                                 ("believes every dual-capable slot is dual", dual)):
                    tot, sp = run_fixed(net, seed, K, cap, compat=compat, quorum=qm)
                    ret.setdefault(name, []).append(tot)
                    used.setdefault(name, []).append(sp)
        print(f"\n== 9.7 DSR-Q p_dual={p_dual} (dual share {100 * np.mean(share):.0f}%), K={K}: reference {np.mean(ret['oracle']):.1f}")
        for name in list(ret)[1:]:
            _row(f"{name} (budget {np.mean(used[name]):.1f})", ret[name], ret["oracle"])


PARTS = {"9.1": t91, "9.2": t92, "9.4": t94, "9.5": t95, "9.6": t96, "9.7": t97}

if __name__ == "__main__":
    part = sys.argv[1]
    args = [int(x) for x in sys.argv[2:]]
    PARTS[part](*args)
