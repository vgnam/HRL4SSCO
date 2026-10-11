"""All methods on the new problems DSR-C, DSR-Q, DSR-A (first episode on networks 0-11 x 8 seeds).

usage:
  python -m lexlaw.experiments.compare synth            # the language model writes the schema programs
  python -m lexlaw.experiments.compare run C|Q|A        # B0, B1, GA, WS-option, LexLaw (LLM / reference program)

Programs: one per text level (2, 3) and text family ("C": DSR text, used for C and A; "Q": DSR text
with the dual-sourcing sentence), written from networks 0-2 with at most 1 + 3 calls each (Section
4.1), plus 4 calls of the slot-by-slot reader for check C7. Stored in lexlaw_cache/programs/.
"""
import json
import os
import re
import sys

import numpy as np

from ..agent import gap, run_episode
from ..baselines.ga import run_ga
from ..baselines.readers import read_llm
from ..baselines.simple import run_fixed
from ..envs.dsr import make_network
from ..envs.text import MISSING, add_quorum_sentence, make_texts
from ..llm.client import ROOT, Chat
from ..llm.synthesize import synthesize
from ..model.belief import Belief
from ..model.schema import Schema
from ..plan.budget import BudgetPlanner

PROG_DIR = os.path.join(ROOT, "lexlaw_cache", "programs")
SHOCK = (7, 0.15)


def texts(net, level, family):
    sup, req = make_texts(net, level)
    if family == "Q":
        req = add_quorum_sentence(net, req, level)
    return sup, req


def synth():
    os.makedirs(PROG_DIR, exist_ok=True)
    chat = Chat()
    for family in ("C", "Q"):
        for level in (2, 3):
            path = os.path.join(PROG_DIR, f"{family}_level{level}.py")
            if os.path.exists(path):
                print(f"== {family} level {level}: exists, skipped")
                continue
            print(f"== program for text family {family}, level {level}", flush=True)
            nets_texts = []
            for i in range(3):
                net = make_network(i, family)
                sup, req = texts(net, level, family)
                nets_texts.append((net, sup, req))
            rng = np.random.default_rng(0)                    # C7: 100 random slots that carry a requirement
            pool = [(n, k) for n, (net, sup, req) in enumerate(nets_texts) for k, r in sorted(req.items()) if r != MISSING]
            pick = [pool[x] for x in sorted(rng.choice(len(pool), 100, replace=False))]
            answers = []
            for n, (net, sup, req) in enumerate(nets_texts):
                keys = [k for m, k in pick if m == n]
                answers.append(read_llm(net, sup, req, chat, keys))
            calls0 = chat.calls
            schema, chk, tried = synthesize(chat, nets_texts, answers)
            info = dict(family=family, level=level, attempts=len(tried), calls_total=chat.calls,
                        synthesis_calls=chat.calls - calls0, tokens_in=chat.tok_in, tokens_out=chat.tok_out)
            if schema is None:
                print(f"   no program passed Section 4.2 after {len(tried)} attempts", flush=True)
                for t, src in enumerate(tried):
                    with open(path.replace(".py", f".failed{t}.py"), "w", encoding="utf-8") as f:
                        f.write(src)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(schema.source)
                info["checks"] = {k: v for k, v in chk.items() if k != "examples"}
            with open(path.replace(".py", ".json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=1, default=str)
            print(f"   {info}", flush=True)


def load_program(family, level):
    path = os.path.join(PROG_DIR, f"{family}_level{level}.py")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return Schema(f.read(), sandbox=True)


def ws_option(variant):
    path = os.path.join(ROOT, "DSR", "results", f"dsr-{variant}-K=20-cap=4-epoch=20-iteration=10-seed=0-test.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("WS-option"):
                m = re.search(r"return\s+([\d.]+)\s+gap vs B0\s+([+-][\d.]+)% .([\d.]+)", line)
                return float(m.group(1)), float(m.group(2)), float(m.group(3))


def partition_rate(net, sp):
    ok = 0
    for slot in sp.slots:
        g = net.grade[slot.links]
        ok += sorted(tuple(m) for m in slot.masks) == sorted(tuple(g == c) for c in sorted(set(g)))
    return ok / len(sp.slots)


def run(variant, n_net=12, n_seed=8):
    family = "Q" if variant == "Q" else "C"
    shock = SHOCK if variant == "A" else None
    programs = {f"LexLaw, LLM program, level {L}": load_program(family, L) for L in (2, 3)}
    programs.update({f"LexLaw, reference program, level {L}": Schema.reference() for L in (2, 3)})
    ret, part = {}, {}
    for i in range(n_net):
        net = make_network(i, family)
        spaces = {}
        for name, schema in programs.items():
            if schema is None:
                continue
            L = int(name[-1])
            sup, req = texts(net, L, family)
            spaces[name] = schema.spaces(net, sup, req)
            part.setdefault(name, []).append(partition_rate(net, spaces[name]))
        for s in range(n_seed):
            seed = 7919 * i + 13 * s
            ret.setdefault("B0 greedy, true relation (reference)", []).append(run_episode(net, None, seed, aftershock=shock))
            ret.setdefault("B1 greedy, law-free (reach)", []).append(run_fixed(net, seed, policy="reach", aftershock=shock)[0])
            ret.setdefault("GA, true relation", []).append(run_ga(net, seed, aftershock=shock))
            for name, sp in spaces.items():
                b = Belief(net, sp, 0.2, quorum=(variant == "Q"))
                bp = BudgetPlanner(net, 20, 4, variant="known", D=SHOCK[0], f=SHOCK[1], phi=0.01) if variant == "A" else None
                ret.setdefault(name, []).append(run_episode(net, b, seed, aftershock=shock, budget=bp))
        print(f"   network {i} done", flush=True)
    ref = ret["B0 greedy, true relation (reference)"]
    print(f"\n== DSR-{variant}: {n_net} networks x {n_seed} first episodes, K=20, cap 4"
          + (f", aftershock day {SHOCK[0]} f={SHOCK[1]}" if shock else "") + f"; B0 return {np.mean(ref):.1f}")
    print(f"   {'method':<44} {'return':>7}   gap vs B0")
    for k, v in ret.items():
        g, se = gap(v, ref)
        extra = f"   (links partitioned as the truth in {100 * np.mean(part[k]):.1f}% of slots)" if k in part else ""
        print(f"   {k:<44} {np.mean(v):7.1f}   {g:+6.1f}% ±{se:.1f}{extra}")
    ws = ws_option(variant)
    if ws:
        print(f"   {'WS-option (DSR/Main.py, 20x10 episodes)':<44} {ws[0]:7.1f}   {ws[1]:+6.1f}% ±{ws[2]:.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "synth":
        synth()
    else:
        run(sys.argv[2])
