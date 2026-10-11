"""WS-option on DSR (baseline B8 of method.md): train on networks 30-99, test on networks 0-11.

usage: python Main.py [--variant C|Q|A] [--seed 0] [--epochs 20] [--episodes 10] [--test-nets 12] [--test-seeds 8]
Results are compared on identical random streams with the reference planner given the true
relation (B0) and the law-free heuristic B1, as in method.md Section 9.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

from Agent import Agent
from Env import Env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lexlaw.baselines.simple import run_fixed  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="C", choices=["C", "Q", "A"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--test-nets", type=int, default=12)
    ap.add_argument("--test-seeds", type=int, default=8)
    args = ap.parse_args()
    set_seed(args.seed)
    shock = (7, 0.15) if args.variant == "A" else None
    env = Env(30, args.variant, args.budget, args.cap, aftershock=shock)
    model = Agent(env)
    os.makedirs("results", exist_ok=True)
    job = f"dsr-{args.variant}-K={args.budget}-cap={args.cap}-epoch={args.epochs}-iteration={args.episodes}-seed={args.seed}"

    with open(f"results/{job}-training.txt", "w", encoding="utf-8") as f:
        log = lambda s: (print(s, file=f, flush=True), print(s, flush=True))  # noqa: E731
        model.train_model(list(range(30, 100)), args.epochs, args.episodes, np.random.default_rng(args.seed), log)

    model.eval()
    ret = {"WS-option": [], "B0 reference planner": [], "B1 reach": [], "average budget + learned selection": []}
    for i in range(args.test_nets):
        model.set_network(i)
        for s in range(args.test_seeds):
            seed = 7919 * i + 13 * s
            ret["WS-option"].append(model.run_episode(seed)[-1])
            ret["average budget + learned selection"].append(model.run_episode(seed, "average")[-1])
            ret["B0 reference planner"].append(run_fixed(env.net, seed, args.budget, args.cap, aftershock=shock)[0])
            ret["B1 reach"].append(run_fixed(env.net, seed, args.budget, args.cap, policy="reach", aftershock=shock)[0])
    o = np.array(ret["B0 reference planner"])
    with open(f"results/{job}-test.txt", "w", encoding="utf-8") as f:
        for k, v in ret.items():
            d = np.array(v) - o
            line = (f"{k:<36} return {np.mean(v):7.1f}   gap vs B0 {100 * d.mean() / o.mean():+6.1f}% "
                    f"±{100 * d.std(ddof=1) / np.sqrt(len(d)) / o.mean():.1f}")
            print(line, file=f)
            print(line)


if __name__ == "__main__":
    main()
