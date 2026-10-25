"""Run STRATA with the LLM under the ``strata.eval.run`` protocol and keep its build report.

Same instances, logs and episode seeds as ``strata.eval.run`` (so rows pair with
the other CSVs of ``out/compare_im``), plus one JSON per instance with the L1
hypotheses and the L2 operator candidates, so no offline replay is needed to see
what the LLM contributed.

    py -3.11 experiments/strata_vs_ws/strata_llm.py --datasets dblp --laws WC CC2 \
        --tag llm2 --out-dir out/compare_im
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from strata.agent import StrataAgent
from strata.baselines import model_based
from strata.eval import protocol
from strata.llm import Chat

CSV_FIELDS = ("instance", "domain", "dataset", "n", "graph_seed", "law", "T", "B",
              "method", "episode", "value", "oracle_value", "regret")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--laws", nargs="+", default=["WC", "CC2"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--log-episodes", type=int, default=200)
    parser.add_argument("--selection-episodes", type=int, default=2)
    parser.add_argument("--n-sims", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="llm")
    parser.add_argument("--out-dir", default="out/compare_im")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    for dataset in args.datasets:
        rows, reports = [], {}
        for law in args.laws:
            instance = protocol.make_instance("im", dataset=dataset, n=args.n, graph_seed=args.seed,
                                              law=law, T=args.T, B=args.budget)
            logs = protocol.make_logs(instance, policies=None, n_episodes=args.log_episodes,
                                      seed=args.seed)
            oracle = protocol.evaluate(
                model_based.Oracle(logs, selection_episodes=args.selection_episodes, seed=args.seed),
                instance, episodes=args.episodes, seed=args.seed)
            agent = StrataAgent(logs, chat=Chat(), seed=args.seed, n_sims=args.n_sims)
            start = time.time()
            agent.build(instance)
            build_seconds = time.time() - start
            values = protocol.evaluate(agent, instance, episodes=args.episodes, seed=args.seed)
            print("%s build %.0fs strata %.2f oracle %.2f library %s" % (
                instance.name, build_seconds, values.mean(), oracle.mean(), agent.report["library"]),
                flush=True)
            report = dict(agent.report)
            report["build_seconds"] = build_seconds
            reports[instance.name] = report
            for method, vals in (("oracle", oracle), ("strata", values)):
                for ep in range(args.episodes):
                    rows.append({
                        "instance": instance.name, "domain": "im", "dataset": dataset,
                        "n": args.n, "graph_seed": args.seed, "law": law, "T": args.T,
                        "B": args.budget, "method": method, "episode": ep,
                        "value": float(vals[ep]), "oracle_value": float(oracle[ep]),
                        "regret": float((oracle[ep] - vals[ep]) / oracle.mean()),
                    })
        with open(os.path.join(args.out_dir, "%s_%s.csv" % (args.tag, dataset)), "w",
                  newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(args.out_dir, "%s_%s_report.json" % (args.tag, dataset)), "w",
                  encoding="utf-8") as handle:
            json.dump(reports, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
