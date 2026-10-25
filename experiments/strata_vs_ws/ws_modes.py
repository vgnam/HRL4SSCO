"""Evaluate every WS-Option test mode of ``IM/Main.py`` under the STRATA protocol.

``strata.eval.run`` only plays WS-Option as (budget=agent, seeding=agent)
without beam search.  The paper's test script instead reports 12
(budget policy, seeding policy) pairs and uses beam search whenever the seeding
policy is ``agent``.  This script trains WS-Option once per (dataset, law)
instance and evaluates every pair -- plus ``agent`` seeding with and without
beam search -- with the same episode seeds as ``strata.eval.run``, so the rows
can be compared one-to-one with the STRATA CSVs.

    py -3.11 -m experiments.strata_vs_ws.ws_modes --datasets dblp --laws WC CC2 \
        --oracle-csv out/compare_im/im_dblp.csv --out out/compare_im/ws_modes_dblp.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from strata.baselines import ws_option
from strata.eval import protocol

BUDGET_POLICIES = ("agent", "average", "static", "normal")
SEEDING_POLICIES = ("agent", "agent+beam", "score", "degree")
# RPP/Agent.py only has the agent and average budget policies.
RPP_BUDGET_POLICIES = ("agent", "average")
RPP_SEEDING_POLICIES = ("agent", "agent+beam")

CSV_FIELDS = ("instance", "dataset", "n", "law", "method", "episode", "value",
              "oracle_value", "regret", "train_seconds")


class WSModePolicy:
    """A trained WS-Option agent played with one (budget, seeding) mode."""

    def __init__(self, trained: ws_option.WSOptionPolicy, budget_policy, seeding_policy):
        self.agent = trained.agent
        self.adapter = trained.eval_adapter
        self.budget_policy = budget_policy
        self.seeding_policy = seeding_policy

    def reset(self, instance, rng) -> None:
        return None

    def _budget(self, state3, t_r, budget_r):
        agent = self.agent
        if self.budget_policy == "agent":
            return agent.get_option(state3, t_r, budget_r, True, epsilon=0)
        if self.budget_policy == "static":
            return agent.static_seeding_policy(t_r, budget_r)
        if self.budget_policy == "normal":
            return agent.normal_policy(t_r, budget_r=budget_r)
        return agent.average_policy(t_r, budget_r=budget_r)

    def _seeds(self, state3, k):
        agent = self.agent
        if k == 0:
            return []
        if self.seeding_policy == "agent":
            return agent.get_action(state=state3, budget=k, epsilon=0)
        if self.seeding_policy == "agent+beam":
            return agent.beam_search(state3, k)
        return agent.get_action_score(state=state3, budget=k, mode=self.seeding_policy)

    def _act_rpp(self, state, t):
        agent, adapter = self.agent, self.adapter
        # beam search reads the day and position from the adapter
        adapter.t = int(state.t)
        adapter.profit = np.asarray(state.profit[0], dtype=np.float64).copy()
        adapter.current_node = int(state.current[0])
        budget_r = int(np.asarray(state.budget_left).reshape(-1)[0])
        adapter.remaining_budget = budget_r
        feat, context = adapter.get_state(), adapter.get_context()
        t_r = adapter.T - int(t)
        if self.budget_policy == "agent":
            k = agent.get_option(feat, context, t_r, budget_r, epsilon=0)
        else:
            k = agent.average_policy(t_r, budget_r)
        k = int(min(k, budget_r))
        if k == 0:
            return np.zeros(0, dtype=np.int64)
        if self.seeding_policy == "agent+beam":
            action = agent.beam_search(feat, context, k, beta=10, gamma=10)
        else:
            action = agent.get_action(feat, context, k, epsilon=0)
        return np.asarray(action, dtype=np.int64)

    def act(self, state, t: int) -> np.ndarray:
        if getattr(state, "profit", None) is not None:
            return self._act_rpp(state, t)
        state3 = ws_option._to_im_state(state.status[0])
        budget_r = int(np.asarray(state.budget_left).reshape(-1)[0])
        t_r = self.agent.env.T - int(t)
        k = int(min(self._budget(state3, t_r, budget_r), budget_r))
        return np.asarray(self._seeds(state3, k), dtype=np.int64)

    def observe(self, state_before, action, state_after, info) -> None:
        return None


def _oracle_values(path, instance_name, episodes):
    """Per-episode oracle values from a ``strata.eval.run`` CSV (or ``None``)."""
    if not path or not os.path.exists(path):
        return None
    values = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["instance"] == instance_name and row["method"] == "oracle":
                values[int(row["episode"])] = float(row["value"])
    if len(values) < episodes:
        return None
    return np.array([values[ep] for ep in range(episodes)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", choices=("im", "rpp"), default="im")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="dataset names; 'random' is the synthetic RPP instance")
    parser.add_argument("--laws", nargs="+", default=["WC", "CC2"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle-csv", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    rpp = args.domain == "rpp"
    budget_policies = RPP_BUDGET_POLICIES if rpp else BUDGET_POLICIES
    seeding_policies = RPP_SEEDING_POLICIES if rpp else SEEDING_POLICIES
    for dataset in args.datasets:
        for law in args.laws:
            instance = protocol.make_instance(
                args.domain, dataset=None if dataset == "random" else dataset, n=args.n,
                graph_seed=args.seed, law=law, T=args.T, B=args.budget,
            )
            trained = ws_option.WSOptionPolicy(
                epochs=args.epochs, iterations=args.iterations, seed=args.seed
            )
            start = time.time()
            trained.reset(instance, None)
            train_seconds = time.time() - start
            print("%s trained in %.0fs" % (instance.name, train_seconds), flush=True)

            oracle = _oracle_values(args.oracle_csv, instance.name, args.episodes)
            for budget_policy in budget_policies:
                for seeding_policy in seeding_policies:
                    policy = WSModePolicy(trained, budget_policy, seeding_policy)
                    values = protocol.evaluate(
                        policy, instance, episodes=args.episodes, seed=args.seed
                    )
                    method = "ws:%s-%s" % (budget_policy, seeding_policy)
                    print("  %-24s %.2f" % (method, values.mean()), flush=True)
                    for ep in range(args.episodes):
                        if oracle is None:
                            oracle_value, regret = float("nan"), float("nan")
                        else:
                            oracle_value = float(oracle[ep])
                            regret = (oracle_value - values[ep]) / float(oracle.mean())
                        rows.append({
                            "instance": instance.name, "dataset": dataset, "n": args.n,
                            "law": law, "method": method, "episode": ep,
                            "value": float(values[ep]), "oracle_value": oracle_value,
                            "regret": regret, "train_seconds": round(train_seconds, 1),
                        })

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
