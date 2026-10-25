"""Command-line benchmark driver for STRATA.

Typical use::

    python -m strata.eval.run --domain im --datasets dblp --n 300 \
        --laws CC2 --methods flat-hub* flat-any* wc-default libA libB libC oracle \
        --episodes 2 --log-episodes 100 --out out/strata_im.csv

The driver builds one :class:`strata.eval.protocol.Instance` per
(dataset, n, law) cell, collects behaviour logs once, evaluates the oracle
reference under the true law and then every requested method with the same
episode seeds (common random numbers).  The output is one CSV row per
(instance, method, episode), with the per-episode regret relative to the oracle.

Methods: ``flat-hub*``, ``flat-any*``, ``wc-default``, ``libA``, ``libB``,
``libC``, ``oracle``, ``ws-option``, ``greedy``, ``ga``.  ``ws-option`` is skipped
(with a warning) when PyTorch / the original agent files cannot be imported;
``greedy`` and ``ga`` (the WS-Option paper's RPP baselines) are RPP-only.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from strata.baselines import flat as flat_mod
from strata.baselines import model_based
from strata.baselines import ws_option
from strata.eval import protocol

METHODS = (
    "flat-hub*",
    "flat-any*",
    "wc-default",
    "libA",
    "libB",
    "libC",
    "oracle",
    "ws-option",
    "greedy",
    "ga",
)

CSV_FIELDS = (
    "instance",
    "domain",
    "dataset",
    "n",
    "graph_seed",
    "law",
    "T",
    "B",
    "method",
    "episode",
    "value",
    "oracle_value",
    "regret",
)


def _default_dataset(domain: str) -> str:
    return "dblp" if domain == "im" else "random"


def _default_law(domain: str) -> str:
    return "CC2" if domain == "im" else "IID"


def _normalise_dataset(domain: str, dataset: str) -> Optional[str]:
    if domain == "rpp" and str(dataset).lower() in ("random", "synthetic", "none"):
        return None
    return dataset


def _make_policy(method: str, instance, logs, args):
    """Build the policy object of ``method`` (``None`` when unavailable)."""
    domain = instance.domain
    if method == "flat-hub*":
        return flat_mod.best_flat(
            instance,
            ops=flat_mod.hub_ops(domain),
            allocs=flat_mod.ALLOCS,
            episodes=args.selection_episodes,
            seed=args.seed,
        )
    if method == "flat-any*":
        return flat_mod.best_flat(
            instance,
            ops=flat_mod.default_ops(domain),
            allocs=flat_mod.ALLOCS,
            episodes=args.selection_episodes,
            seed=args.seed,
        )
    if method == "wc-default":
        return model_based.WCDefault(
            logs,
            selection_episodes=args.selection_episodes,
            seed=args.seed,
        )
    if method == "libA":
        return model_based.LibA(
            logs,
            selection_episodes=args.selection_episodes,
            seed=args.seed,
        )
    if method == "libB":
        return model_based.LibB(
            logs,
            selection_episodes=args.selection_episodes,
            seed=args.seed,
        )
    if method == "libC":
        return model_based.LibC(
            logs,
            selection_episodes=args.selection_episodes,
            seed=args.seed,
        )
    if method == "oracle":
        return model_based.Oracle(
            logs,
            selection_episodes=args.selection_episodes,
            seed=args.seed,
        )
    if method == "ws-option":
        if not ws_option.available():
            return None
        return ws_option.WSOptionPolicy(seed=args.seed)
    if method in ("greedy", "ga"):
        if domain != "rpp":
            return None  # the paper's GA / greedy baselines are RPP-only
        from strata.baselines import rpp_classic

        if method == "greedy":
            return rpp_classic.GreedyPolicy(seed=args.seed)
        return rpp_classic.GAPolicy(seed=args.seed)
    if method.startswith("strata"):
        from strata.agent import StrataAgent, make_strata_methods

        variants = make_strata_methods()
        if method not in variants:
            raise KeyError("unknown STRATA variant %r; choose from %s" % (method, sorted(variants)))
        chat = None
        if getattr(args, "llm", False):
            from strata.llm import Chat

            chat = Chat()
        return StrataAgent(logs, chat=chat, seed=args.seed, n_sims=args.n_sims,
                           **variants[method])
    raise KeyError("unknown method %r; choose from %s" % (method, METHODS))


def _instance_tag(domain: str, laws: Sequence[str], ns: Sequence[int]) -> str:
    law_part = "-".join(str(law) for law in laws)
    n_part = "-".join(str(n) for n in ns)
    return "%s_%s_n%s" % (domain, law_part, n_part)


def build_instances(args) -> List[protocol.Instance]:
    instances = []
    for dataset in args.datasets:
        for n in args.n:
            for law in args.laws:
                instances.append(
                    protocol.make_instance(
                        args.domain,
                        dataset=_normalise_dataset(args.domain, dataset),
                        n=int(n),
                        graph_seed=args.seed,
                        law=law,
                        T=args.T,
                        B=args.budget,
                        sample_method=args.sample_method,
                    )
                )
    return instances


def run(args) -> str:
    out_path = args.out
    if out_path is None:
        tag = args.tag or _instance_tag(args.domain, args.laws, args.n)
        out_path = os.path.join("out", "strata_%s.csv" % tag)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    instances = build_instances(args)
    methods = list(args.methods)
    rows = []

    for instance in instances:
        logs = protocol.make_logs(
            instance,
            policies=args.log_policies,
            n_episodes=args.log_episodes,
            seed=args.seed,
        )
        oracle_policy = model_based.Oracle(
            logs,
            selection_episodes=args.selection_episodes,
            seed=args.seed,
        )
        oracle_values = protocol.evaluate(
            oracle_policy, instance, episodes=args.episodes, seed=args.seed
        )
        oracle_mean = float(np.mean(oracle_values))
        cache: Dict[str, np.ndarray] = {"oracle": oracle_values}

        for method in methods:
            if method == "oracle":
                values = oracle_values
            elif method in cache:
                values = cache[method]
            else:
                policy = _make_policy(method, instance, logs, args)
                if policy is None:
                    reason = (
                        "WS-Option unavailable"
                        if method == "ws-option"
                        else "policy unavailable"
                    )
                    print(
                        "warning: skipping %s (%s)" % (method, reason),
                        file=sys.stderr,
                    )
                    continue
                values = protocol.evaluate(
                    policy, instance, episodes=args.episodes, seed=args.seed
                )
                cache[method] = values

            for ep in range(args.episodes):
                if abs(oracle_mean) > 1e-12:
                    regret = (oracle_values[ep] - values[ep]) / oracle_mean
                else:
                    regret = oracle_values[ep] - values[ep]
                rows.append(
                    {
                        "instance": instance.name,
                        "domain": instance.domain,
                        "dataset": instance.dataset_name,
                        "n": instance.n,
                        "graph_seed": instance.graph_seed,
                        "law": instance.law_name,
                        "T": instance.T,
                        "B": instance.B,
                        "method": method,
                        "episode": ep,
                        "value": float(values[ep]),
                        "oracle_value": float(oracle_values[ep]),
                        "regret": float(regret),
                    }
                )

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", choices=("im", "rpp"), default="im")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n", nargs="+", type=int, default=[100])
    parser.add_argument("--laws", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--log-episodes", type=int, default=100)
    parser.add_argument(
        "--log-policies",
        nargs="+",
        default=None,
        help="behaviour policies for the log (default: domain config)",
    )
    parser.add_argument("--selection-episodes", type=int, default=2)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--sample-method", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--llm", action="store_true",
                        help="give STRATA methods the LLM from .env (otherwise they fall back to the DSL)")
    parser.add_argument("--n-sims", type=int, default=150, help="MCTS simulations per decision")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> str:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.datasets is None:
        args.datasets = [_default_dataset(args.domain)]
    if args.laws is None:
        args.laws = [_default_law(args.domain)]
    path = run(args)
    print(path)
    return path


if __name__ == "__main__":  # pragma: no cover
    main()
