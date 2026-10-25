"""L2 of STRATA, main path: the LLM writes new macro-operators.

An operator is only a *scoring rule*; the grounding loop stays in the
framework:

* IM: ``objective(f, state, region_mask, chosen) -> (M, n)`` is the gain of
  adding each node given the seeds already chosen this round; refine runs
  greedy selection exactly like ``Hub`` (recompute features after each pick,
  restrict to the region and to nodes that are not active).
* RPP: ``score(env, state, region_mask) -> (M, n)`` ranks POIs; refine takes
  the top-k unvisited POIs of the region and routes them like ``Harvest``
  (nearest neighbour + 2-opt + trimming to the daily distance).

``synthesize_operators`` shows the LLM the hypotheses of L1 and a table of the
existing operators' values under every hypothesis, loads the answers through
:mod:`strata.llm.synthesis`, checks them numerically and keeps an operator
only if, used as a flat policy, it raises the best value under at least one
hypothesis by more than ``delta``. LLM operators serialise with their source
code, so a library round-trips through ``OperatorLibrary`` JSON.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import numpy as np

from strata.abstraction import library as library_mod
from strata.abstraction import operators as ops_mod
from strata.core.types import STATUS_ACTIVE
from strata.domains.im import env as im_env
from strata.domains.rpp import env as rpp_env
from strata.llm.synthesis import SynthesisError, extract_code_blocks, load_definitions

IM_OPERATOR_TEMPLATE = '''```python
NAME = "short-name"   # e.g. "shared-3"

def objective(f, state, region_mask, chosen, graph, prob):
    # f: Features of the current partial seed set (seeds already chosen this round count as active):
    #    f.a (M, n) active in-neighbours, f.E (M, n) exposure, f.eligible (M, n) bool
    #    (inactive and not chosen), f.indeg (n,), f.community (n,)
    # state.status (M, n) in {0 inactive, 1 active, 2 removed}; region_mask (n,) bool;
    # chosen (M, n) bool seeds already picked this round.
    # graph.adj: scipy CSR (n, n) with adj[u, v] = 1 for an edge u -> v; graph.adjT its transpose,
    #    so X @ graph.adjT sums a per-target quantity X (M, n) over each node's out-neighbours.
    # prob(a): (M, n) activation probability of every node under the hypothesis being planned
    #    for, if it had `a` active in-neighbours (prob(f.a) now, prob(f.a + 1) with one more).
    #    Use it instead of hard-coding law parameters: the planner calls the operator under
    #    every hypothesis.
    # Return the (M, n) gain of adding each node as the next seed (higher = better).
    return np.zeros_like(f.a)
```'''

# The built-in operators, shown to the LLM in the same signature so it can build on them.
IM_REFERENCE_OPERATORS = '''```python
def objective(f, state, region_mask, chosen, graph, prob):   # "hub": expected one-round gain
    P0, P1 = prob(f.a), prob(f.a + 1)
    D = np.where(f.eligible, P1 - P0, 0.0)          # marginal activation of each target
    return D @ graph.adjT + f.eligible * (1.0 - P0)  # spread to out-neighbours + the seed itself

def objective(f, state, region_mask, chosen, graph, prob):   # "gate-2" (gate-3: 2 -> 3)
    comp = f.eligible & (f.a == 1)                   # one more active in-neighbour completes the gate
    prog = f.eligible & (f.a < 1)
    return (comp + 0.3 * prog).astype(float) @ graph.adjT + 1e-3 * f.eligible
```
"hedge-g" alternates hub and gate-g picks; "prime" seeds towards maximal exposure (only for laws
with exposure memory); "wait" seeds nothing this round.'''

IM_RULES = (
    "Rules of the environment: each round the agent seeds k nodes; the reward of a round is the "
    "number of nodes that become active in it, seeds included. Seeds and the previous round's "
    "newly active nodes are the active in-neighbours that drive this round's activations; active "
    "nodes are removed after one round. In round 0 nothing is active yet, so f.a is zero "
    "everywhere. Selection is greedy: argmax of your gain, recomputed after each pick, so a gain "
    "that is constant over the candidates picks the lowest node ids -- always keep a non-zero "
    "reach term (as hub does) to break ties sensibly."
)

RPP_OPERATOR_TEMPLATE = '''```python
NAME = "short-name"

def score(env, state, region_mask):
    # state.profit (M, n), state.visited (M, n) bool, state.current (M,), state.t (1-based day);
    # env.dist (n, n), env.loc (n, 2), env.initial_profit (n,), env.T; region_mask (n,) bool.
    # Return the (M, n) priority of visiting each POI today (higher = better).
    return state.profit.copy()
```'''


def _compile(source, names):
    ns = load_definitions(source, names)
    return ns


class LLMObjectiveOperator:
    """IM operator whose greedy gain is an LLM-written objective."""

    kind = "llm"

    def __init__(self, name="llm", source="", objective=None):
        self.name = str(name)
        self.source = source
        self.params = {"name": self.name, "source": source}
        self.objective = objective if objective is not None else _compile(source, ["objective"])["objective"]
        # Objectives written against the older five-argument template do not take ``prob``.
        try:
            self._takes_prob = len(inspect.signature(self.objective).parameters) >= 6
        except (TypeError, ValueError):
            self._takes_prob = False
        self.last = {}

    def precondition(self, z, k, region):
        return int(k) > 0

    def refine(self, graph, law, state, k, region_mask):
        status = np.asarray(state.status)
        M, n = status.shape
        k = int(max(0, min(int(k), n)))
        seeds = -np.ones((M, k), dtype=np.int64)
        chosen = np.zeros((M, n), dtype=bool)
        rows = np.arange(M)
        region_mask = np.asarray(region_mask, dtype=bool)
        for j in range(k):
            tmp = status.copy()
            tmp[chosen] = STATUS_ACTIVE
            feats = im_env.compute_features(graph, law, tmp, state.E)
            args = (feats, state, region_mask, chosen, graph)
            if self._takes_prob:
                eligible = feats.eligible

                def prob(a, eligible=eligible):
                    a = np.asarray(a, dtype=np.float32)
                    return np.asarray(ops_mod._im_prob(
                        law, ops_mod._accumulate(law, state.E, a), a, graph, eligible),
                        dtype=np.float64)

                args += (prob,)
            gain = np.asarray(self.objective(*args), dtype=np.float64)
            gain = np.broadcast_to(gain, (M, n))
            cand = (status != STATUS_ACTIVE) & ~chosen & region_mask[None, :]
            gain = np.where(cand & np.isfinite(gain), gain, -np.inf)
            best = np.argmax(gain, axis=1)
            ok = np.isfinite(gain[rows, best])
            seeds[ok, j] = best[ok]
            chosen[rows[ok], best[ok]] = True
        self.last = {"seeds": seeds}
        return seeds

    def checker(self, before, after, region_mask):
        new = (np.asarray(before.status) == 0) & (np.asarray(after.status) != 0)
        return (new & np.asarray(region_mask, dtype=bool)[None, :]).any(axis=1)


class LLMScoreOperator(ops_mod._RPPRoute):
    """RPP operator whose POI ranking is an LLM-written score."""

    kind = "llm"

    def __init__(self, name="llm", source="", score=None):
        super().__init__()
        self.name = str(name)
        self.source = source
        self.params = {"name": self.name, "source": source}
        self.score = score if score is not None else _compile(source, ["score"])["score"]
        self._env = None

    def refine(self, env, law, state, k, region_mask):
        self._env = env
        self._scores = np.broadcast_to(
            np.asarray(self.score(env, state, np.asarray(region_mask, bool)), dtype=np.float64),
            state.profit.shape)
        return super().refine(env, law, state, k, region_mask)

    def _select(self, env, state, path, region_mask, k):
        avail = np.flatnonzero(ops_mod._rpp_available(state, path, region_mask))
        if avail.size == 0 or k <= 0:
            return np.zeros(0, dtype=np.int64)
        s = self._scores[path, avail]
        order = np.argsort(-s, kind="stable")[:k]
        return avail[order].astype(np.int64)


# serialisation: the library finds the classes by name and rebuilds them from source
library_mod._REGISTRY["LLMObjectiveOperator"] = LLMObjectiveOperator
library_mod._REGISTRY["LLMScoreOperator"] = LLMScoreOperator


def operator_from_answer(domain, code):
    """Build an operator from one code block of an LLM answer."""
    fn = "objective" if domain == "im" else "score"
    ns = _compile(code, [fn])
    name = "llm-" + str(ns.get("NAME", "op")).strip().replace(" ", "-")[:24]
    if domain == "im":
        return LLMObjectiveOperator(name, code, ns["objective"])
    return LLMScoreOperator(name, code, ns["score"])


# =========================================================================
# Evaluation of operators as flat policies under each hypothesis
# =========================================================================


def flat_value(domain, op, law, obj, T, budget, episodes=2, seed=0):
    """Mean total reward of ``op`` used with the even schedule on the whole region."""
    vals = []
    for ep in range(episodes):
        rng = np.random.default_rng(seed * 1000 + ep)
        if domain == "im":
            n = obj.adj.shape[0]
            state = im_env.make_state(obj, M=1, budget_left=budget)
            mask = np.ones(n, dtype=bool)
            total, rem = 0.0, int(budget)
            for t in range(T):
                k = rem if t == T - 1 else min(rem, int(np.ceil(rem / (T - t))))
                action = op.refine(obj, law, state, k, mask)
                _, r, _ = im_env.step(obj, law, state, action, rng)
                total += float(r[0])
                rem -= int((np.asarray(action) >= 0).sum())
            vals.append(total)
        else:
            state = obj.reset(M=1)
            mask = np.ones(obj.n, dtype=bool)
            total = 0.0
            while int(state.t) <= T:
                rem = int(state.budget_left[0])
                left = T - int(state.t) + 1
                k = rem if left <= 1 else min(rem, int(np.ceil(rem / left)))
                action = op.refine(obj, law, state, k, mask)
                _, r, done, _ = rpp_env.step(obj, state, action, law=law, rng=rng, return_info=True)
                total += float(r[0])
                if done:
                    break
            vals.append(total)
    return float(np.mean(vals))


def value_table(domain, library, laws, obj, T, budget, episodes=2, seed=0):
    names, rows = [], []
    for op in library.operators:
        if op.name == "wait":
            continue
        names.append(op.name)
        rows.append([flat_value(domain, op, law, obj, T, budget, episodes, seed) for law in laws])
    return names, np.asarray(rows, dtype=np.float64)


def _describe_law(law):
    return getattr(law, "source", "") or getattr(law, "name", type(law).__name__)


def operator_context(domain, laws, library, obj, T, budget, episodes=2, seed=0):
    names, table = value_table(domain, library, laws, obj, T, budget, episodes, seed)
    lines = ["Hypotheses about the hidden law (the planner considers all of them):"]
    for i, law in enumerate(laws):
        lines.append("- H%d: %s" % (i, _describe_law(law)))
    lines.append("")
    lines.append("Value of each existing operator used every round (even budget schedule), per hypothesis:")
    lines.append("| operator | " + " | ".join("H%d" % i for i in range(len(laws))) + " |")
    lines.append("|---|" + "---|" * len(laws))
    for name, row in zip(names, table):
        lines.append("| %s | %s |" % (name, " | ".join("%.2f" % v for v in row)))
    return "\n".join(lines), names, table


def _feedback_lines(candidates, delta):
    lines = ["Results of your previous operators (value per hypothesis, gain over the best existing "
             "operator; accepted iff some gain > %.2f):" % delta]
    for c in candidates:
        lines.append("- %s: values %s, gain %s -> %s" % (
            c["name"], ", ".join("%.1f" % v for v in c["values"]),
            ", ".join("%+.1f" % g for g in c["gain"]), "accepted" if c["accepted"] else "rejected"))
    lines.append("A large negative gain usually means the gain is flat or zero for most candidates "
                 "(ties pick the lowest node ids) or that it ignores the seeds' own activation.")
    return lines


def _prompt(domain, context, k, errors=None, feedback=None, delta=0.5):
    template = IM_OPERATOR_TEMPLATE if domain == "im" else RPP_OPERATOR_TEMPLATE
    task = ("choose a set of seed nodes each round in influence maximisation"
            if domain == "im" else "choose which POIs to visit each day in route planning with drifting profits")
    lines = [
        "You design macro-operators for a planner that must %s." % task,
        "An operator is a scoring rule; the framework does the greedy selection / routing.",
    ]
    if domain == "im":
        lines += [IM_RULES, "", "The built-in operators, in the same signature:", IM_REFERENCE_OPERATORS]
    lines += ["", context, "",
              "An operator is kept only if, played every round with the even budget schedule, it "
              "beats the best value in the table by more than %.2f under at least one hypothesis. "
              "Aim at what the existing operators miss for a specific hypothesis (e.g. use prob to "
              "trade spread now against nodes left one step from activation), rather than "
              "re-deriving hub." % delta, ""]
    if feedback:
        lines += list(feedback) + [""]
    if errors:
        lines += ["Your previous operators failed with these errors; fix them:"] + ["- " + e for e in errors] + [""]
    lines += [
        "Write %d new operators that would beat the existing ones under at least one hypothesis, "
        "each as its own ```python block following exactly this template (numpy as np, math as math; "
        "no other imports, no I/O, vectorised over M):" % k,
        template,
    ]
    return [{"role": "user", "content": "\n".join(lines)}]


def _check(domain, op, obj, law):
    if domain == "im":
        state = im_env.make_state(obj, M=2, budget_left=4)
        seeds = op.refine(obj, law, state, 3, np.ones(obj.adj.shape[0], bool))
        if seeds.shape != (2, 3):
            raise SynthesisError("refine produced a wrong shape")
    else:
        state = obj.reset(M=2)
        route = op.refine(obj, law, state, 3, np.ones(obj.n, bool))
        if route.shape[0] != 2:
            raise SynthesisError("refine produced a wrong shape")


def synthesize_operators(domain, hyps, library, obj, chat, T, budget, rounds=1, k=3,
                         budget_calls=3, delta=0.5, episodes=2, seed=0):
    """Extend ``library`` with accepted LLM operators; returns ``(library, report)``."""
    domain = str(domain).lower()
    laws = [getattr(h, "law", h) for h in getattr(hyps, "hyps", hyps)]
    context, names, table = operator_context(domain, laws, library, obj, T, budget, episodes, seed)
    best = table.max(axis=0) if table.size else np.full(len(laws), -np.inf)
    lib = library_mod.OperatorLibrary(list(library.operators))
    report: Dict[str, Any] = {"calls": 0, "candidates": [], "accepted": []}
    errors: List[str] = []
    feedback = None
    for r in range(int(rounds)):
        if report["calls"] >= budget_calls:
            break
        answer = chat(_prompt(domain, context, k, errors or None, feedback, delta),
                      sample=report["calls"])
        report["calls"] += 1
        errors = []
        first = len(report["candidates"])
        for code in extract_code_blocks(answer):
            try:
                op = operator_from_answer(domain, code)
                _check(domain, op, obj, laws[0])
                vals = np.asarray([flat_value(domain, op, law, obj, T, budget, episodes, seed)
                                   for law in laws])
            except Exception as e:  # noqa: BLE001 - reported back to the LLM
                errors.append("%s: %s" % (type(e).__name__, str(e)[:200]))
                continue
            gain = vals - best
            keep = bool(np.any(gain > delta))
            report["candidates"].append({"name": op.name, "values": vals.tolist(),
                                         "gain": gain.tolist(), "accepted": keep, "source": code})
            if keep and op.name not in lib.names():
                lib.add(op)
                best = np.maximum(best, vals)
                report["accepted"].append(op.name)
        report.setdefault("errors", []).extend(errors)
        round_candidates = report["candidates"][first:]
        feedback = _feedback_lines(round_candidates, delta) if round_candidates else None
    return lib, report
