"""The STRATA agent: L1 hypotheses -> L2 operator library -> L3 belief planner.

``StrataAgent`` implements the policy interface of :mod:`strata.eval.protocol`
(``reset``, ``act``, ``observe``) for both domains.

build (once per instance, from the behaviour logs)
  L1  candidate laws from :func:`strata.laws.llm_propose.propose` (LLM-written
      programs, the enumerated DSL as safety net) -> hypothesis set
      :func:`strata.laws.hypotheses.build_hypothesis_set` (likelihood band plus
      every observational tie, with the pairwise discrimination counts).
  L2  :class:`OperatorLibrary` standard operators, extended by LLM operators
      (:func:`strata.abstraction.llm_operators.synthesize_operators`).
  L3  :class:`MCTSPlanner` (or the one-step :class:`RolloutPlanner`) over
      macros, root-sampling hypotheses from a :class:`Belief`.

per round
  act      the planner picks a macro (operator, region, k); the operator grounds
           it into seeds / a route under the MAP hypothesis.
  observe  Bayes update of the belief with the node-level likelihood of the
           observed round (the low level is the sensor); one diagnosis record.

Configuration flags give the ablations of the paper: law_source,
operator_source, no_hypset, no_belief, planner, mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from strata.abstraction import llm_operators
from strata.abstraction.library import OperatorLibrary
from strata.laws import hypotheses as hyp
from strata.laws import llm_propose
from strata.planning.belief import Belief
from strata.planning.mcts import MCTSPlanner
from strata.planning.rollout_planner import RolloutPlanner


@dataclass
class StrataConfig:
    law_source: str = "llm+dsl"        # llm | dsl | llm+dsl
    operator_source: str = "llm+hand"  # llm | hand | llm+hand
    level_im: int = 2
    level_rpp: int = 3
    val_frac: float = 0.3
    eps: float = 5.0
    cap: int = 8
    planner: str = "mcts"              # mcts | rollout
    mode: str = "mean"                 # mean | cvar | maxmin | optimistic
    alpha: float = 0.2
    n_sims: int = 150
    tree: str = "root"                 # root | full (original open-loop MCTS tree)
    M: int = 8
    no_hypset: bool = False
    true_law: bool = False             # upper bound: plan with the instance's true law only
    no_belief: bool = False
    llm_rounds: int = 2
    llm_k: int = 3
    law_mixture: str = "none"          # none | poe (PoE-World products) | mix (problem-shaped)
    mix_boot: int = 6                  # bootstrap refits tried as alternative mixtures
    decision_sims: int = 60            # MCTS simulations of the first-decision check
    decision_margin: float = 0.01      # relative value gap for an alternative to count
    mix_disagree_round: bool = True    # one LLM round to tell decision-changing mixtures apart
    llm_budget: int = 6
    op_rounds: int = 2                 # round 2 sees the values of round 1's operators
    op_k: int = 3
    op_budget: int = 2
    op_delta: float = 0.5
    diagnose: bool = False             # store predictive reward bands per macro
    band_paths: int = 64
    seed: int = 0


class StrataAgent:
    method = "strata"

    def __init__(self, logs, config: Optional[StrataConfig] = None, chat=None, **overrides):
        self.logs = logs
        self.config = config or StrataConfig()
        for k, v in overrides.items():
            setattr(self.config, k, v)
        self.chat = chat
        self._key = None
        self.report: Dict[str, Any] = {}
        self.records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ build
    def build(self, instance):
        cfg = self.config
        domain = instance.domain
        obj = instance.graph if domain == "im" else instance.env
        logs = list(self.logs)
        n_val = max(1, int(round(cfg.val_frac * len(logs))))
        val, train = logs[:n_val], logs[n_val:] or logs

        level = cfg.level_im if domain == "im" else cfg.level_rpp
        if cfg.true_law:
            cands, law_report = [instance.law], {"source": "true"}
        else:
            cands, law_report = llm_propose.propose(
                domain, train, val, obj, chat=self.chat, source=cfg.law_source, level=level,
                rounds=cfg.llm_rounds, k=cfg.llm_k, budget_calls=cfg.llm_budget, seed=cfg.seed,
                mixture=cfg.law_mixture)
            contexts = law_report.pop("_mix_contexts", None)
            if contexts:
                extra, law_report["mix_alternatives"] = self._decision_alternatives(
                    instance, obj, contexts[0], train, val)
                seen = {id(c) for c in extra}
                cands = extra + [c for c in cands if id(c) not in seen]
        kw = {"indeg": obj.indeg, "graph": obj} if domain == "im" else {"env": obj}
        if cfg.true_law and domain != "im":
            # RPP drift laws have no likelihood (logpdf); with one hypothesis none is needed
            from strata.core.types import Hypothesis, HypothesisSet

            hset = HypothesisSet([Hypothesis(instance.law, 0.0, 1.0, 0)],
                                 np.zeros((1, 1), dtype=np.int64))
        else:
            hset = hyp.build_hypothesis_set(cands, val, eps=cfg.eps, cap=cfg.cap, domain=domain,
                                            **kw)
        if not hset.hyps:
            raise RuntimeError("no hypothesis survived L1 on %s" % instance.name)
        if cfg.no_hypset:
            hset = type(hset)([hset.hyps[0]], np.zeros((1, 1), dtype=np.int64))

        library = OperatorLibrary.standard(domain)
        op_report: Dict[str, Any] = {"source": cfg.operator_source}
        if "llm" in cfg.operator_source:
            if self.chat is None:
                op_report["fallback"] = True
            else:
                library, rep = llm_operators.synthesize_operators(
                    domain, hset, library, obj, self.chat, T=instance.T, budget=instance.B,
                    rounds=cfg.op_rounds, k=cfg.op_k, budget_calls=cfg.op_budget,
                    delta=cfg.op_delta, seed=cfg.seed)
                op_report.update(rep)
                if cfg.operator_source == "llm":
                    base = "hub" if domain == "im" else "harvest"
                    keep = {base, "wait"} | set(rep.get("accepted", []))
                    library = OperatorLibrary([op for op in library.operators if op.name in keep])

        if cfg.planner == "rollout":
            planner = RolloutPlanner(library, M=cfg.M, mode=cfg.mode, alpha=cfg.alpha,
                                     seed=cfg.seed, T=instance.T)
        else:
            planner = MCTSPlanner(library, n_sims=cfg.n_sims, M_leaf=max(1, cfg.M // 2),
                                  mode=cfg.mode, alpha=cfg.alpha, seed=cfg.seed, T=instance.T,
                                  tree=cfg.tree)

        self.domain, self.obj, self.hset, self.library, self.planner = domain, obj, hset, library, planner
        self._key = instance.name
        self.report = {
            "config": asdict(cfg),
            "laws": law_report,
            "hypotheses": [(getattr(h.law, "name", str(h.law)), float(h.loglik), int(h.tie_group))
                           for h in hset.hyps],
            "identifiability": hyp.identifiability_report(hset),
            "operators": op_report,
            "library": library.names(),
            "llm_usage": self.chat.usage() if hasattr(self.chat, "usage") else None,
        }

    # ------------------------------------------------- decision-relevant laws
    def _first_decision(self, instance, obj, law):
        """Root macro and root scores of the planner at the initial state under ``law`` alone."""
        from strata.core.types import Hypothesis, HypothesisSet
        from strata.domains.im import env as im_env

        cfg = self.config
        if hasattr(law, "reset_gate"):
            law.reset_gate()
        planner = MCTSPlanner(OperatorLibrary.standard(instance.domain), n_sims=cfg.decision_sims,
                              M_leaf=max(1, cfg.M // 2), seed=cfg.seed, T=instance.T, tree=cfg.tree)
        belief = Belief(HypothesisSet([Hypothesis(law, 0.0, 1.0, 0)], np.zeros((1, 1), np.int64)),
                        instance.domain)
        state = (im_env.make_state(obj, M=1, budget_left=instance.B) if instance.domain == "im"
                 else obj.reset(M=1))
        macro = planner.choose(obj, state, belief, 0)
        info = planner.last_info
        scores = {(m.op, int(m.region), int(m.k)): float(s)
                  for m, s in zip(info.get("macros", []), np.ravel(info.get("scores", [])))}
        return (macro.op, int(macro.region), int(macro.k)), scores

    def _decision_alternatives(self, instance, obj, ctx, train, val):
        """Alternative mixtures that fit the logs about as well but change the first decision.

        An alternative counts only when, under its own planner, its chosen macro
        beats the best mixture's macro by ``decision_margin`` (relative), so
        near-ties of the planner are not mistaken for different decisions.
        Optionally one LLM round writes experts that tell such mixtures apart,
        after which the mixture and its alternatives are refitted once.
        """
        from strata.laws import mixture

        cfg, domain = self.config, instance.domain

        def changing(ctx):
            base, _ = self._first_decision(instance, obj, ctx.law)
            alts = ctx.alternatives(eps=cfg.eps, n_boot=cfg.mix_boot, seed=cfg.seed)
            keep = {}
            for alt in sorted(alts, key=lambda a: -float(a.ll_val)):
                macro, scores = self._first_decision(instance, obj, alt)
                if macro == base or macro in keep:
                    continue
                own, other = scores.get(macro), scores.get(base)
                if own is not None and other is not None and \
                        own - other <= cfg.decision_margin * max(abs(own), 1e-9):
                    continue
                keep[macro] = alt
            return base, alts, keep

        base, alts, keep = changing(ctx)
        rep: Dict[str, Any] = {"base_macro": list(base), "alternatives": len(alts),
                               "decision_changing": [list(m) for m in keep]}
        laws = [ctx.law]
        if keep and self.chat is not None and cfg.mix_disagree_round:
            table = mixture.disagreement_table(domain, ctx.law, list(keep.values()), train, obj)
            new, call_rep = llm_propose.discriminating_experts(
                domain, train, val, obj, self.chat, table, k=cfg.llm_k, seed=cfg.seed)
            rep["disagreement_round"] = call_rep
            if new:
                contexts, mrep = mixture.build_mixtures(domain, list(ctx.pool.experts) + new,
                                                        train, val, obj, seed=cfg.seed)
                rep["refit"] = mrep
                if contexts and contexts[0].val_ll >= ctx.val_ll:
                    ctx = contexts[0]
                    laws = [ctx.law] + new
                    base, alts, keep = changing(ctx)
                    rep.update({"base_macro_after": list(base), "alternatives_after": len(alts),
                                "decision_changing_after": [list(m) for m in keep]})
        return laws + list(keep.values()), rep

    # -------------------------------------------------------------- policy API
    def reset(self, instance, rng):
        if self._key != instance.name:
            self.build(instance)
        for h in self.hset.hyps:  # per-episode latent state (node gates) starts from the logs
            if hasattr(h.law, "reset_gate"):
                h.law.reset_gate()
        self.belief = Belief(self.hset, self.domain)
        self.records = []

    def act(self, state, t):
        macro = self.planner.choose(self.obj, state, self.belief, t)
        law = self.belief.laws[self.belief.map_index()]
        mask = self.planner.regions[macro.region][1]
        if self.domain == "im":
            k = min(int(macro.k), int(state.budget_left))
        else:
            k = min(int(macro.k), int(np.asarray(state.budget_left).reshape(-1)[0]))
        action = self.planner._operator(macro.op).refine(self.obj, law, state, k, mask)
        self.records.append({"t": int(t), "macro": macro, "weights": self.belief.weights.tolist()})
        return np.asarray(action)[0]

    def observe(self, state_before, action, state_after, info):
        a = np.asarray(action, dtype=np.int64).reshape(1, -1)
        if self.config.diagnose and self.records:
            self.records[-1].update(self._band(state_before, a, state_after, info))
        if not self.config.no_belief:
            self.belief.update(self.obj, state_before, a, state_after, info)
        if self.records:
            self.records[-1]["weights_after"] = self.belief.weights.tolist()

    def _band(self, before, action, after, info):
        """Realised round reward and its 95% predictive band under the MAP law."""
        from strata.domains.im import env as im_env
        from strata.domains.rpp import env as rpp_env
        from strata.planning.rollout_planner import _expand_im, _expand_rpp

        law = self.belief.laws[self.belief.map_index()]
        P = int(self.config.band_paths)
        rng = np.random.default_rng(self.config.seed * 7 + int(before.t))
        if self.domain == "im":
            reward = float(im_env.reward_count(before.status, after.status)[0])
            st = _expand_im(before, P)
            _, sim, _ = im_env.step(self.obj, law, st, action[0], rng)
        else:
            reward = float(np.asarray(info["base_reward"])[0] + np.asarray(info["penalty"])[0])
            st = _expand_rpp(before, P)
            _, sim, _, _ = rpp_env.step(self.obj, st, np.repeat(action, P, axis=0), law=law,
                                        rng=rng, return_info=True)
        lo, hi = np.percentile(np.asarray(sim, dtype=np.float64), [2.5, 97.5])
        return {"reward": reward, "band": (float(lo), float(hi))}

    def diagnose(self, episodes):
        """Attribute the recorded macros of the last episode (needs ``diagnose=True``)."""
        from strata.diagnosis.attribution import attribute
        from strata.diagnosis.calibration import calibration

        law = self.belief.laws[self.belief.map_index()]
        cal = calibration(self.domain, law, episodes, self.obj)
        rows, summary = attribute(self.records, cal["p_value"])
        return {"calibration": cal, "rows": rows, "summary": summary}


def make_strata_methods():
    """Method name -> config overrides, for strata.eval.run (main method + ablations)."""
    return {
        "strata": {},
        "strata-dsl": {"law_source": "dsl", "operator_source": "hand"},
        "strata-map": {"no_hypset": True},
        "strata-nobelief": {"no_belief": True},
        "strata-rollout": {"planner": "rollout"},
        "strata-cvar": {"mode": "cvar"},
        "strata-maxmin": {"mode": "maxmin"},
        # a single hypothesis needs no belief updates (RPP drift laws have no logpdf)
        "strata-true": {"true_law": True, "operator_source": "hand", "no_belief": True},
        "strata-poe": {"law_mixture": "poe", "llm_k": 5},
        "strata-mix": {"law_mixture": "mix", "llm_k": 5},
        "strata-fulltree": {"tree": "full"},
        "strata-true-fulltree": {"true_law": True, "operator_source": "hand", "tree": "full"},
    }
