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
    M: int = 8
    no_hypset: bool = False
    no_belief: bool = False
    llm_rounds: int = 2
    llm_k: int = 3
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
        cands, law_report = llm_propose.propose(
            domain, train, val, obj, chat=self.chat, source=cfg.law_source, level=level,
            rounds=cfg.llm_rounds, k=cfg.llm_k, budget_calls=cfg.llm_budget, seed=cfg.seed)
        kw = {"indeg": obj.indeg, "graph": obj} if domain == "im" else {"env": obj}
        hset = hyp.build_hypothesis_set(cands, val, eps=cfg.eps, cap=cfg.cap, domain=domain, **kw)
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
                                  mode=cfg.mode, alpha=cfg.alpha, seed=cfg.seed, T=instance.T)

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

    # -------------------------------------------------------------- policy API
    def reset(self, instance, rng):
        if self._key != instance.name:
            self.build(instance)
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
    }
