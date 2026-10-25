"""L3 planning: objectives, belief and the rollout planner.

The modules of this package implement the decision layer of STRATA:

* :mod:`strata.planning.objectives` -- scalarisation of a value vector indexed
  by hypotheses (``mean``, ``cvar``, ``maxmin``, ``optimistic``);
* :mod:`strata.planning.belief` -- finite Bayesian belief over the L1
  hypothesis set, with node-level IM and profit-drift RPP likelihoods;
* :mod:`strata.planning.rollout_planner` -- one-step macro lookahead with a
  flat continuation, ported from the verified ``ktb_hier`` kill test.

``MCTSPlanner`` (M3b) lives in :mod:`strata.planning.mcts` and is intentionally
not imported here so that M3a and M3b remain independent.
"""

from strata.planning.belief import Belief
from strata.planning.objectives import aggregate
from strata.planning.rollout_planner import RolloutPlanner, allocation_k

__all__ = ["Belief", "aggregate", "RolloutPlanner", "allocation_k"]
