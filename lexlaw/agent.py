"""The LexLaw agent loop (method Section 6) and episode runners shared by the experiments."""
import numpy as np

from .envs.dsr import Episode
from .model.rates import Rates
from .plan.restore import restore_greedy


def run_episode(net, belief, seed, K=20, cap=4, T=15, em=True, schedule=None, budget=None, rates=None,
                rho=0.9, phi=0.01, aftershock=None, model=None):
    """One episode. belief=None plans with `model` (default: the true relation), i.e. the reference
    planner pi deg. `schedule` is a static callable (t, B) -> room; `budget` a rollout chooser
    (plan.budget.BudgetPlanner) that overrides it. Returns the episode's return."""
    ep = Episode(net, seed, K, cap, T=T, rho=rho, phi=phi, aftershock=aftershock)
    if belief is None and model is None:
        model = (net.true_compat(), net.true_quorum() if net.quorum is not None else None)
    rates = rates if rates is not None else Rates()
    prev = None
    if budget is not None:
        budget.reset(ep)
    while not ep.done:
        up, op, B, t = ep.observe()
        shock = budget.observe(t, up, rates) if budget is not None else bool(aftershock and t == aftershock[0])
        if prev is not None:
            rates.update(net, prev[0], prev[1], up, shock_between=shock)
        if belief is not None:
            belief.observe(up, op)
            if em:
                belief.em()
            compat, quorum = belief.map_model()
        else:
            compat, quorum = model
        if budget is not None:
            room = budget.room(t, B, up, op, compat, quorum, rates)
        elif schedule is not None:
            room = schedule(t, B)
        else:
            room = min(cap, B)
        rep = restore_greedy(net, up, compat, room, quorum)
        prev = (up, rep)
        ep.step(rep)
    up, op, _, _ = ep.observe()
    if belief is not None:
        belief.observe(up, op)                              # final observation: learning only, no EM
    return ep.total


def gap(values, reference):
    """Paired relative gap in percent and its standard error (over the pairs)."""
    d = np.asarray(values, float) - np.asarray(reference, float)
    o = np.mean(reference)
    se = np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return 100 * d.mean() / o, 100 * se / o
