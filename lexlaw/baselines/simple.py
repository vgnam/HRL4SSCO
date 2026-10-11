"""Law-free and fixed-model baselines: B0 (reference planner), B1 (plan_reach), noisy models."""
import numpy as np

from ..envs.dsr import Episode
from ..plan.restore import restore_greedy


def plan_reach(net, up, room):
    """B1: propagate customer weight upstream, splitting a consumer's weight equally among the links
    of each slot, and repair the down nodes with the largest total (Section 12)."""
    reach = net.w.copy()
    for i in (2, 1, 0):
        add = np.zeros(net.n)
        for v in net.idx[i + 1]:
            for links, _ in net.slots[int(v)]:
                add[links] += reach[v] / max(len(links), 1)
        reach[net.idx[i]] = add[net.idx[i]]
    down = np.where(~up)[0]
    return [int(v) for v in down[np.argsort(-reach[down])][:room]]


def schedule_room(schedule, t, B, cap, T):
    """Room of a static schedule; `schedule` is a callable (t, B) -> room, or None for asap."""
    return min(cap, B) if schedule is None else schedule(t, B)


def run_fixed(net, seed, K=20, cap=4, compat=None, quorum=None, policy="model", schedule=None, **env):
    """One episode with a fixed model (B0 when compat is the truth) or the law-free heuristic B1."""
    ep = Episode(net, seed, K, cap, **env)
    if compat is None:
        compat = net.true_compat()
    if quorum is None and net.quorum is not None:
        quorum = net.true_quorum()
    while not ep.done:
        up, _, B, t = ep.observe()
        room = schedule_room(schedule, t, B, cap, ep.T)
        if policy == "reach":
            rep = plan_reach(net, up, room)
        else:
            rep = restore_greedy(net, up, compat, room, quorum)
        ep.step(rep)
    return ep.total, K - ep.B
