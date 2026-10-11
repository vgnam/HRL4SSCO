"""Genetic-algorithm planner for DSR (the RPP baseline of HRL4SSCO, adapted).

Like GA and greedy in RPP, it plans with full knowledge of the law: here the true compatibility
relation (and quorums). Every day it evolves the repair set (at most `room` down producers) that
maximises the customer weight served after the repairs, assuming they succeed, plus a small term
for upstream nodes that operate weighted by their downstream reach (B1's weights), so that partial
progress towards restoring a customer is rewarded.
"""
import numpy as np

from ..envs.dsr import Episode, operating


def reach_weights(net):
    reach = net.w.copy()
    for i in (2, 1, 0):
        add = np.zeros(net.n)
        for v in net.idx[i + 1]:
            for links, _ in net.slots[int(v)]:
                add[links] += reach[v] / max(len(links), 1)
        reach[net.idx[i]] = add[net.idx[i]]
    return reach


class DailyGA:
    def __init__(self, net, compat, quorum=None, pop=40, gens=30, p_mut=0.3, eps=0.05, seed=0):
        self.net, self.compat, self.quorum = net, compat, quorum
        self.pop, self.gens, self.p_mut, self.eps = pop, gens, p_mut, eps
        self.rng = np.random.default_rng(seed)
        r = reach_weights(net)
        self.shape = np.where(net.layer < 3, r, 0.0) * eps / max(r[net.layer < 3].sum(), 1e-9) * net.w.sum()

    def fitness(self, up, sol):
        u = up.copy()
        u[list(sol)] = True
        op = operating(self.net, u, self.compat, self.quorum)
        return float(self.net.w[op].sum() + self.shape[op].sum())

    def plan(self, up, room):
        cand = np.where((self.net.layer < 3) & ~up)[0]
        if room <= 0 or len(cand) == 0:
            return []
        k = min(room, len(cand))
        rng = self.rng
        P = [frozenset(rng.choice(cand, k, replace=False).tolist()) for _ in range(self.pop)]
        cache = {}

        def fit(s):
            if s not in cache:
                cache[s] = self.fitness(up, s)
            return cache[s]
        for _ in range(self.gens):
            f = np.array([fit(s) for s in P])
            elite = P[int(np.argmax(f))]
            new = [elite]
            while len(new) < self.pop:
                a, b = (P[max(rng.choice(len(P), 2, replace=False), key=lambda i: f[i])] for _ in range(2))
                pool = list(a | b)
                child = set(rng.choice(pool, min(k, len(pool)), replace=False).tolist())
                if rng.random() < self.p_mut and child:            # swap one repair for another down producer
                    child.discard(rng.choice(list(child)))
                    rest = [int(x) for x in cand if x not in child]
                    if rest:
                        child.add(int(rng.choice(rest)))
                while len(child) < k:
                    child.add(int(rng.choice(cand)))
                new.append(frozenset(child))
            P = new
        best = max(P, key=fit)
        # drop repairs that do not change the fitness (save budget for later days)
        best = set(best)
        for u in sorted(best):
            if fit(frozenset(best - {u})) >= fit(frozenset(best)):
                best.discard(u)
        return sorted(int(x) for x in best)


def run_ga(net, seed, K=20, cap=4, **env):
    ep = Episode(net, seed, K, cap, **env)
    ga = DailyGA(net, net.true_compat(), net.true_quorum() if net.quorum is not None else None, seed=seed)
    while not ep.done:
        up, _, B, _ = ep.observe()
        ep.step(ga.plan(up, min(cap, B)))
    return ep.total
