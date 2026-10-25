"""KT-L0 stage 2: do the fitted Higgs laws lead to different seeding decisions?

Seeds are placed at hour h0 on a follower subgraph and the laws fitted by ktl0_higgs.py (background included)
run for HZ hours. The score of a plan is the number of adoptions it adds over placing no seed (incremental),
so the external news background, which seeding cannot change, does not hide the differences between laws.
Plans: CELF greedy under each law (Monte Carlo, common random numbers) and the follower-count heuristic.
"""
import os, sys, time
import numpy as np, scipy.sparse as sp
from scipy.special import expit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ktl0_higgs import load, RAW, CAP_EVER

K, HZ, M_PLAN, M_EVAL, N_SUB, N_CAND = 20, 24, 16, 200, 5000, 200
MODELS = ['M1', 'M2', 'M3', 'M5']


def subgraph(F, n, rng):
    """Random walk with restart on the undirected follower graph, N_SUB users."""
    U = (F + F.T).tocsr()
    start = rng.integers(n)
    while U.indptr[start + 1] == U.indptr[start]: start = rng.integers(n)
    seen, cur = {start}, start
    while len(seen) < N_SUB:
        if rng.random() < 0.15: cur = start
        nb = U.indices[U.indptr[cur]:U.indptr[cur + 1]]
        cur = int(nb[rng.integers(len(nb))]); seen.add(cur)
    idx = np.array(sorted(seen))
    return F[idx][:, idx].tocsr(), idx


class Law:
    def __init__(self, model, w):
        self.model, self.lam, self.w = model, expit(w[:168]), w[168:]

    def survival(self, a_new, a_ever):
        w = self.w
        if self.model == 'M1': return (1 - expit(w[0])) ** a_new
        if self.model == 'M2': return (1 - expit(w[0])) ** a_ever
        if self.model == 'M3':
            g = np.r_[0, expit(w[:CAP_EVER])]
            return 1 - g[np.minimum(a_ever, CAP_EVER).astype(int)]
        if self.model == 'M5': return (1 - expit(w[0])) ** a_new * (1 - expit(w[1])) ** (a_ever - a_new)


def rollout(law, Fs, ever0, seeds, h0, M, seed):
    """Total adopters after HZ hours for M rollouts at once (columns), seeds adopt at h0."""
    rng = np.random.default_rng(seed); n = Fs.shape[0]
    ever = np.repeat(ever0[:, None], M, 1); new = np.zeros((n, M), bool)
    ever[seeds] = True; new[seeds] = True
    for h in range(h0, h0 + HZ):
        a_new = Fs @ new.astype(np.float32); a_ever = Fs @ ever.astype(np.float32)
        p = 1 - (1 - law.lam[min(h, 167)]) * law.survival(a_new, a_ever)
        new = (rng.random((n, M)) < p) & ~ever; ever |= new
    return ever.sum(0).mean()


def celf(law, Fs, ever0, cands, h0):
    base = rollout(law, Fs, ever0, [], h0, M_PLAN, 0)
    gains = [(rollout(law, Fs, ever0, [v], h0, M_PLAN, 0) - base, v, 0) for v in cands]
    seeds, cur = [], base
    while len(seeds) < K:
        gains.sort(key=lambda x: -x[0]); g, v, it = gains.pop(0)
        if it == len(seeds):
            seeds.append(v); cur += g
        else:
            gains.append((rollout(law, Fs, ever0, seeds + [v], h0, M_PLAN, 0) - cur, v, len(seeds)))
    return seeds


if __name__ == '__main__':
    t0 = time.time(); F, adopt, atype, n = load()
    fits = np.load(os.path.join(RAW, 'fits.npz')); laws = {m: Law(m, fits[m]) for m in MODELS}
    for h0, label in [(24, 'quiet (day 2)'), (78, 'announcement (July 4)')]:
        for rep in range(2):
            rng = np.random.default_rng(100 * h0 + rep)
            Fs, idx = subgraph(F, n, rng); ever0 = adopt[idx] < h0
            followers = np.asarray(Fs.sum(0)).ravel()  # users following v inside the subgraph
            cands = [int(v) for v in np.argsort(-followers) if not ever0[v]][:N_CAND]
            plans = {m: celf(laws[m], Fs, ever0, cands, h0) for m in MODELS}
            plans['followers'] = cands[:K]
            inc = {}
            for truth in MODELS:
                base = rollout(laws[truth], Fs, ever0, [], h0, M_EVAL, 1)
                inc[truth] = {p: rollout(laws[truth], Fs, ever0, s, h0, M_EVAL, 1) - base for p, s in plans.items()}
            print(f"\n{label} rep {rep}: subgraph {N_SUB} users, {Fs.nnz} follow edges, already adopted {ever0.sum()}  [{time.time()-t0:.0f}s]")
            print("incremental adoptions (rows: true law, cols: plan); K =", K)
            print("        " + ''.join(f"{p:>16s}" for p in plans))
            for truth in MODELS:
                best = inc[truth][truth]
                cells = [f"{inc[truth][p]:.1f} ({100 * (inc[truth][p] / best - 1):+.0f}%)" for p in plans]
                print(f"{truth:8s}" + ''.join(f"{c:>16s}" for c in cells), flush=True)
