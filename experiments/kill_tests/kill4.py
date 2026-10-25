"""Kill test 4: identification is hard. Mechanisms: S = IC weighted cascade (as before);
H = weak simple part (p = PS/indeg per new active in-neighbour) + complex boost (activate with prob Q per step once
>= 2 in-neighbours were ever active). Spread seeds look alike under S and H; only seeds sharing an out-neighbour
separate them. Prior 1/2-1/2, full observation, one episode."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, time, numpy as np
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'IM'))
from kt_common import *
from utils.datasets import load_dataset, sample_subgraph

K, T, R, E, N, REAL = 20, 10, 20, 8, 200, 3
PS, Q = 1.0, 0.5

class HWorld:
    def __init__(self, A, rng):
        self.AT = A.T.tocsr(); self.indeg = np.maximum(np.diff(self.AT.indptr), 1); self.rng = rng
        n = A.shape[0]; self.ever = np.zeros(n, bool); self.new = np.zeros(n, bool)
    def step(self, seeds):
        seeds = [s for s in seeds if not self.ever[s]]
        self.ever[seeds] = True; self.new[seeds] = True
        a_new = self.AT @ self.new.astype(float); a_ever = self.AT @ self.ever.astype(float)
        p = 1 - (1 - PS / self.indeg) ** a_new * (1 - Q * (a_ever >= 2))
        act = (self.rng.random(len(p)) < p) & ~self.ever
        self.ever |= act; self.new = act
        return len(seeds) + int(act.sum()), np.flatnonzero(act)

def h_rollouts(A, ever, new, seeds, steps, M, seed):
    """Mean activations of placing `seeds` now from H state (ever, new), common random numbers via `seed`."""
    rng = np.random.default_rng(seed); tot = 0
    for _ in range(M):
        w = HWorld(A, rng); w.ever = ever.copy(); w.new = new.copy()
        for t in range(steps): tot += w.step(seeds if t == 0 else [])[0]
    return tot / M

def h_greedy(A, P, ever, new, k, h, rng, M=10, C=30):
    """Greedy under H by Monte Carlo, candidates: best by simple-contagion gain + nodes that close a pair."""
    if k <= 0: return []
    n = A.shape[0]; AT = A.T.tocsr(); outdeg = np.asarray(A.sum(1)).ravel()
    status = np.where(ever, 2, 0).astype(np.int8); status[new] = 1
    s_order = simple_greedy(P, status, min(C, int((~ever).sum())), h, 5, rng)
    seeds, base = [], h_rollouts(A, ever, new, [], h, M, 0)
    for _ in range(k):
        cur = ever.copy(); cur[seeds] = True
        touched = np.asarray(A[cur].sum(0)).ravel() > 0  # out-neighbours of the cluster
        closers = np.flatnonzero((np.asarray(A[:, touched].sum(1)).ravel() > 0) & ~cur)
        closers = closers[np.argsort(-outdeg[closers])][:C]
        cands = [v for v in dict.fromkeys(list(s_order) + list(closers)) if not cur[v]]
        vals = {v: h_rollouts(A, ever, new, seeds + [int(v)], h, M, 0) for v in cands}
        v = max(vals, key=vals.get); seeds.append(int(v)); base = vals[v]
    return seeds

def h_value(A, seeds, rng, M=20):
    tot = 0
    for _ in range(M):
        w = HWorld(A, rng)
        for t in range(T): tot += w.step(seeds if t == 0 else [])[0]
    return tot / M

def lik(A, seeds, activated):
    """Likelihood of the step-0 activations under S and under H."""
    n = A.shape[0]; AT = A.T.tocsr(); indeg = np.maximum(np.diff(AT.indptr), 1)
    s = np.zeros(n, bool); s[seeds] = True; a = AT @ s.astype(float)
    act = np.zeros(n, bool); act[activated] = True; m = ~s
    pS = 1 - (1 - 1 / indeg) ** a
    pH = 1 - (1 - PS / indeg) ** a * (1 - Q * (a >= 2))
    l = lambda p: np.exp(np.sum(np.log(np.clip(np.where(act[m], p[m], 1 - p[m]), 1e-300, 1))))
    return l(pS), l(pH)

def probe(A, k=5):
    """Pairs of seeds sharing an out-neighbour of large in-degree (where S and H disagree most), then fill."""
    AT = A.T.tocsr(); indeg = np.diff(AT.indptr); outdeg = np.asarray(A.sum(1)).ravel(); seeds = []
    for w in np.argsort(-indeg):
        ins = [u for u in AT.indices[AT.indptr[w]:AT.indptr[w + 1]] if u not in seeds]
        if len(ins) >= 2 and len(seeds) + 2 <= k:
            seeds += [int(u) for u in sorted(ins, key=lambda u: -outdeg[u])[:2]]
        if len(seeds) + 2 > k: break
    for u in np.argsort(-outdeg):
        if len(seeds) == k: break
        if int(u) not in seeds: seeds.append(int(u))
    return seeds

def run(mech, A, P, first, rest_fn, seed):
    """Seeds `first` at step 0, then rest_fn(obs, state) -> seeds at step 1."""
    rng = np.random.default_rng(seed); n = A.shape[0]; total = 0
    if mech == 'S':
        st = SimpleState(n); L = sample_live(P, rng); r, new = simple_step(st, first, L); total += r
        rest = rest_fn(np.flatnonzero(new), st.status > 0, st.status)
        for t in range(1, T): total += simple_step(st, rest if t == 1 else [], L)[0]
    else:
        w = HWorld(A, rng); r, act = w.step(first); total += r
        status = np.where(w.ever, 2, 0).astype(np.int8); status[act] = 1
        rest = rest_fn(act, w.ever.copy(), status)
        for t in range(1, T): total += w.step(rest if t == 1 else [])[0]
    return total

def plan_for(m, A, P, ever, status, k, h, rng):
    if k <= 0: return []
    return simple_greedy(P, status, k, h, R, rng) if m == 'S' else h_greedy(A, P, ever, status == 1, k, h, rng)

def evaluate(name):
    out, ident = {}, {'probe': [], 'split5': []}
    for e in range(E):
        g = sample_subgraph(load_dataset(name), N, seed=e); A, P = graph_arrays(g); rng = np.random.default_rng(e)
        plans = {'S': simple_greedy(P, np.zeros(N, np.int8), K, T, R, rng),
                 'H': h_greedy(A, P, np.zeros(N, bool), np.zeros(N, bool), K, T, rng)}
        firsts = {'probe': probe(A), 'split5': plans['S'][:5]}
        for mech in 'SH':
            for real in range(REAL):
                seed = 1000 * e + 10 * real + (mech == 'H')
                none = lambda act, ever, status: []
                vals = {'commit-S': run(mech, A, P, plans['S'], none, seed), 'commit-H': run(mech, A, P, plans['H'], none, seed)}
                vals['oracle'] = vals['commit-' + mech]
                vals['oracle-split5'] = run(mech, A, P, plans[mech][:5],
                                            lambda act, ever, status: plan_for(mech, A, P, ever, status, K - 5, T - 1, rng), seed)
                for pol, first in firsts.items():
                    chosen = {}
                    def rest(act, ever, status, first=first, chosen=chosen):
                        lS, lH = lik(A, first, act); m = 'S' if lS >= lH else 'H'; chosen['m'] = m
                        return plan_for(m, A, P, ever, status, K - len(first), T - 1, rng)
                    vals[pol] = run(mech, A, P, first, rest, seed); ident[pol].append(chosen['m'] == mech)
                for k, v in vals.items(): out.setdefault(k, []).append(v)
    return out, ident

for name in sys.argv[1:]:
    t0 = time.time(); out, ident = evaluate(name)
    orc = np.maximum(np.array(out['oracle']), np.array(out['oracle-split5']))
    row = '  '.join(f"{k}={100*np.mean(np.array(v)-orc)/orc.mean():+6.1f}%" for k, v in out.items())
    print(f"{name:9s} oracle={orc.mean():6.1f}  {row}  ident: " + ' '.join(f"{k}={100*np.mean(v):.0f}%" for k, v in ident.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
