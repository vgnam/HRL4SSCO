"""Kill test 2 (AIM): value of adaptivity with the law known. Open-loop optimum seeds everything at t=0
(balls only shrink with time), so adaptive - open-loop bounds what budget timing can add."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, time, numpy as np, networkx as nx
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'IM'))
from kt_common import *
from utils.datasets import load_dataset, sample_subgraph

K, T, R, E = 20, 10, 30, 25

def schedule(name):
    if name == 'all-now': return [K] + [0] * (T - 1)
    if name == 'even': return [K // T] * T
    if name == 'half': return [K // 2] + [((K - K // 2) * (t + 1)) // (T - 1) - ((K - K // 2) * t) // (T - 1) for t in range(T - 1)]
    if name == 'quarter': return [K // 4] + [((K - K // 4) * (t + 1)) // (T - 1) - ((K - K // 4) * t) // (T - 1) for t in range(T - 1)]

def run(P, Ltrue, sched, rng, R=R):
    st = SimpleState(P.shape[0]); total = 0; left = K
    for t in range(T):
        k = min(sched[t], left); left -= k
        seeds = simple_greedy(P, st.status, k, T - t, R, rng)
        total += simple_step(st, seeds, Ltrue)[0]
    return total

def prophet(P, Ltrue):
    X = reach_matrix(Ltrue, T, np.ones(P.shape[0], bool)); cov = np.zeros(P.shape[0], bool); seeds = []
    for _ in range(K):
        g = X @ (~cov).astype(float); g[seeds] = -1; v = int(g.argmax()); seeds.append(v); cov |= X[v].toarray().ravel() > 0
    return cov.sum()

graphs = [('ER n=200 p=0.01', lambda s: nx.erdos_renyi_graph(200, 0.01, seed=s, directed=True)),
          ('ER n=500 p=0.01', lambda s: nx.erdos_renyi_graph(500, 0.01, seed=s, directed=True))]
for gname, make in graphs:
    res = {'all-now R=30': [], 'all-now R=300': [], 'quarter R=30': []}
    for e in range(E):
        g = make(e); A, P = graph_arrays(g); Ltrue = sample_live(P, np.random.default_rng(10_000 + e))
        res['all-now R=30'].append(run(P, Ltrue, schedule('all-now'), np.random.default_rng(e)))
        res['all-now R=300'].append(run(P, Ltrue, schedule('all-now'), np.random.default_rng(e), R=300))
        res['quarter R=30'].append(run(P, Ltrue, schedule('quarter'), np.random.default_rng(e)))
    ol = np.array(res['all-now R=30'])
    print(gname, '  '.join(f"{k}={np.mean(v):6.1f} ({100*np.mean(np.array(v)-ol)/ol.mean():+5.1f}% +-{100*np.std(np.array(v)-ol)/np.sqrt(E)/ol.mean():.1f})" for k, v in res.items()), flush=True)
