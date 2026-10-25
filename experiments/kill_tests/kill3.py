"""Kill test 3: one episode, mechanism unknown, prior 1/2 simple (S) / 1/2 complex (C, threshold 2).
Every policy is run under both mechanisms on the same graph; the score is the prior average."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, time, numpy as np
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'IM'))
from kt_common import *
from utils.datasets import load_dataset, sample_subgraph

K, T, R, E, N = 20, 10, 20, 8, 200

def mix_greedy(A, P, k, rng):
    """Seeds now maximising 1/2 E_S + 1/2 C (both from the empty state, horizon T)."""
    n = A.shape[0]; ones = np.ones(n, bool)
    Xs = [reach_matrix(sample_live(P, rng), T, ones) for _ in range(R)]
    cov = [np.zeros(n, bool) for _ in Xs]; ever0 = np.zeros(n, bool); seeds = []
    baseC = 0
    for _ in range(k):
        gS = sum(X @ (~c).astype(float) for X, c in zip(Xs, cov)) / R
        gC = np.array([complex_run(A, ever0, seeds + [v], T).sum() - baseC if v not in seeds else -1e9 for v in range(n)])
        g = 0.5 * gS + 0.5 * gC; g[seeds] = -1e9
        v = int(g.argmax()); seeds.append(v); baseC = complex_run(A, ever0, seeds, T).sum()
        for i, X in enumerate(Xs): cov[i] |= X[v].toarray().ravel() > 0
    return seeds

def probe_pair(A):
    """Two seeds with a common out-neighbour w of large in-degree: under C, w activates surely;
    under S, with probability 1 - (1 - 1/indeg(w))^2."""
    AT = A.T.tocsr(); indeg = np.diff(AT.indptr); outdeg = np.asarray(A.sum(1)).ravel()
    for w in np.argsort(-indeg):
        ins = AT.indices[AT.indptr[w]:AT.indptr[w + 1]]
        if len(ins) >= 2:
            a, b = ins[np.argsort(-outdeg[ins])[:2]]
            return [int(a), int(b)]

# ---- one episode under a mechanism, with a policy that may adapt after step 0 ----
def run(mech, A, P, Ltrue, first, adapt, rng):
    """first: seeds at step 0. adapt(obs) -> (mechanism to commit to) or None; the rest of the budget is
    then planned at step 1 for that mechanism."""
    n = A.shape[0]; total = 0; left = K - len(first)
    if mech == 'S':
        st = SimpleState(n); r, new = simple_step(st, first, Ltrue); total += r
        obs = dict(seeds=first, activated=np.flatnonzero(new))
    else:
        ever = np.zeros(n, bool); ever[first] = True
        new = (A.T.tocsr() @ ever.astype(float) >= THETA) & ~ever; ever |= new; total += len(first) + int(new.sum())
        obs = dict(seeds=first, activated=np.flatnonzero(new))
    commit = adapt(obs) if left > 0 else None
    if mech == 'S':
        rest = []
        if left > 0:
            rest = simple_greedy(P, st.status, left, T - 1, R, rng) if commit == 'S' else \
                   complex_greedy(A, st.status > 0, left, T - 1)
        for t in range(1, T):
            total += simple_step(st, rest if t == 1 else [], Ltrue)[0]
    else:
        rest = []
        if left > 0:
            if commit == 'C':
                rest = complex_greedy(A, ever, left, T - 1)
            else:
                status = np.where(ever, 2, 0).astype(np.int8); status[obs['activated']] = 1
                rest = simple_greedy(P, status, left, T - 1, R, rng)
        before = ever.sum(); ever = complex_run(A, ever, rest, T - 1); total += int(ever.sum() - before)
    return total, commit

def posterior_S(A, P, obs):
    """P(S | step-0 observation), prior 1/2. Under C the activation set is deterministic."""
    n = A.shape[0]; seeds = np.zeros(n, bool); seeds[obs['seeds']] = True
    act = np.zeros(n, bool); act[obs['activated']] = True
    predC = (A.T.tocsr() @ seeds.astype(float) >= THETA) & ~seeds
    likC = float((predC == act).all())
    q = np.ones(n)  # prob of not being activated under S
    PT = P.T.tocsr()
    for v in range(n):
        ps = PT.data[PT.indptr[v]:PT.indptr[v + 1]][seeds[PT.indices[PT.indptr[v]:PT.indptr[v + 1]]]]
        q[v] = np.prod(1 - ps)
    cand = ~seeds
    likS = float(np.prod(np.where(act[cand], 1 - q[cand], q[cand])))
    return likS / (likS + likC) if likS + likC > 0 else 0.5

def evaluate(name):
    out = {}; ident = {'probe': [], 'split5': [], 'split10': []}
    for e in range(E):
        g = sample_subgraph(load_dataset(name), N, seed=e); A, P = graph_arrays(g)
        Ltrue = sample_live(P, np.random.default_rng(10_000 + e)); rng = np.random.default_rng(e)
        plans = {'commit-S': simple_greedy(P, np.zeros(N, np.int8), K, T, R, rng),
                 'commit-C': complex_greedy(A, np.zeros(N, bool), K, T),
                 'mixture': mix_greedy(A, P, K, rng)}
        for mech in 'SC':
            static = {k: run(mech, A, P, Ltrue, s, lambda o: None, rng)[0] for k, s in plans.items()}
            static['oracle'] = static['commit-S'] if mech == 'S' else static['commit-C']
            adaptive = {}
            own = plans['commit-S'] if mech == 'S' else plans['commit-C']
            adaptive['oracle-split5'] = run(mech, A, P, Ltrue, own[:5], lambda o: mech, rng)[0]
            for pol, first in [('probe', probe_pair(A)), ('split5', plans['mixture'][:5]), ('split10', plans['mixture'][:10])]:
                v, c = run(mech, A, P, Ltrue, first, lambda o: 'S' if posterior_S(A, P, o) > 0.5 else 'C', rng)
                adaptive[pol] = v; ident[pol].append(c == mech)
            for k, v in {**static, **adaptive}.items(): out.setdefault(k, []).append(v)
    return out, ident

for name in sys.argv[1:]:
    t0 = time.time(); out, ident = evaluate(name)
    orc = np.maximum(np.array(out['oracle']), np.array(out['oracle-split5']))  # best oracle per (graph, mechanism)
    row = '  '.join(f"{k}={100*np.mean(np.array(v)-orc)/orc.mean():+6.1f}%" for k, v in out.items())
    print(f"{name:9s} oracle={orc.mean():6.1f}  {row}  ident: " +
          ' '.join(f"{k}={100*np.mean(v):.0f}%" for k, v in ident.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
