"""Shared dynamics for the kill tests.
Simple contagion S: IC with weighted cascade p(u->v)=1/indeg(v), SIR as IM/Env.py (each node active for one
step). In a step, seeds become active, then every active node tries its out-edges once. Live-edge
equivalent: a seed placed at step t reaches nodes within live distance T - t.
Complex contagion C: a node activates when >= THETA in-neighbours have ever been active (one hop per step)."""
import numpy as np, networkx as nx, scipy.sparse as sp

THETA = 2

def graph_arrays(g):
    """Adjacency A and edge probabilities P: the dataset probability 'p' when every edge has one (as IM/Env.py),
    otherwise the weighted cascade 1 / indeg(v)."""
    n = g.number_of_nodes()
    A = nx.to_scipy_sparse_array(g, nodelist=range(n), format='csr').astype(np.float64)
    A.data[:] = 1
    if g.number_of_edges() and all('p' in d for _, _, d in g.edges(data=True)):
        P = nx.to_scipy_sparse_array(g, nodelist=range(n), weight='p', format='csr').astype(np.float64)
    else:
        indeg = np.maximum(np.asarray(A.sum(0)).ravel(), 1)
        P = A @ sp.diags(1.0 / indeg)
    return sp.csr_matrix(A), sp.csr_matrix(P)

def sample_live(P, rng):
    L = P.copy(); L.data = (rng.random(L.nnz) < P.data).astype(np.float64); L.eliminate_zeros()
    return L

def reach_matrix(L, h, allowed):
    """X[s, v] = 1 if v is within live distance h of s, walking only into `allowed` nodes."""
    n = L.shape[0]
    L = L @ sp.diags(allowed.astype(np.float64))
    X = sp.identity(n, format='csr')
    for _ in range(h):
        Y = X + X @ L; Y.data[:] = 1
        if Y.nnz == X.nnz: break
        X = Y
    return sp.csr_matrix(X)

# ---------- simple contagion ----------
class SimpleState:
    def __init__(self, n):
        self.status = np.zeros(n, np.int8)  # 0 inactive, 1 active (propagates this step), 2 removed

def simple_step(state, seeds, Ltrue):
    seeds = [s for s in seeds if state.status[s] == 0]
    state.status[seeds] = 1
    act = state.status == 1
    hit = np.asarray(Ltrue[act].sum(0)).ravel() > 0
    new = hit & (state.status == 0)
    state.status[act] = 2; state.status[new] = 1
    return len(seeds) + int(new.sum()), new

def simple_greedy(P, status, k, h, R, rng):
    """Greedy seeds now, maximising expected activations within horizon h; untried edges are fresh."""
    if k <= 0: return []
    inactive, active = status == 0, status == 1
    Xs = [reach_matrix(sample_live(P, rng), h, inactive) for _ in range(R)]
    cov = [np.asarray(X[active].sum(0)).ravel() > 0 for X in Xs]
    seeds = []
    for _ in range(k):
        gain = sum(X @ (~c).astype(np.float64) for X, c in zip(Xs, cov))
        gain[~inactive] = -1; gain[seeds] = -1
        v = int(gain.argmax()); seeds.append(v)
        for i, X in enumerate(Xs): cov[i] |= X[v].toarray().ravel() > 0
    return seeds

def simple_value(P, seeds, T, R, rng):
    n = P.shape[0]; tot = 0
    for _ in range(R):
        st = SimpleState(n); L = sample_live(P, rng)
        for t in range(T):
            tot += simple_step(st, seeds if t == 0 else [], L)[0]
    return tot / R

# ---------- complex contagion ----------
def complex_run(A, ever, seeds, steps):
    """Return the ever-active mask after placing seeds now and running `steps` steps (seed step included)."""
    ever = ever.copy(); ever[list(seeds)] = True
    AT = A.T.tocsr()
    for _ in range(steps):
        new = (AT @ ever.astype(np.float64) >= THETA) & ~ever
        if not new.any(): break
        ever |= new
    return ever

def complex_greedy(A, ever, k, steps):
    if k <= 0: return []
    seeds = []; AT = A.T.tocsr(); outdeg = np.asarray(A.sum(1)).ravel()
    base = complex_run(A, ever, [], steps).sum()
    for _ in range(k):
        cur = ever.copy(); cur[seeds] = True
        tie = AT @ cur.astype(np.float64) if cur.any() else outdeg  # in-edges from the current cluster
        best, bv = -np.inf, None
        for v in np.flatnonzero(~cur):
            val = complex_run(A, ever, seeds + [int(v)], steps).sum()
            score = (val - base) * 1e4 + tie[v] + outdeg[v] * 1e-3
            if score > best: best, bv, bval = score, int(v), val
        seeds.append(bv); base = bval
    return seeds
