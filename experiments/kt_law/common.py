"""Shared helpers for the law/decision kill tests: graphs, arrays, output."""

import os
import sys
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
IM_DIR = os.path.join(REPO, 'IM')
OUT_DIR = os.path.join(HERE, 'out')

if IM_DIR not in sys.path:
    sys.path.insert(0, IM_DIR)


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def load_subgraph(dataset, n, seed, method='rw'):
    """An n-node induced subgraph of a repo dataset (same sampler the paper uses)."""
    from utils.datasets import dataset_sampler
    sampler = dataset_sampler(dataset, n, method=method, seed=seed)
    return sampler()


def er_graph(n, m_arcs, seed):
    """Erdos-Renyi digraph with roughly `m_arcs` arcs and no isolated node."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        p = m_arcs / float(n * (n - 1)) if n > 1 else 0.0
        G = nx.gnp_random_graph(n, p, seed=int(rng.integers(1 << 30)), directed=True)
        if G.number_of_edges() == 0:
            continue
        isolated = [v for v in G.nodes if G.in_degree(v) == 0 and G.out_degree(v) == 0]
        if not isolated:
            return G
    return G


def build_arrays(G):
    """Return (adj csr float32 with adj[u,v]=1 for u->v, indeg int array, outdeg int array)."""
    n = G.number_of_nodes()
    nodes = list(range(n))
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=np.float32, format='csr')
    A = sp.csr_matrix(A)
    indeg = np.asarray(A.sum(axis=0)).ravel().astype(np.int64)
    outdeg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    return A, indeg, outdeg


def law_operator(law, adj):
    """Influence operator for a law (cached by the caller when reused)."""
    return law.op(adj)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f'[{self.label}: {time.time() - self.t0:.1f}s]', flush=True)


def table(rows, header, path=None, float_fmt='{:.2f}'):
    """Render a simple fixed-width table (and optionally append it to `path`)."""
    widths = [max(len(str(header[i])), *(len(str(r[i])) for r in rows)) if rows else len(str(header[i]))
              for i in range(len(header))]
    lines = ['  '.join(str(header[i]).ljust(widths[i]) for i in range(len(header)))]
    lines.append('  '.join('-' * widths[i] for i in range(len(header))))
    for r in rows:
        lines.append('  '.join(str(r[i]).ljust(widths[i]) for i in range(len(header))))
    text = '\n'.join(lines)
    print(text, flush=True)
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    return text
