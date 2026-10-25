"""Graph loading for the IM domain.

The dataset readers and the random-walk/BFS samplers are direct ports of
``IM/utils/datasets.py`` so that STRATA instances are bit-identical to the
instances used by the existing kill tests.  We add the two things the plan
needs: a :class:`strata.core.types.Graph` bundle (sparse adjacency, degrees,
Louvain communities, optional node types) and a deterministic Louvain pass
(``nx.community.louvain_communities(seed=0)``).

Only the unweighted ``u -> v`` topology is kept: the IM environment and the
verified simulator both use weighted-cascade style laws driven by in-degree,
not the per-edge probabilities stored in a few dataset files.
"""

import os
import random
import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp

from strata.core.types import Graph

# Repository root: .../strata/domains/im/graph.py -> .../
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATASET_DIR = os.path.join(REPO_ROOT, "IM", "datasets")

# name: (file, directed, weighted) -- copied from IM/utils/datasets.py.
DATASETS = {
    "dblp": ("dblp.txt", False, False),
    "hepph": ("hepph.txt", True, False),
    "wiki": ("wiki.txt", True, False),
    "power2500": ("power2500.txt", False, False),
    "higgs_100": ("higgs_100.txt", True, True),
    "reddit": ("reddit.txt", True, True),
}

_cache = {}


def load_dataset(name):
    """Load a whole dataset as a ``networkx.DiGraph`` (cached)."""
    if name in _cache:
        return _cache[name]
    if name not in DATASETS:
        raise ValueError(
            "unknown dataset %r, choose from %s" % (name, sorted(DATASETS))
        )
    file_name, directed, weighted = DATASETS[name]

    G = nx.DiGraph()
    num_invalid = 0
    with open(os.path.join(DATASET_DIR, file_name), "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            attr = {}
            if weighted:
                p = float(parts[2])
                if not 0 < p <= 1:
                    num_invalid += 1
                    continue
                attr["p"] = p
            G.add_edge(u, v, **attr)
            if not directed:
                G.add_edge(v, u, **attr)
    if num_invalid > 0:
        warnings.warn(
            "%s: dropped %d edges with probability outside (0, 1]" % (name, num_invalid)
        )
    _cache[name] = G
    return G


def sample_subgraph(G, n, method="rw", restart=0.15, seed=None):
    """Sample a connected ``n``-node induced subgraph, relabelled to ``0..n-1``.

    Exact port of ``IM/utils/datasets.sample_subgraph`` (including the Python
    ``random.Random`` stream), which is what makes graph instances reproducible
    across STRATA and the existing experiment scripts.
    """
    rng = random.Random(seed)
    U = G.to_undirected(as_view=True)
    component = max(nx.connected_components(U), key=len)
    if len(component) < n:
        raise ValueError(
            "largest component has %d nodes < n = %d" % (len(component), n)
        )
    if len(component) == n:
        sampled = component
    elif method == "bfs":
        start = rng.choice(list(component))
        sampled, frontier = {start}, [start]
        while len(sampled) < n:
            next_frontier = []
            for u in frontier:
                neighbors = list(U.neighbors(u))
                rng.shuffle(neighbors)
                for v in neighbors:
                    if v not in sampled:
                        sampled.add(v)
                        next_frontier.append(v)
                        if len(sampled) == n:
                            break
                if len(sampled) == n:
                    break
            frontier = next_frontier
    elif method == "rw":
        component = list(component)
        start = rng.choice(component)
        sampled, cur, stuck = {start}, start, 0
        while len(sampled) < n:
            if rng.random() < restart:
                cur = start
            cur = rng.choice(list(U.neighbors(cur)))
            if cur in sampled:
                stuck += 1
                if stuck > 100 * n:
                    start = cur = rng.choice(component)
                    sampled.add(start)
                    stuck = 0
            else:
                sampled.add(cur)
                stuck = 0
    else:
        raise ValueError("unknown sampling method %r" % (method,))

    mapping = {u: i for i, u in enumerate(sorted(sampled))}
    sub = nx.DiGraph()
    sub.add_nodes_from(range(len(mapping)))
    sub.add_edges_from(
        (mapping[u], mapping[v], d) for u, v, d in G.subgraph(sampled).edges(data=True)
    )
    return sub


def dataset_sampler(name, n, method="rw", seed=None):
    """Return a callable producing a new ``n``-node subgraph on each call."""
    G = load_dataset(name)
    rng = random.Random(seed)
    return lambda: sample_subgraph(G, n, method=method, seed=rng.random())


def louvain_labels(G, seed=0):
    """Labelled Louvain partition of the underlying undirected graph."""
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    U = G.to_undirected(as_view=True)
    parts = nx.community.louvain_communities(U, seed=seed)
    # Deterministic, contiguous ids ordered by the smallest member.
    parts = sorted((sorted(int(v) for v in part) for part in parts), key=lambda p: p[0])
    labels = np.empty(n, dtype=np.int64)
    for cid, part in enumerate(parts):
        labels[part] = cid
    return labels


def build_graph(G, name="", community=True, node_type=None, louvain_seed=0):
    """Bundle a NetworkX digraph into the immutable :class:`Graph` contract."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    if nodes != list(range(n)):
        mapping = {v: i for i, v in enumerate(nodes)}
        G = nx.relabel_nodes(G, mapping, copy=True)
    A = nx.to_scipy_sparse_array(
        G, nodelist=list(range(n)), dtype=np.float32, format="csr"
    )
    adj = sp.csr_matrix(A)
    adjT = sp.csr_matrix(adj.T)
    indeg = np.asarray(adj.sum(axis=0)).ravel().astype(np.int64)
    outdeg = np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)
    comm = louvain_labels(G, seed=louvain_seed) if community else np.zeros(n, np.int64)
    nt = None if node_type is None else np.asarray(node_type, dtype=np.int64).copy()
    return Graph(
        adj=adj,
        adjT=adjT,
        indeg=indeg,
        outdeg=outdeg,
        community=comm,
        node_type=nt,
        name=name,
    )


def sample_instance(
    dataset, n, seed, method="rw", name=None, community=True, node_type=None
):
    """Convenience wrapper: load -> sample -> build a :class:`Graph`."""
    G = sample_subgraph(load_dataset(dataset), n, method=method, seed=seed)
    return build_graph(
        G,
        name=name or "%s-n%d-seed%d" % (dataset, n, seed),
        community=community,
        node_type=node_type,
    )
