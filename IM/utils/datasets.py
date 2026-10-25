import os
import random
import warnings

import networkx as nx

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')

# name: (file, directed, weighted)
# directed=False: each undirected edge is stored once in the file, so both directions are added.
# weighted=True: third column is the edge activation probability p(u -> v) of the IC model.
# Unweighted graphs fall back to the weighted cascade p(u -> v) = 1 / in_degree(v) in Env.
DATASETS = {
    'dblp': ('dblp.txt', False, False),
    'hepph': ('hepph.txt', True, False),
    'wiki': ('wiki.txt', True, False),
    'power2500': ('power2500.txt', False, False),
    'higgs_100': ('higgs_100.txt', True, True),
    'reddit': ('reddit.txt', True, True),  # probabilities are trivalency {0.1, 0.05, 0.01}
}

_cache = {}


def load_dataset(name):
    if name in _cache:
        return _cache[name]
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}', choose from {list(DATASETS)}")
    file_name, directed, weighted = DATASETS[name]

    G = nx.DiGraph()
    num_invalid = 0
    with open(os.path.join(DATASET_DIR, file_name), 'r') as file:
        for line in file:
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
                attr['p'] = p
            G.add_edge(u, v, **attr)
            if not directed:
                G.add_edge(v, u, **attr)
    if num_invalid > 0:
        warnings.warn(f"{name}: dropped {num_invalid} edges with probability outside (0, 1]")

    _cache[name] = G
    return G


def sample_subgraph(G, n, method='rw', restart=0.15, seed=None):
    """Sample a connected induced subgraph with exactly n nodes, relabelled to 0..n-1.

    method='rw': random walk with restart (keeps degree structure of the original graph).
    method='bfs': breadth-first expansion (tends to collect a hub and its neighbours).
    """
    rng = random.Random(seed)
    U = G.to_undirected(as_view=True)
    component = max(nx.connected_components(U), key=len)
    if len(component) < n:
        raise ValueError(f"largest component has {len(component)} nodes < n = {n}")
    if len(component) == n:
        sampled = component
    elif method == 'bfs':
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
    elif method == 'rw':
        component = list(component)
        start = rng.choice(component)
        sampled, cur, stuck = {start}, start, 0
        while len(sampled) < n:
            if rng.random() < restart:
                cur = start
            cur = rng.choice(list(U.neighbors(cur)))
            if cur in sampled:
                stuck += 1
                # jump to a new start node when the walk stops discovering nodes
                if stuck > 100 * n:
                    start = cur = rng.choice(component)
                    sampled.add(start)
                    stuck = 0
            else:
                sampled.add(cur)
                stuck = 0
    else:
        raise ValueError(f"unknown sampling method '{method}'")

    mapping = {u: i for i, u in enumerate(sorted(sampled))}
    sub = nx.DiGraph()
    sub.add_nodes_from(range(len(mapping)))
    sub.add_edges_from((mapping[u], mapping[v], d) for u, v, d in G.subgraph(sampled).edges(data=True))
    return sub


def dataset_sampler(name, n, method='rw', seed=None):
    """Return a callable producing a new n-node subgraph of the dataset on each call."""
    G = load_dataset(name)
    rng = random.Random(seed)
    return lambda: sample_subgraph(G, n, method=method, seed=rng.random())
