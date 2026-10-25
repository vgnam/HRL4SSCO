"""Debug helper for the greedy planner (T3a showed it losing to plain heuristics)."""

import numpy as np

from common import build_arrays, er_graph
from laws import get_law
from planner import greedy_sequence, select_heuristic
from sim import rollout


def value_of(law, adj, indeg, seeds, M=20000):
    R = rollout(law, adj, indeg, [np.asarray(seeds, dtype=np.int64)], M, 0)
    return float(R.sum(axis=1).mean())


if __name__ == '__main__':
    # --- minimal check of the batched candidate evaluation -------------------
    G = er_graph(4, 4, seed=1)
    adj, indeg, _ = build_arrays(G)
    law = get_law('WC')
    print('graph arcs:', [(u, v) for u, v in G.edges()], 'indeg', indeg)
    for k in [1, 2]:
        seeds, gains = greedy_sequence(law, adj, None if False else law.op(adj), indeg,
                                       np.zeros(4, np.int8), np.zeros(4, np.float32), k, 4000, 0)
        print(f'greedy k={k}: seeds {seeds.tolist()} gains {np.round(gains, 3).tolist()} '
              f'value {value_of(law, adj, indeg, seeds):.3f}')
    for v in range(4):
        print(f'  single seed {v}: value {value_of(law, adj, indeg, [v]):.3f}')

    # --- case (a) from T3a ---------------------------------------------------
    G = er_graph(20, 60, seed=3)
    adj, indeg, _ = build_arrays(G)
    law = get_law('WC')
    op = law.op(adj)
    print('\nn=20, arcs', G.number_of_edges())
    seeds, gains = greedy_sequence(law, adj, op, indeg, np.zeros(20, np.int8),
                                   np.zeros(20, np.float32), 4, 96, 0)
    print('greedy seeds', seeds.tolist(), 'gains', np.round(gains, 3).tolist(),
          'value', round(value_of(law, adj, indeg, seeds), 3))
    seeds2, gains2 = greedy_sequence(law, adj, op, indeg, np.zeros(20, np.int8),
                                     np.zeros(20, np.float32), 4, 20000, 0)
    print('greedy (M=20000) seeds', seeds2.tolist(), 'gains', np.round(gains2, 3).tolist(),
          'value', round(value_of(law, adj, indeg, seeds2), 3))
    for rule in ['score', 'prob', 'outdeg', 'indeg']:
        S = select_heuristic(rule, law, adj, op, indeg, np.zeros((1, 20), np.int8),
                             np.zeros((1, 20), np.float32), 4)[0]
        print(f'{rule:7s} seeds {S.tolist()} value {value_of(law, adj, indeg, S):.3f}')

    # per-seed values for the top score nodes, to see what greedy should pick
    from planner import immediate_prob
    sc = ((np.ones((1, 20)) * law.edge_weight(indeg)[None, :]) @ adj.T)[0]
    for v in np.argsort(-sc)[:6]:
        print(f'  score-rank node {v}: score {sc[v]:.3f} single value {value_of(law, adj, indeg, [v]):.3f} '
              f'p1 {immediate_prob(law, op, indeg, np.zeros((1, 20), np.int8), np.zeros((1, 20), np.float32))[0, v]:.3f}')
