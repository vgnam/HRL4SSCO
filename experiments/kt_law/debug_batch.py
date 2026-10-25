"""Narrow diagnostic of the batched candidate evaluation in greedy_sequence."""

import numpy as np

from common import build_arrays, er_graph
from laws import STATUS_ACTIVE, get_law
from sim import transition

if __name__ == '__main__':
    G = er_graph(4, 4, seed=1)
    adj, indeg, _ = build_arrays(G)
    law = get_law('WC')
    op = law.op(adj)
    status_row = np.zeros(4, np.int8)
    E_row = np.zeros(4, np.float32)
    C, M = 4, 4000
    cand = np.arange(4)

    st = np.repeat(status_row[None, :], C * M, axis=0).astype(np.int8)
    Es = np.repeat(E_row[None, :], C * M, axis=0).astype(np.float32)
    rows = np.repeat(np.arange(C), M)
    st[rows, np.repeat(cand, M)] = STATUS_ACTIVE

    act = st == STATUS_ACTIVE
    a = act.astype(np.float32) @ op
    E = law.accumulate(Es, a)
    eligible = st == 0
    p = np.where(eligible, law.prob(E, a, indeg), 0.0)
    rng = np.random.default_rng(0)
    new = (rng.random(st.shape) < p) & eligible
    print('adj matrix:\n', adj.toarray().astype(int))
    print('op matrix:\n', np.asarray(op.todense()).astype(int))
    print('active per block head:', act[:3].astype(int).tolist())
    print('a per block:', [a[c * M].round(2).tolist() for c in range(C)])
    print('p per block:', [p[c * M].round(3).tolist() for c in range(C)])
    print('new mean per block:', new.reshape(C * M, -1).mean(axis=0), 
          [new[c * M:(c + 1) * M].mean() for c in range(C)])

    st2, E2, r = transition(law, op, indeg, st, Es, None, np.random.default_rng(0))
    print('transition reward per block:', [r[c * M:(c + 1) * M].mean().round(4) for c in range(C)])
