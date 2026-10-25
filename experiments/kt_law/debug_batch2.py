"""Raw shapes/values of the batched candidate evaluation."""

import numpy as np
import scipy.sparse as sp

from common import build_arrays, er_graph
from laws import STATUS_ACTIVE, get_law

if __name__ == '__main__':
    G = er_graph(4, 4, seed=1)
    adj, indeg, _ = build_arrays(G)
    op = get_law('WC').op(adj)
    C, M = 4, 4000
    cand = np.arange(4)
    st = np.zeros((C * M, 4), np.int8)
    rows = np.repeat(np.arange(C), M)
    cols = np.repeat(cand, M)
    st[rows, cols] = STATUS_ACTIVE
    act = st == STATUS_ACTIVE
    a = act.astype(np.float32) @ op
    print('st.shape', st.shape, 'rows.shape', rows.shape, 'cols.shape', cols.shape)
    print('rows[:6]', rows[:6], 'cols[:6]', cols[:6])
    print('act rows 0,1,2,3999,4000,4001:\n', act[[0, 1, 2, 3999, 4000, 4001]].astype(int))
    print('act.sum(0)', act.sum(0), 'act.sum(1)[:6]', act.sum(1)[:6])
    print('a rows 0,4000,8000:\n', a[[0, 4000, 8000]])
    print('adj dense:\n', adj.toarray().astype(int))
    print('a[:6]:\n', a[:6])

    # direct micro-check of dense @ csr
    X = np.zeros((3, 4), np.float32)
    X[0, 0] = 1.0
    X[1, 1] = 1.0
    print('micro dense @ csr:\n', X @ op)
    print('micro (op.T @ X.T).T:\n', (op.T @ X.T).T)
