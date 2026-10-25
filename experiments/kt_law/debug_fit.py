"""Diagnostic for T2: why does MLE not select the gate law when the truth is a gate?"""

import numpy as np

from common import build_arrays, load_subgraph
from laws import get_law
from t2_fit import ParamLaw, candidate_laws, collect_logs, loglik

if __name__ == '__main__':
    law_true = get_law('CC2')
    G = load_subgraph('wiki', 100, seed=101)
    adj, indeg, _ = build_arrays(G)
    eps = collect_logs(law_true, adj, indeg, T=10, budget=20, n_episodes=20, seed=101)

    acts = sum(int(r['y'].sum()) for ep in eps for r in ep)
    viol = sum(int((r['y'] & (r['a'] < 2)).sum()) for ep in eps for r in ep)
    elig = sum(int(r['eligible'].sum()) for ep in eps for r in ep)
    hi = sum(int(((r['a'] >= 2) & r['eligible']).sum()) for ep in eps for r in ep)
    hi_act = sum(int(((r['a'] >= 2) & r['eligible'] & r['y']).sum()) for ep in eps for r in ep)
    print(f'episodes {len(eps)}  activations {acts}  violations(y with a<2) {viol}')
    print(f'eligible node-rounds {elig}  of which a>=2: {hi}  activated: {hi_act}')

    print('\nlog-likelihoods on the same logs:')
    for law in [ParamLaw('gate', g=g) for g in (1, 2, 3, 4, 5)] + \
               [ParamLaw('wc', w=w) for w in (0.02, 0.05, 0.08, 0.2)] + \
               [ParamLaw('uni', c=0.05), law_true]:
        print(f'  {law.name:24s} {loglik(law, eps, indeg):14.1f}')

    cands = candidate_laws(True)
    lls = [(loglik(l, eps, indeg), l.name) for l in cands]
    lls.sort(reverse=True)
    print('\ntop-5 candidates on these logs:')
    for ll, name in lls[:5]:
        print(f'  {name:24s} {ll:14.1f}')
