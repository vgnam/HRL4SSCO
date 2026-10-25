"""T3b: planner comparison at benchmark scale (n = 100, T = 10, budget = 20).

The point is not to invent a planner but to fix one that is good enough that a
model comparison is not dominated by planning noise. Everything is evaluated under
the law it was planned with, with common random numbers inside an instance.
"""

import os
import sys

import numpy as np

from common import OUT_DIR, build_arrays, ensure_out_dir, load_subgraph, table, Timer
from laws import get_law
from planner import plan_episode, plan_episode_heuristic
from sim import rollout

DATASETS = ['wiki', 'dblp', 'reddit', 'higgs_100', 'hepph']
N = 100
T = 10
BUDGET = 20
INSTANCE_SEEDS = [11, 12]
EPISODES = 3
M_EVAL = 400

PLANNERS = [
    ('greedy+even', lambda law, adj, indeg, s: plan_episode(law, adj, indeg, T, BUDGET, alloc='even',
                                                            select='greedy', M_greedy=96, seed=s)),
    ('greedy+normal', lambda law, adj, indeg, s: plan_episode(law, adj, indeg, T, BUDGET, alloc='normal',
                                                              select='greedy', M_greedy=96, seed=s)),
    ('greedy+static2', lambda law, adj, indeg, s: plan_episode(law, adj, indeg, T, BUDGET, alloc='static',
                                                               select='greedy', M_greedy=96, seed=s, period=2)),
    ('score+even', lambda law, adj, indeg, s: plan_episode_heuristic(law, adj, indeg, T, BUDGET, 'score',
                                                                     'even', seed=s)),
    ('score+normal', lambda law, adj, indeg, s: plan_episode_heuristic(law, adj, indeg, T, BUDGET, 'score',
                                                                       'normal', seed=s)),
    ('prob+even', lambda law, adj, indeg, s: plan_episode_heuristic(law, adj, indeg, T, BUDGET, 'prob',
                                                                    'even', seed=s)),
    ('indeg+even', lambda law, adj, indeg, s: plan_episode_heuristic(law, adj, indeg, T, BUDGET, 'indeg',
                                                                     'even', seed=s)),
    ('outdeg+even', lambda law, adj, indeg, s: plan_episode_heuristic(law, adj, indeg, T, BUDGET, 'outdeg',
                                                                      'even', seed=s)),
    ('random+even', lambda law, adj, indeg, s: plan_episode_heuristic(law, adj, indeg, T, BUDGET, 'random',
                                                                      'even', seed=s)),
]


def evaluate(law, adj, indeg, plan, op, M=M_EVAL, seed=0):
    R = rollout(law, adj, indeg, [np.asarray(S, dtype=np.int64) for S in plan], M, seed, op=op)
    return float(R.sum(axis=1).mean())


def main():
    ensure_out_dir()
    out = os.path.join(OUT_DIR, 't3b_planners.out')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'T3b planner comparison under the repo law (WC), n={N} T={T} B={BUDGET}, '
                f'{len(INSTANCE_SEEDS)} seeds x {EPISODES} episodes per dataset\n\n')

    for law_name in ['WC', 'CC2', 'DECAY']:
        law = get_law(law_name)
        print(f'\n########## law {law_name} ##########', flush=True)
        per_planner = {name: [] for name, _ in PLANNERS}
        per_instance_best = []
        for ds in DATASETS:
            for iseed in INSTANCE_SEEDS:
                G = load_subgraph(ds, N, seed=iseed)
                adj, indeg, _ = build_arrays(G)
                op = law.op(adj)
                with Timer(f'  instance {ds}-{iseed} ({G.number_of_edges()} arcs)'):
                    vals = {}
                    for name, fn in PLANNERS:
                        ep_vals = []
                        for ep in range(EPISODES):
                            plan = fn(law, adj, indeg, 1000 * ep + iseed)
                            ep_vals.append(evaluate(law, adj, indeg, plan, op, seed=7))
                        vals[name] = float(np.mean(ep_vals))
                        per_planner[name].append(vals[name])
                    best = max(vals.values())
                    per_instance_best.append(best)
        rows = []
        for name, _ in PLANNERS:
            v = np.array(per_planner[name])
            gap = np.mean(v / np.array(per_instance_best) - 1.0)
            rows.append([name, f'{v.mean():8.2f}', f'{v.std(ddof=1):5.2f}', f'{gap:+.1%}',
                         f'{int((v >= np.array(per_instance_best) - 1e-9).sum())}/{len(v)}'])
        table(rows, ['planner', 'mean value', 'sd', 'gap vs best', 'best count'], path=out)
        with open(out, 'a', encoding='utf-8') as f:
            f.write(f'(law {law_name})\n\n')


if __name__ == '__main__':
    main()
