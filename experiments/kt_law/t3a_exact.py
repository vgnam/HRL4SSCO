"""T3a: is the planner good enough to be trusted?

Exhaustive references on instances small enough to enumerate:

  (a) single round, n = 20, budget K = 4   -> all C(20,4) seed sets
  (b) two rounds,  n = 8,  budget 3        -> all (S1, S2) with |S1| + |S2| <= 3
  (c) three rounds, n = 6, budget 2        -> all (S1, S2, S3) with total <= 2
                                             (this one isolates *when to spend*)

Each planner is run for several episodes under the true law; the realised plan is
replayed under the same law with the same Monte-Carlo budget as the exhaustive
sweep, so the comparison is apples to apples.
"""

import itertools
import os
import sys

import numpy as np

from common import OUT_DIR, build_arrays, ensure_out_dir, er_graph, table, Timer
from laws import get_law
from planner import plan_episode, plan_episode_heuristic
from sim import rollout

M_EXACT = 20000
M_REFINE = 200000


def all_policies(n, T, budget):
    """Every (S_1..S_T) with sum |S_t| <= budget, as tuples of node tuples."""
    subs = {k: [tuple(c) for c in itertools.combinations(range(n), k)] for k in range(budget + 1)}
    out = []

    def rec(t, rem, acc):
        if t == T:
            out.append(tuple(acc))
            return
        for k in range(rem + 1):
            for S in subs[k]:
                rec(t + 1, rem - k, acc + [S])

    rec(0, budget, [])
    return out


def policy_value(law, adj, indeg, policy, M, seed=0):
    seed_sets = [np.asarray(S, dtype=np.int64) for S in policy]
    R = rollout(law, adj, indeg, seed_sets, M, seed)
    return float(R.sum(axis=1).mean())


def exhaustive(law, adj, indeg, T, budget, M=M_EXACT, refine=12, seed=0):
    n = adj.shape[0]
    policies = all_policies(n, T, budget)
    vals = np.array([policy_value(law, adj, indeg, p, M, seed) for p in policies])
    order = np.argsort(-vals)
    best = int(order[0])
    # sharpen the top candidates so the reference is not limited by MC noise
    for i in order[:refine]:
        vals[i] = policy_value(law, adj, indeg, policies[i], M_REFINE, seed)
    best = int(np.argmax(vals))
    return policies, vals, best


def allocation_pattern(policy):
    return '+'.join(str(len(S)) for S in policy)


def run_case(name, law, adj, indeg, T, budget, n_episodes=12, seed=0, verbose_best=6):
    n = adj.shape[0]
    print(f'\n=== {name} | law {law.name} | n={n} T={T} budget={budget} ===', flush=True)
    with Timer('exhaustive'):
        policies, vals, best = exhaustive(law, adj, indeg, T, budget, seed=seed)
    order = np.argsort(-vals)
    opt = vals[best]
    print(f'policies enumerated: {len(policies)}   exact best value: {opt:.3f}   '
          f'best allocation: {allocation_pattern(policies[best])}')
    print('top exact policies:')
    for i in order[:verbose_best]:
        print(f'   {allocation_pattern(policies[i]):10s} {policies[i]}  {vals[i]:.3f}')

    planner_cfgs = [
        ('greedy+lookahead', lambda ep: plan_episode(law, adj, indeg, T, budget, alloc='lookahead',
                                                     select='greedy', M_greedy=96, M_look=64, seed=ep)),
        ('greedy+even', lambda ep: plan_episode(law, adj, indeg, T, budget, alloc='even',
                                                select='greedy', M_greedy=96, seed=ep)),
        ('greedy+normal', lambda ep: plan_episode(law, adj, indeg, T, budget, alloc='normal',
                                                  select='greedy', M_greedy=96, seed=ep)),
        ('prob+even', lambda ep: plan_episode_heuristic(law, adj, indeg, T, budget, 'prob', 'even', seed=ep)),
        ('score+even', lambda ep: plan_episode_heuristic(law, adj, indeg, T, budget, 'score', 'even', seed=ep)),
        ('indeg+even', lambda ep: plan_episode_heuristic(law, adj, indeg, T, budget, 'indeg', 'even', seed=ep)),
        ('outdeg+even', lambda ep: plan_episode_heuristic(law, adj, indeg, T, budget, 'outdeg', 'even', seed=ep)),
        ('random+even', lambda ep: plan_episode_heuristic(law, adj, indeg, T, budget, 'random', 'even', seed=ep)),
    ]
    rows = []
    for label, fn in planner_cfgs:
        vals_ep, pats = [], []
        for ep in range(n_episodes):
            plan = fn(ep)
            v = policy_value(law, adj, indeg, [tuple(int(x) for x in S) for S in plan], M_EXACT, seed)
            vals_ep.append(v)
            pats.append(allocation_pattern(plan))
        mean = float(np.mean(vals_ep))
        rows.append([label, f'{mean:.3f}', f'{mean / opt - 1:+.1%}', pats[0], f'{np.std(vals_ep, ddof=1):.3f}'])
    table(rows, ['planner', 'value', 'gap vs exact', 'alloc (ep0)', 'sd'],
          path=os.path.join(ensure_out_dir(), 't3a_exact.out'))
    with open(os.path.join(OUT_DIR, 't3a_exact.out'), 'a', encoding='utf-8') as f:
        f.write(f'case {name} law {law.name} n={n} T={T} B={budget} exact={opt:.3f} '
                f'({len(policies)} policies)\n\n')
    return opt, rows


if __name__ == '__main__':
    ensure_out_dir()
    out = os.path.join(OUT_DIR, 't3a_exact.out')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('T3a planner calibration against exhaustive references\n\n')

    # (a) single round, selection only
    G = er_graph(20, 60, seed=3)
    adj, indeg, _ = build_arrays(G)
    run_case('a: single round, selection', get_law('WC'), adj, indeg, T=1, budget=4, n_episodes=8)

    # (b) two rounds, selection + allocation
    G = er_graph(8, 14, seed=11)
    adj, indeg, _ = build_arrays(G)
    run_case('b: two rounds', get_law('WC'), adj, indeg, T=2, budget=3, n_episodes=8)

    # (c) three rounds, allocation only: does memory in the law change when to spend?
    G = er_graph(6, 10, seed=5)
    adj, indeg, _ = build_arrays(G)
    for law_name in ['WC', 'REINF', 'DECAY', 'CC2']:
        run_case(f'c: three rounds ({law_name})', get_law(law_name), adj, indeg, T=3, budget=2, n_episodes=8)
