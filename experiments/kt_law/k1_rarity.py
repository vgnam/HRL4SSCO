"""K1: how rare are gate events in the T2 logs -- and does rarity explain libA's regret?

Background (arXiv:2607.14169): a world model that passes a sampling gate can still
lose at play when the rule it misses is pivotal but rare; the chance that N sampled
transitions never show the rule is (1 - rarity)^N.

For each T2 instance (same graphs, seeds and behaviour logs as t2_fit.py) this
script measures, on the training logs and per behaviour policy:
  r_node   fraction of eligible node-rounds with a_v >= g   (a_v = active in-neighbours)
  r_round  fraction of rounds with at least one such node
  r_ep     fraction of episodes with at least one such round
  miss     (1 - r_round)^N_rounds and (1 - r_ep)^N_ep: probability of never seeing the event
It also checks whether the gate threshold itself is identified (train log-likelihood
of gate(g) for g = 1..5; t2_fit keeps the first of tied candidates), and measures the
same rarity on the state distribution of the *plans* (oracle plan under the truth,
plan under the libA law) replayed under the truth -- the policy-guided check of the
paper. Finally it lines rarity up against libA regret.
"""

import argparse
import os

import numpy as np

from common import OUT_DIR, build_arrays, ensure_out_dir, load_subgraph, table, Timer
from laws import get_law
from planner import plan_episode
from sim import rollout, transition
from t2_fit import BEHAVIOUR, ParamLaw, collect_logs, fit_library, loglik

GATES = (2, 3)


def log_rarity(episodes, g):
    """Gate-event rarity of a list of logged episodes (dicts with 'a', 'eligible')."""
    elig_nr = hit_nr = rounds = hit_rounds = hit_eps = 0
    a1_nr = 0
    for rounds_ep in episodes:
        ep_hit = False
        for r in rounds_ep:
            e = r['eligible']
            hit = e & (r['a'] >= g)
            elig_nr += int(e.sum())
            hit_nr += int(hit.sum())
            a1_nr += int((e & (r['a'] == 1)).sum())
            rounds += 1
            if hit.any():
                hit_rounds += 1
                ep_hit = True
        hit_eps += int(ep_hit)
    n_ep = len(episodes)
    return {'r_node': hit_nr / max(elig_nr, 1), 'r_round': hit_rounds / max(rounds, 1),
            'r_ep': hit_eps / max(n_ep, 1), 'n_hit': hit_nr, 'n_a1': a1_nr,
            'N_rounds': rounds, 'N_ep': n_ep}


def plan_rarity(law_true, adj, indeg, seed_sets, g_list, M=200, seed=7):
    """Rarity of gate events on the states a fixed plan visits under the truth."""
    op = law_true.op(adj)
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    status = np.zeros((M, n), dtype=np.int8)
    E = np.zeros((M, n), dtype=np.float32)
    stats = {g: [0, 0, 0, 0] for g in g_list}   # elig node-rounds, hit node-rounds, rounds, hit rounds
    for S in seed_sets:
        status, E, _, info = transition(law_true, op, indeg, status, E,
                                        np.asarray(S, np.int64), rng, return_info=True)
        for g in g_list:
            hit = info['eligible'] & (info['a'] >= g)
            st = stats[g]
            st[0] += int(info['eligible'].sum())
            st[1] += int(hit.sum())
            st[2] += M
            st[3] += int(hit.any(axis=1).sum())
    return {g: {'r_node': s[1] / max(s[0], 1), 'r_round': s[3] / max(s[2], 1)}
            for g, s in stats.items()}


def best_plan(law, adj, indeg, T, budget, seed, M_greedy=96):
    """Same plan selection as t2_fit.regret_of."""
    plans = []
    for alloc in ['even', 'normal', 'static']:
        p = plan_episode(law, adj, indeg, T, budget, alloc=alloc, select='greedy',
                         M_greedy=M_greedy, seed=seed, period=2)
        v = rollout(law, adj, indeg, [np.asarray(S, np.int64) for S in p], 400, 99,
                    op=law.op(adj)).sum(axis=1).mean()
        plans.append((v, alloc, p))
    plans.sort(key=lambda x: -x[0])
    return plans[0][2], plans[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', default='CC2')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--T', type=int, default=10)
    ap.add_argument('--budget', type=int, default=20)
    ap.add_argument('--datasets', nargs='+', default=['wiki', 'dblp', 'reddit'])
    ap.add_argument('--seeds-per-dataset', type=int, default=1)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--test-frac', type=float, default=0.3)
    ap.add_argument('--M-eval', type=int, default=2000)
    args = ap.parse_args()

    ensure_out_dir()
    out = os.path.join(OUT_DIR, f'k1_rarity_{args.truth}_{args.n}_s{args.seeds_per_dataset}.out')
    law_true = get_law(args.truth)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'K1 gate-event rarity | hidden truth {args.truth} | n={args.n} T={args.T} '
                f'B={args.budget} | {args.episodes} behaviour episodes, same logs as t2_fit '
                f'(train = last {int(100 * (1 - args.test_frac))}%)\n\n')

    rows_log, rows_gate, rows_plan, rows_sum, rows_pol = [], [], [], [], []
    mix_points, pol_points = [], []
    for ds in args.datasets:
        for iseed in range(args.seeds_per_dataset):
            seed = 101 + 17 * iseed
            inst = f'{ds}-{iseed}'
            with Timer(f'instance {inst}'):
                G = load_subgraph(ds, args.n, seed=seed)
                adj, indeg, _ = build_arrays(G)
                episodes = collect_logs(law_true, adj, indeg, args.T, args.budget,
                                        args.episodes, seed=seed)
                n_test = int(round(args.test_frac * len(episodes)))
                ep_test, ep_train = episodes[:n_test], episodes[n_test:]
                # behaviour policy of episode i is BEHAVIOUR[i % 4]; keep the original index
                idx_train = list(range(n_test, len(episodes)))

                # 1. rarity in the training logs, overall and per behaviour policy
                for g in GATES:
                    groups = [('all', ep_train)] + [
                        (b, [episodes[i] for i in idx_train if BEHAVIOUR[i % len(BEHAVIOUR)] == b])
                        for b in BEHAVIOUR]
                    for name, eps in groups:
                        s = log_rarity(eps, g)
                        rows_log.append([inst, g, name, s['N_ep'], s['N_rounds'],
                                         f"{s['r_node']:.4f}", f"{s['r_round']:.3f}",
                                         f"{s['r_ep']:.3f}", s['n_hit'], s['n_a1'],
                                         f"{(1 - s['r_round']) ** s['N_rounds']:.2e}",
                                         f"{(1 - s['r_ep']) ** s['N_ep']:.2e}"])

                # 2. is the threshold identified? train LL of every gate(g)
                lls = [loglik(ParamLaw('gate', g=gg), ep_train, indeg) for gg in range(1, 6)]
                best = max(lls)
                tied = [gg for gg, ll in zip(range(1, 6), lls) if ll == best]
                rows_gate.append([inst] + [f'{ll:.0f}' for ll in lls] + [','.join(map(str, tied))])

                # 3. fitted laws + regret (as t2_fit) and policy-guided rarity
                law_a, _, _ = fit_library(ep_train, ep_test, indeg, False)
                plan_true, alloc_true = best_plan(law_true, adj, indeg, args.T, args.budget, seed)
                plan_a, alloc_a = best_plan(law_a, adj, indeg, args.T, args.budget, seed)
                op_true = law_true.op(adj)
                v_true = rollout(law_true, adj, indeg, [np.asarray(S, np.int64) for S in plan_true],
                                 args.M_eval, 7, op=op_true).sum(axis=1).mean()
                v_a = rollout(law_true, adj, indeg, [np.asarray(S, np.int64) for S in plan_a],
                              args.M_eval, 7, op=op_true).sum(axis=1).mean()
                regret_a = 1.0 - v_a / v_true

                pr_true = plan_rarity(law_true, adj, indeg, plan_true, GATES)
                pr_a = plan_rarity(law_true, adj, indeg, plan_a, GATES)
                for g in GATES:
                    rows_plan.append([inst, g, f"{log_rarity(ep_train, g)['r_node']:.4f}",
                                      f"{pr_true[g]['r_node']:.4f}", f"{pr_a[g]['r_node']:.4f}",
                                      f"{log_rarity(ep_train, g)['r_round']:.3f}",
                                      f"{pr_true[g]['r_round']:.3f}", f"{pr_a[g]['r_round']:.3f}"])

                s2 = log_rarity(ep_train, 2)
                rows_sum.append([inst, f'{100 * regret_a:+.1f}%', law_a.name[:22],
                                 f'{v_true:.1f}/{v_a:.1f}', alloc_true, alloc_a,
                                 f"{s2['r_node']:.4f}", f"{pr_true[2]['r_node']:.4f}",
                                 f"{pr_true[2]['r_node'] / max(s2['r_node'], 1e-9):.1f}x",
                                 f"{(1 - s2['r_round']) ** s2['N_rounds']:.1e}"])

                # 4. logs from a single behaviour policy: vary rarity, refit both libraries
                plan_cache = {law_true.name: plan_true, law_a.name: plan_a}

                def value_of(law):
                    if law.name not in plan_cache:
                        plan_cache[law.name] = best_plan(law, adj, indeg, args.T, args.budget, seed)[0]
                    return rollout(law_true, adj, indeg,
                                   [np.asarray(S, np.int64) for S in plan_cache[law.name]],
                                   args.M_eval, 7, op=op_true).sum(axis=1).mean()

                for b in BEHAVIOUR:
                    tr = [episodes[i] for i in idx_train if BEHAVIOUR[i % len(BEHAVIOUR)] == b]
                    te = [episodes[i] for i in range(n_test) if BEHAVIOUR[i % len(BEHAVIOUR)] == b]
                    la, _, _ = fit_library(tr, te, indeg, False)
                    lb, llb, _ = fit_library(tr, te, indeg, True)
                    gate_ll = {gg: loglik(ParamLaw('gate', g=gg), tr, indeg) for gg in range(1, 6)}
                    tied = [gg for gg, ll in gate_ll.items() if ll == llb] if lb.mech == 'gate' else []
                    # regret if the tie had been broken any other way (grid order is arbitrary)
                    worst = max((1 - value_of(ParamLaw('gate', g=gg)) / v_true for gg in tied),
                                default=float('nan'))
                    s = log_rarity(tr, 2)
                    r_a = 1 - value_of(la) / v_true
                    r_b = 1 - value_of(lb) / v_true
                    pol_points.append((s['r_node'], r_a))
                    rows_pol.append([inst, b, s['N_ep'], f"{s['r_node']:.4f}", s['n_hit'], s['n_a1'],
                                     f"{(1 - s['r_round']) ** s['N_rounds']:.1e}",
                                     la.name[:20], f'{100 * r_a:+.1f}%',
                                     lb.name[:16], f'{100 * r_b:+.1f}%',
                                     ','.join(map(str, tied)) or '-',
                                     f'{100 * worst:+.1f}%' if tied else '-'])
                mix_points.append((s2['r_node'], regret_a))

    with open(out, 'a', encoding='utf-8') as f:
        f.write('## 1. Gate-event rarity in the training logs\n'
                'r_node = share of eligible node-rounds with a>=g; r_round / r_ep = share of rounds / '
                'episodes with at least one; n_hit = such node-rounds; n_a1 = eligible node-rounds with a=1 '
                '(the events that refute IC under a gate); miss = (1-r)^N\n\n')
    table(rows_log, ['inst', 'g', 'policy', 'N_ep', 'N_rnd', 'r_node', 'r_round', 'r_ep',
                     'n_hit', 'n_a1', 'miss_rnd', 'miss_ep'], path=out)
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\n## 2. Is the threshold identified? train log-likelihood of gate(g)\n'
                't2_fit keeps the first of tied candidates, so ties make g=2 win by grid order\n\n')
    table(rows_gate, ['inst', 'g=1', 'g=2', 'g=3', 'g=4', 'g=5', 'argmax (ties)'], path=out)
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\n## 3. Policy-guided rarity: logs vs states visited by the plans (replayed under the truth)\n\n')
    table(rows_plan, ['inst', 'g', 'r_node log', 'r_node oracle', 'r_node planA',
                      'r_rnd log', 'r_rnd oracle', 'r_rnd planA'], path=out)
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\n## 4. Rarity vs libA regret (g=2)\n'
                'shift = r_node under the oracle plan / r_node in the logs\n\n')
    table(rows_sum, ['inst', 'regret libA', 'law A', 'value true/A', 'alloc true', 'alloc A',
                     'r_node log', 'r_node oracle', 'shift', 'miss_rnd'], path=out)
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\n## 5. Logs from one behaviour policy only (35 train episodes each): rarity vs regret\n'
                'n_a1 = eligible node-rounds with a=1 (separate g=1 from g=2); tied g = gate thresholds '
                'whose train LL equals the selected law (fit keeps the first by grid order);\n'
                'worst tied = regret of the worst threshold in that tie\n\n')
    table(rows_pol, ['inst', 'policy', 'N_ep', 'r_node', 'n_hit', 'n_a1', 'miss_rnd',
                     'law A', 'regret A', 'law B', 'regret B', 'tied g', 'worst tied'], path=out)

    def spearman(points):
        x, y = np.array(points, dtype=float).T
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    with open(out, 'a', encoding='utf-8') as f:
        f.write(f'\nSpearman(r_node log, libA regret): mixed logs {spearman(mix_points):+.2f} '
                f'(n={len(mix_points)}); single-policy logs {spearman(pol_points):+.2f} '
                f'(n={len(pol_points)})\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
