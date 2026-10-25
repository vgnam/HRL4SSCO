"""T1: does the diffusion law change the decision?

For every ordered pair (L_true, L_plan):
    plan  = planner run under L_plan
    value = the plan replayed under L_true
    regret = 1 - value / value(plan run under L_true)

A plan is an open-loop schedule (seed sets per round), so replaying it under a
different law also pays for seeds that are wasted because the state evolved
differently. Common random numbers are used inside an instance, so the comparison
between two plans is much sharper than the absolute values.

Kill criterion: max regret over all pairs < 5%  ->  the law is decision-irrelevant
here and the symbolic-world-model direction has no decision value in this benchmark.
"""

import argparse
import csv
import os

import numpy as np

from common import OUT_DIR, build_arrays, ensure_out_dir, load_subgraph, table, Timer
from laws import T1_LAWS, get_law
from planner import plan_episode
from sim import rollout


SCHEDULES = ['even', 'normal', 'static2']


def build_plan(law, adj, indeg, T, budget, seed, op, M_greedy=96, M_sel=400,
               schedules=SCHEDULES):
    """Plan under `law`, letting the law choose its own budget schedule.

    T3b showed the best schedule depends on the law (front-loading wins under gate
    laws), so a fixed schedule would handicap some laws and bias the regret matrix.
    """
    best, best_v, best_sch = None, -np.inf, None
    for sch in schedules:
        alloc = 'static' if sch == 'static2' else sch
        plan = plan_episode(law, adj, indeg, T, budget, alloc=alloc, select='greedy',
                            M_greedy=M_greedy, seed=seed, period=2)
        v = evaluate(law, adj, indeg, plan, op, M_sel, seed=99)
        if v > best_v:
            best, best_v, best_sch = plan, v, sch
    return best, best_sch


def evaluate(law, adj, indeg, plan, op, M, seed):
    R = rollout(law, adj, indeg, [np.asarray(S, dtype=np.int64) for S in plan], M, seed, op=op)
    return float(R.sum(axis=1).mean())


def evaluate_paths(law, adj, indeg, plan, op, M, seed):
    """Per-path totals, so two plans can be compared with paired (common-random-number) errors."""
    R = rollout(law, adj, indeg, [np.asarray(S, dtype=np.int64) for S in plan], M, seed, op=op)
    return R.sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--T', type=int, default=10)
    ap.add_argument('--budget', type=int, default=20)
    ap.add_argument('--datasets', nargs='+', default=['wiki', 'dblp', 'reddit', 'higgs_100', 'hepph'])
    ap.add_argument('--seeds-per-dataset', type=int, default=2)
    ap.add_argument('--M-eval', type=int, default=2000)
    ap.add_argument('--M-greedy', type=int, default=96)
    ap.add_argument('--alloc', default='even')
    ap.add_argument('--laws', nargs='+', default=T1_LAWS)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    ensure_out_dir()
    tag = args.tag or f'n{args.n}'
    out = os.path.join(OUT_DIR, f't1_regret_{tag}.out')
    csv_path = os.path.join(OUT_DIR, f't1_regret_{tag}.csv')

    laws = {name: get_law(name) for name in args.laws}
    names = list(args.laws)
    # regret[true][plan] over instances, plus raw values
    regrets = {a: {b: [] for b in names} for a in names}
    regrets_self = {a: {b: [] for b in names} for a in names}
    values = {a: {b: [] for b in names} for a in names}
    tstats = {a: {b: [] for b in names} for a in names}
    per_instance_rows = []
    schedule_log = []

    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'T1 law-vs-decision regret | n={args.n} T={args.T} B={args.budget} '
                f'planner=greedy with per-law schedule search {SCHEDULES} '
                f'M_eval={args.M_eval} laws={names}\n\n')

    for ds in args.datasets:
        for iseed in range(args.seeds_per_dataset):
            seed = 101 + 17 * iseed
            with Timer(f'instance {ds}-{iseed}'):
                G = load_subgraph(ds, args.n, seed=seed)
                adj, indeg, _ = build_arrays(G)
                ops = {name: laws[name].op(adj) for name in names}
                plans, schedules = {}, {}
                for name in names:
                    plans[name], schedules[name] = build_plan(
                        laws[name], adj, indeg, args.T, args.budget, seed=seed,
                        op=ops[name], M_greedy=args.M_greedy)
                schedule_log.append((ds, iseed, dict(schedules)))
                for L_true in names:
                    base = evaluate_paths(laws[L_true], adj, indeg, plans[L_true], ops[L_true],
                                          args.M_eval, seed=7)
                    oracle = float(base.mean())
                    paths_all = {L_plan: (base if L_plan == L_true else
                                          evaluate_paths(laws[L_true], adj, indeg, plans[L_plan],
                                                         ops[L_true], args.M_eval, seed=7))
                                 for L_plan in names}
                    best_known = max(float(p.mean()) for p in paths_all.values())
                    row = []
                    for L_plan in names:
                        paths = paths_all[L_plan]
                        v = float(paths.mean())
                        r = 1.0 - v / best_known if best_known > 0 else 0.0
                        r_self = 1.0 - v / oracle if oracle > 0 else 0.0
                        d = base - paths                       # paired gain of the true-law plan
                        se = float(d.std(ddof=1) / np.sqrt(len(d)))
                        if se > 0:
                            t = float(d.mean() / se)
                        else:                                  # deterministic law: no MC noise
                            t = 0.0 if abs(d.mean()) < 1e-12 else np.inf * np.sign(d.mean())
                        regrets[L_true][L_plan].append(r)
                        regrets_self[L_true][L_plan].append(r_self)
                        values[L_true][L_plan].append(v)
                        tstats[L_true][L_plan].append(t)
                        row.append(r)
                    per_instance_rows.append((ds, iseed, L_true, best_known, row))

    # summary tables
    header = ['true \\ plan'] + names + ['best known', 'self oracle']
    rows_mean = []
    rows_max = []
    rows_self = []
    for L_true in names:
        mean_row, max_row, self_row = [], [], []
        for L_plan in names:
            rs = np.array(regrets[L_true][L_plan])
            mean_row.append(f'{100 * rs.mean():+6.1f}%')
            max_row.append(f'{100 * rs.max():+6.1f}%')
            self_row.append(f'{100 * np.mean(regrets_self[L_true][L_plan]):+6.1f}%')
        best_mean = float(np.mean(np.max([values[L_true][b] for b in names], axis=0)))
        self_mean = float(np.mean(values[L_true][L_true]))
        rows_mean.append([L_true] + mean_row + [f'{best_mean:.1f}', f'{self_mean:.1f}'])
        rows_max.append([L_true] + max_row + [f'{best_mean:.1f}', f'{self_mean:.1f}'])
        rows_self.append([L_true] + self_row + [f'{best_mean:.1f}', f'{self_mean:.1f}'])
    with open(out, 'a', encoding='utf-8') as f:
        f.write('MEAN regret vs the best plan found under the truth '
                '(row = law that is true, column = law used to plan)\n')
    table(rows_mean, header, path=out)
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\nMAX regret over instances (vs best plan found under the truth)\n')
    table(rows_max, header, path=out)
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\nMEAN regret vs the TRUE-LAW plan (negative = another law\'s plan happens to be '
                'better, i.e. the planner is weak for that law)\n')
    table(rows_self, header, path=out)

    # the practically important summary: planning with the repo/default law WC
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\npaired significance of the full matrix (mean paired t over instances, '
                'vs the true-law plan; |t| > 3 means the difference exceeds Monte-Carlo noise)\n')
    rows_t = []
    for L_true in names:
        cells = []
        for b in names:
            m = float(np.mean(tstats[L_true][b]))
            cells.append('   det' if not np.isfinite(m) else f'{m:+6.1f}')
        rows_t.append([L_true] + cells)
    table(rows_t, ['true \\ plan'] + names, path=out)

    with open(out, 'a', encoding='utf-8') as f:
        f.write('\nusing the default law WC to plan, when the truth is:\n')
    rows_wc = []
    for L_true in names:
        rs = np.array(regrets[L_true]['WC'])
        ts = np.array(tstats[L_true]['WC'])
        rows_wc.append([L_true, f'{100 * rs.mean():+.1f}%', f'{100 * rs.max():+.1f}%',
                        f'{ts.mean():+.1f}', f'{np.mean(values[L_true]["WC"]):.2f}',
                        f'{np.mean(values[L_true][L_true]):.2f}'])
    table(rows_wc, ['true law', 'mean regret', 'max regret', 'paired t', 'value (plan WC)',
                    'value (plan true)'], path=out)

    # worst pair overall
    pairs = [(100 * np.mean(regrets[a][b]), a, b) for a in names for b in names]
    pairs.sort(reverse=True)
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\nworst pairs (mean regret):\n')
        for r, a, b in pairs[:8]:
            f.write(f'   true {a:6s} plan {b:6s} {r:+6.1f}%\n')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'instance', 'true_law', 'oracle_value'] + [f'plan_{n}' for n in names])
        for ds, iseed, L_true, oracle, row in per_instance_rows:
            w.writerow([ds, iseed, L_true, f'{oracle:.4f}'] + [f'{r:.5f}' for r in row])

    with open(out, 'a', encoding='utf-8') as f:
        f.write('\nschedule chosen by each law (per instance):\n')
    counts = {name: {sch: 0 for sch in SCHEDULES} for name in names}
    for _, _, sched in schedule_log:
        for name, sch in sched.items():
            counts[name][sch] += 1
    table([[name] + [str(counts[name][sch]) for sch in SCHEDULES] for name in names],
          ['law'] + SCHEDULES, path=out)

    print(f'\nwrote {out}\nwrote {csv_path}')


if __name__ == '__main__':
    main()
