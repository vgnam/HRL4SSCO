"""T2: fitting a law from logged cascades -- does the *decision* come back?

Protocol
  1. Ground truth is a hidden law (default CC2: activation needs two active
     in-neighbours, a hard gate -- nothing like the repo's independent cascade).
  2. Logs: episodes run under suboptimal behaviour policies (random / score /
     indeg / outdeg), recording per round the active-neighbour counts, the
     eligible mask and which nodes activated.
  3. Two law libraries are fitted by maximum likelihood on 70% of the episodes and
     compared on the held-out 30%:
       A  independent-cascade family only (per-edge probability reshuffling)
       B  A plus gate laws (activation requires a threshold number of neighbours)
  4. The selected law of each library is used to plan; the plan is replayed under
     the hidden truth. Regret is measured against planning with the true law.
     The baseline is planning with the repo default (WC, i.e. p = 1/indeg).

The question T2 answers: when the truth is structural, does better likelihood buy
better decisions -- and does the answer depend on whether the library can express
the structure at all?
"""

import argparse
import os

import numpy as np

from common import OUT_DIR, build_arrays, ensure_out_dir, load_subgraph, table, Timer
from laws import Law, STATUS_INACTIVE, STATUS_ACTIVE, get_law
from planner import plan_episode, select_heuristic
from sim import rollout, transition

BEHAVIOUR = ['random', 'score', 'indeg', 'outdeg']


# --------------------------------------------------------------------- law family

class ParamLaw(Law):
    """A law with fitted parameters, plug-compatible with planner/sim."""

    def __init__(self, mech, **params):
        self.mech = mech
        self.params = params
        self.rho = float(params.get('rho', 0.0))
        self.name = mech + '(' + ','.join(f'{k}={v:g}' for k, v in sorted(params.items())) + ')'

    def prob(self, E, a, indeg):
        m = self.mech
        if m == 'wc':
            w = self.params['w']
            return 1.0 - np.power(1.0 - w, E)
        if m == 'uni':
            c = self.params['c']
            return 1.0 - np.power(1.0 - c, E)
        if m == 'lt':
            alpha = self.params['alpha']
            return np.clip(alpha * a / np.maximum(indeg, 1), 0.0, 1.0)
        if m == 'reinf':
            w = self.params['w']
            return 1.0 - np.power(1.0 - w, E)
        if m == 'decay':
            c = self.params['c']
            return 1.0 - np.power(1.0 - c, E)
        if m == 'gate':
            return (a >= self.params['g']).astype(np.float32)
        raise KeyError(m)

    def edge_weight(self, indeg):
        if self.mech in ('uni', 'decay'):
            return np.full_like(np.asarray(indeg, dtype=np.float64), self.params.get('c', 0.05))
        return 1.0 / np.maximum(indeg, 1)


W_GRID = np.logspace(-3.0, -0.15, 16)
C_GRID = np.logspace(-3.5, -0.5, 16)
ALPHA_GRID = np.logspace(-0.7, 0.5, 16)
RHO_GRID = [0.0, 0.5, 0.9, 1.0]
GATE_GRID = [1, 2, 3, 4, 5]


def candidate_laws(include_gates):
    out = []
    for w in W_GRID:
        out.append(ParamLaw('wc', w=float(w)))
    for c in C_GRID:
        out.append(ParamLaw('uni', c=float(c)))
    for al in ALPHA_GRID:
        out.append(ParamLaw('lt', alpha=float(al)))
    for rho in RHO_GRID:
        for w in W_GRID[::2]:
            out.append(ParamLaw('reinf', w=float(w), rho=rho))
        for c in C_GRID[::2]:
            out.append(ParamLaw('decay', c=float(c), rho=rho))
    if include_gates:
        for g in GATE_GRID:
            out.append(ParamLaw('gate', g=g))
    return out


# --------------------------------------------------------------------- log + fit

def collect_logs(law_true, adj, indeg, T, budget, n_episodes, seed):
    """Episodes under suboptimal behaviour policies; returns per-episode round records."""
    n = adj.shape[0]
    op = law_true.op(adj)
    rng = np.random.default_rng(seed)
    episodes = []
    for ep in range(n_episodes):
        status = np.zeros((1, n), dtype=np.int8)
        E = np.zeros((1, n), dtype=np.float32)
        remaining = budget
        rounds = []
        for t in range(T):
            rounds_left = T - t
            k = 0 if remaining == 0 else (remaining if rounds_left == 1
                                          else int(np.ceil(remaining / rounds_left)))
            k = int(min(k, remaining))
            if k > 0:
                seeds = select_heuristic(BEHAVIOUR[ep % len(BEHAVIOUR)], law_true, adj, op,
                                         indeg, status, E, k, rng=rng)[0]
            else:
                seeds = np.zeros(0, dtype=np.int64)
            status_before = status[0].copy()
            status, E, _, info = transition(law_true, op, indeg, status, E, seeds, rng,
                                           return_info=True)
            rounds.append({'a': info['a'][0].copy(), 'eligible': info['eligible'][0].copy(),
                           'y': info['new'][0].copy(), 'seeded': info['seeded'][0].copy(),
                           'before': status_before, 'after': status[0].copy()})
            remaining -= int(seeds.size)
        episodes.append(rounds)
    return episodes


def loglik(law, episodes, indeg):
    """Summed transition log-likelihood of a candidate law (E recursion uses the candidate's rho)."""
    ll = 0.0
    for rounds in episodes:
        n = indeg.shape[0]
        E = np.zeros(n, dtype=np.float32)
        for r in rounds:
            eligible = r['eligible']
            if not eligible.any():
                continue
            E = law.accumulate(E[None, :], r['a'][None, :])[0]
            # only eligible nodes are predictable: a non-eligible node must contribute 0
            p_full = np.where(eligible, law.prob(E[None, :], r['a'][None, :], indeg)[0], 0.0)
            p = np.clip(p_full.astype(np.float64), 1e-12, 1.0 - 1e-12)
            y = r['y'] & eligible
            ll += float(np.sum(np.where(y, np.log(p), np.log(1.0 - p))))
            E = np.where(y, 0.0, E)
            E[r['after'] != STATUS_INACTIVE] = 0.0
    return ll


def fit_library(episodes_train, episodes_test, indeg, include_gates):
    best = None
    for law in candidate_laws(include_gates):
        ll_train = loglik(law, episodes_train, indeg)
        if best is None or ll_train > best[0]:
            best = (ll_train, law)
    ll_test = loglik(best[1], episodes_test, indeg)
    return best[1], best[0], ll_test


# --------------------------------------------------------------------- experiment

def regret_of(law_plan, law_true, adj, indeg, T, budget, seed, M_greedy=96, M_eval=2000):
    """Plan with `law_plan`, replay under `law_true`, compare with planning under the truth."""
    def best_plan(law):
        plans = []
        for alloc in ['even', 'normal', 'static']:
            p = plan_episode(law, adj, indeg, T, budget, alloc=alloc, select='greedy',
                             M_greedy=M_greedy, seed=seed, period=2)
            v = rollout(law, adj, indeg, [np.asarray(S, np.int64) for S in p], 400, 99,
                        op=law.op(adj)).sum(axis=1).mean()
            plans.append((v, p))
        plans.sort(key=lambda x: -x[0])
        return plans[0][1]

    op_true = law_true.op(adj)
    v_true = rollout(law_true, adj, indeg,
                     [np.asarray(S, np.int64) for S in best_plan(law_true)], M_eval, 7,
                     op=op_true).sum(axis=1).mean()
    v_plan = rollout(law_true, adj, indeg,
                     [np.asarray(S, np.int64) for S in best_plan(law_plan)], M_eval, 7,
                     op=op_true).sum(axis=1).mean()
    return 1.0 - v_plan / v_true, float(v_true), float(v_plan)


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
    args = ap.parse_args()

    ensure_out_dir()
    out = os.path.join(OUT_DIR, f't2_fit_{args.truth}_{args.n}.out')
    law_true = get_law(args.truth)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'T2 fit-from-logs | hidden truth {args.truth} | n={args.n} T={args.T} '
                f'B={args.budget} | {args.episodes} behaviour episodes, '
                f'{int(100 * (1 - args.test_frac))}/{int(100 * args.test_frac)} train/test split\n\n')

    rows = []
    for ds in args.datasets:
        for iseed in range(args.seeds_per_dataset):
            seed = 101 + 17 * iseed
            with Timer(f'instance {ds}-{iseed}'):
                G = load_subgraph(ds, args.n, seed=seed)
                adj, indeg, _ = build_arrays(G)
                episodes = collect_logs(law_true, adj, indeg, args.T, args.budget,
                                        args.episodes, seed=seed)
                n_test = int(round(args.test_frac * len(episodes)))
                ep_test, ep_train = episodes[:n_test], episodes[n_test:]

                law_a, ll_a_train, ll_a_test = fit_library(ep_train, ep_test, indeg, False)
                law_b, ll_b_train, ll_b_test = fit_library(ep_train, ep_test, indeg, True)

                r_wc, v_true, _ = regret_of(get_law('WC'), law_true, adj, indeg,
                                            args.T, args.budget, seed=seed)
                r_a, _, v_a = regret_of(law_a, law_true, adj, indeg, args.T, args.budget, seed=seed)
                r_b, _, v_b = regret_of(law_b, law_true, adj, indeg, args.T, args.budget, seed=seed)

                rows.append([ds, f'{100 * r_wc:+.1f}%', f'{100 * r_a:+.1f}%', f'{100 * r_b:+.1f}%',
                             f'{v_true:.1f}/{v_a:.1f}/{v_b:.1f}', f'{ll_a_test:.0f}', f'{ll_b_test:.0f}',
                             law_a.name[:34], law_b.name[:34]])

    table(rows, ['dataset', 'regret WC', 'regret libA', 'regret libB', 'value true/A/B',
                 'testLL A', 'testLL B', 'law A (IC only)', 'law B (+gates)'], path=out)
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\nlibA = IC family only (wc/uni/lt/reinf/decay);  libB = libA + gate laws\n'
                'regret of the WC default shows what assuming the repo law costs;\n'
                'libB shows whether a library that can express the structure recovers the decision.\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
