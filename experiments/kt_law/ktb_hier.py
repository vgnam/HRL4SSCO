"""KT-b: is there headroom for a high level of set-valued operators (STRATA L2/L3)?

Everything is planned under the TRUE law, so the only question is whether choosing
macro-operators (operator, region, budget) beats the best flat policy (greedy seed
selection + a fixed spending schedule). No law learning is involved.

Policies (all closed-loop: they see the realised state each round)
  flat-<alloc>  greedy seeding (exact expected one-round gain under the law) with
                the schedule alloc in {even, normal, static}
  alloc-only    rollout planner whose only choice is k (greedy over all nodes):
                separates the value of adaptive spending from the value of operators
  hier          rollout planner over (operator, region, k):
                  operators  hub     greedy on expected one-round gain
                             gate-g  threshold coverage: seeds sharing out-neighbours,
                                     pushing inactive nodes towards g active in-neighbours
                             hedge-g alternate hub / gate-g picks
                  regions    whole graph + the 3 largest Louvain communities
                  k          {0, even, 2x even, 4x even, all remaining} plus the base k
The rollout planner (one-step lookahead with base-policy continuation, common random
numbers) always includes the base policy's own action, so it cannot be worse than the
base policy in expectation: any gain is headroom for a high level.

Laws: WC (control: no structure), CC2, CC3, MIX (30% of nodes gated at 2, rest WC).
"""

import argparse
import os
import time

import networkx as nx
import numpy as np

from common import OUT_DIR, build_arrays, ensure_out_dir, load_subgraph, table
from laws import Law, STATUS_ACTIVE, STATUS_INACTIVE, get_law
from planner import _allocation_k
from sim import transition


class MixGate(Law):
    """A fixed share of nodes needs two active in-neighbours; the others follow WC."""
    name = 'MIX'

    def __init__(self, mask):
        self.mask = mask

    def prob(self, E, a, indeg):
        w = 1.0 / np.maximum(indeg, 1)
        return np.where(self.mask, (a >= 2), 1.0 - np.power(1.0 - w, E)).astype(np.float32)


def make_law(name, n, seed):
    if name == 'MIX':
        rng = np.random.default_rng(seed + 4242)
        return MixGate(rng.random(n) < 0.3)
    return get_law(name)


# ------------------------------------------------------------------ grounding

def ground(op, region_mask, k, law, adj, adjT, indeg, status, E):
    """Seed sets for every path. status/E are (M,n); returns (M,k) ids padded with -1."""
    M, n = status.shape
    seeds = -np.ones((M, k), dtype=np.int64)
    if k <= 0:
        return seeds
    kind, g = (op.split('-') + ['0'])[:2]
    g = int(g)
    act = status == STATUS_ACTIVE
    seeded = np.zeros((M, n), dtype=bool)
    rows = np.arange(M)
    for j in range(k):
        a = (act | seeded).astype(np.float32) @ adj
        eligible = (status == STATUS_INACTIVE) & ~seeded
        use = kind
        if kind == 'hedge':
            use = 'hub' if j % 2 == 0 else 'gate'
        if use == 'hub':
            P0 = law.prob(law.accumulate(E, a), a, indeg)
            P1 = law.prob(law.accumulate(E, a + 1), a + 1, indeg)
            D = np.where(eligible, P1 - P0, 0.0)
            gain = np.asarray(D @ adjT) + eligible * (1.0 - P0)
        else:
            # completion counts most, progress towards the threshold counts a little
            comp = eligible & (a == g - 1)
            prog = eligible & (a < g - 1)
            W = comp + 0.3 * prog
            gain = np.asarray(W.astype(np.float32) @ adjT) + 1e-3 * eligible
        cand = (status != STATUS_ACTIVE) & ~seeded & region_mask[None, :]
        gain = np.where(cand, gain, -np.inf)
        best = np.argmax(gain, axis=1)
        ok = np.isfinite(gain[rows, best])
        seeds[ok, j] = best[ok]
        seeded[rows[ok], best[ok]] = True
    return seeds


# ------------------------------------------------------------------ policies

class Ctx:
    def __init__(self, law, adj, indeg, T, B, regions):
        self.law, self.adj, self.indeg, self.T, self.B = law, adj, indeg, T, B
        self.adjT = adj.T.tocsr()
        self.op = law.op(adj)
        self.regions = regions          # list of (name, mask)


def flat_k(alloc, t, T, remaining):
    return int(min(_allocation_k(alloc, t, T - t, remaining, 2), remaining))


def continue_flat(ctx, alloc, status, E, t0, remaining, rng):
    """Vectorised continuation of the flat policy from round t0 on M paths."""
    M = status.shape[0]
    total = np.zeros(M)
    rem = remaining
    full = ctx.regions[0][1]
    for t in range(t0, ctx.T):
        k = flat_k(alloc, t, ctx.T, rem)
        S = ground('hub', full, k, ctx.law, ctx.adj, ctx.adjT, ctx.indeg, status, E)
        status, E, r = transition(ctx.law, ctx.op, ctx.indeg, status, E, S, rng)
        total += r
        rem -= k
    return total


def k_options(alloc, t, T, remaining):
    if remaining <= 0:
        return [0]
    if t == T - 1:
        return [remaining]
    ev = int(np.ceil(remaining / (T - t)))
    ks = {0, min(ev, remaining), min(2 * ev, remaining), min(4 * ev, remaining), remaining,
          flat_k(alloc, t, T, remaining)}
    return sorted(ks)


def rollout_choice(ctx, base_alloc, status_row, E_row, t, remaining, ops, M, seed):
    """One-step lookahead over macros with the flat policy as continuation (CRN)."""
    best = None
    k_base = flat_k(base_alloc, t, ctx.T, remaining)
    cands = []
    for k in k_options(base_alloc, t, ctx.T, remaining):
        if k == 0:
            cands.append(('wait', 0, 0))
            continue
        for op in ops:
            for ri in range(len(ctx.regions)):
                cands.append((op, ri, k))
    cands.append(('hub', 0, k_base))      # the base policy's own action
    seen = set()
    for op, ri, k in cands:
        if (op, ri, k) in seen:
            continue
        seen.add((op, ri, k))
        st = np.repeat(status_row[None, :], M, axis=0).astype(np.int8)
        Es = np.repeat(E_row[None, :], M, axis=0).astype(np.float32)
        S = ground('hub' if op == 'wait' else op, ctx.regions[ri][1], k,
                   ctx.law, ctx.adj, ctx.adjT, ctx.indeg, st[:1], Es[:1])
        S = np.repeat(S, M, axis=0)
        rng = np.random.default_rng(seed)
        st, Es, r = transition(ctx.law, ctx.op, ctx.indeg, st, Es, S, rng)
        v = float((r + continue_flat(ctx, base_alloc, st, Es, t + 1, remaining - k, rng)).mean())
        if best is None or v > best[0] + 1e-9:
            best = (v, op, ri, k, S[0])
    return best


def run_episode(ctx, policy, ep_seed, base_alloc='even', ops=None, M=48, log=None):
    """Closed-loop episode under the true law. Returns total reward."""
    n = ctx.adj.shape[0]
    rng = np.random.default_rng(ep_seed)
    status = np.zeros((1, n), dtype=np.int8)
    E = np.zeros((1, n), dtype=np.float32)
    remaining, total = ctx.B, 0.0
    full = ctx.regions[0][1]
    for t in range(ctx.T):
        if policy.startswith('flat-'):
            k = flat_k(policy[5:], t, ctx.T, remaining)
            S = ground('hub', full, k, ctx.law, ctx.adj, ctx.adjT, ctx.indeg, status, E)
        else:
            _, op, ri, k, S0 = rollout_choice(ctx, base_alloc, status[0], E[0], t, remaining,
                                              ops, M, seed=ep_seed * 1000 + t)
            S = S0[None, :]
            if log is not None:
                log.append((op, ctx.regions[ri][0], k))
        status, E, r = transition(ctx.law, ctx.op, ctx.indeg, status, E, S, rng)
        total += float(r[0])
        remaining -= int((S >= 0).sum())
    return total


# ------------------------------------------------------------------ experiment

def regions_of(G, n, n_clusters=3):
    comms = nx.community.louvain_communities(G.to_undirected(), seed=0)
    comms = sorted(comms, key=len, reverse=True)[:n_clusters]
    out = [('all', np.ones(n, dtype=bool))]
    for i, c in enumerate(comms):
        m = np.zeros(n, dtype=bool)
        m[list(c)] = True
        out.append((f'c{i}', m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--budget', type=int, default=30)
    ap.add_argument('--T', type=int, default=10)
    ap.add_argument('--datasets', nargs='+', default=['wiki', 'dblp', 'reddit'])
    ap.add_argument('--laws', nargs='+', default=['WC', 'CC2', 'CC3', 'MIX'])
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--M', type=int, default=48)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    ensure_out_dir()
    out = os.path.join(OUT_DIR, f'ktb_hier_n{args.n}_B{args.budget}{args.tag}.out')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'KT-b high-level headroom | n={args.n} T={args.T} B={args.budget} | '
                f'{args.episodes} closed-loop episodes per cell, rollout M={args.M} | planned under the true law\n\n')

    ops_hier = ['hub', 'gate-2', 'gate-3', 'hedge-2']
    rows, per_episode, choices = [], [], {}
    for ds in args.datasets:
        G = load_subgraph(ds, args.n, seed=101)
        adj, indeg, _ = build_arrays(G)
        regions = regions_of(G, args.n)
        for lname in args.laws:
            law = make_law(lname, args.n, 101)
            ctx = Ctx(law, adj, indeg, args.T, args.budget, regions)
            t0 = time.time()
            eps = [1000 + e for e in range(args.episodes)]
            flat = {a: np.array([run_episode(ctx, f'flat-{a}', s) for s in eps])
                    for a in ['even', 'normal', 'static']}
            best_alloc = max(flat, key=lambda a: flat[a].mean())
            v_flat = flat[best_alloc]
            v_alloc = np.array([run_episode(ctx, 'rollout', s, base_alloc=best_alloc,
                                            ops=['hub'], M=args.M) for s in eps])
            log = []
            v_hier = np.array([run_episode(ctx, 'rollout', s, base_alloc=best_alloc,
                                           ops=ops_hier, M=args.M, log=log) for s in eps])
            choices[(ds, lname)] = log

            def gap(v):
                d = v - v_flat
                return (f'{100 * d.mean() / v_flat.mean():+.1f}%',
                        f'{100 * d.std(ddof=1) / np.sqrt(len(d)) / v_flat.mean():.1f}')

            g_alloc, se_alloc = gap(v_alloc)
            g_hier, se_hier = gap(v_hier)
            used = [c for c in log if c[2] > 0]
            share_nonhub = np.mean([(c[0] != 'hub') or (c[1] != 'all') for c in used]) if used else 0.0
            rows.append([ds, lname, ' / '.join(f'{flat[a].mean():.1f}' for a in ['even', 'normal', 'static']),
                         best_alloc, f'{v_flat.mean():.1f}', f'{v_alloc.mean():.1f}', g_alloc, se_alloc,
                         f'{v_hier.mean():.1f}', g_hier, se_hier, f'{100 * share_nonhub:.0f}%',
                         f'{time.time() - t0:.0f}s'])
            per_episode.append((ds, lname, v_flat, v_alloc, v_hier))
            print(rows[-1], flush=True)

    table(rows, ['dataset', 'law', 'flat even/normal/static', 'best', 'flat*', 'alloc-only',
                 'gain', 'se', 'hier', 'gain', 'se', 'non-hub macros', 'time'], path=out)

    # pooled over datasets, per law
    pool = []
    for lname in args.laws:
        sel = [p for p in per_episode if p[1] == lname]
        rel_a = np.concatenate([(p[3] - p[2]) / p[2].mean() for p in sel])
        rel_h = np.concatenate([(p[4] - p[2]) / p[2].mean() for p in sel])
        pool.append([lname, f'{100 * rel_a.mean():+.1f}% (se {100 * rel_a.std(ddof=1) / np.sqrt(len(rel_a)):.1f})',
                     f'{100 * rel_h.mean():+.1f}% (se {100 * rel_h.std(ddof=1) / np.sqrt(len(rel_h)):.1f})'])
    with open(out, 'a', encoding='utf-8') as f:
        f.write('\nPooled over datasets (gain vs best flat policy, per-episode relative)\n\n')
    table(pool, ['law', 'alloc-only', 'hier'], path=out)

    with open(out, 'a', encoding='utf-8') as f:
        f.write('\nMacros chosen by hier (operator, region, k), first episode of each cell\n')
        for (ds, lname), log in choices.items():
            f.write(f'  {ds}/{lname}: ' + ' '.join(f'{o}@{r}:{k}' for o, r, k in log[:args.T]) + '\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
