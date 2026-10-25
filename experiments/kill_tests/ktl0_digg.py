"""KT-L0 on Digg 2009 (Hogg & Lerman), stage 1: which adoption law fits, and does it differ across stories?

Data (Network Repository "digg", in IM/datasets/digg_raw/raw/): digg.edges holds user-user friend links "u,v,t"
(u watches v, so u sees v's votes) and user-story votes "u,s,t"; digg.types gives node types (1 user, 2 story).
For every story, hours are counted from its first vote (H hours). A user who has not voted yet is at risk; its
exposures are the votes of the users it watches. Every story has its own background rate per hour (front page,
search, ...), which seeding cannot change.

Laws (as ktl0_higgs.py): M0 background, M1 IC on new exposures, M2 IC on all exposures, M3 free exposure curve
g(k), M4 IC with separate p for mutual and one-way friends, M5 IC with decay, and MS: a free curve per story
(g_s(1), g_s(2), g_s(3+)), from which the complexity index rho_s = g_s(2) / (1 - (1 - g_s(1))^2) is read
(rho > 1: two exposures do more than two independent ones, i.e. complex contagion).
"""
import os, time
import torch  # before numpy/pandas: importing it later fails to load its DLLs on Windows
import numpy as np, pandas as pd, scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', '..', 'IM', 'datasets', 'digg_raw', 'raw')
CACHE = os.path.join(RAW, 'cells.npz')
H, CAP_NEW, CAP_EVER, CAP_MUT, CAP_MS = 48, 10, 30, 5, 3


def load():
    e = pd.read_csv(os.path.join(RAW, 'digg.edges'), header=None, names=['u', 'v', 't'])
    ty = pd.read_csv(os.path.join(RAW, 'digg.types'), header=None, names=['id', 'type']).set_index('id')['type']
    tu, tv = ty.reindex(e.u).values, ty.reindex(e.v).values
    fr = e[(tu == 1) & (tv == 1)][['u', 'v']].drop_duplicates()
    votes = e[(tu == 1) & (tv == 2)].sort_values('t').drop_duplicates(['u', 'v'])
    n = int(e[['u', 'v']].values.max()) + 1
    W = sp.csr_matrix((np.ones(len(fr), np.int8), (fr.v.values, fr.u.values)), shape=(n, n))  # W[v, u]: u watches v
    mutual = W.multiply(W.T).tocsr()
    return W, mutual, votes, n


def build_cells(W, mutual, votes, n, seed=0):
    half = np.random.default_rng(seed).random(n) < 0.5
    P = votes.u.nunique()                                       # population: users who voted at least once
    out, groups = [], list(votes.groupby('v'))
    stories = np.array([s for s, _ in groups])
    for si, (s, sv) in enumerate(groups):
        hw = ((sv.t.values - sv.t.values.min()) // 3600).astype(int)
        voter, tau_all = sv.u.values, np.minimum(hw, H)
        src = voter[hw < H]; src_h = hw[hw < H]
        cnt = W.indptr[src + 1] - W.indptr[src]
        ev_u = np.concatenate([W.indices[W.indptr[w]:W.indptr[w + 1]] for w in src]) if cnt.sum() else np.zeros(0, int)
        ev_h = np.repeat(src_h, cnt)
        mcnt = mutual.indptr[src + 1] - mutual.indptr[src]
        mu_u = np.concatenate([mutual.indices[mutual.indptr[w]:mutual.indptr[w + 1]] for w in src]) if mcnt.sum() else np.zeros(0, int)
        mu_h = np.repeat(src_h, mcnt)
        users, ui = np.unique(ev_u, return_inverse=True)
        E = np.zeros((len(users), H), np.int32); np.add.at(E, (ui, ev_h), 1)
        Em = np.zeros((len(users), H), np.int32)
        if len(mu_u):
            keep = np.isin(mu_u, users); np.add.at(Em, (np.searchsorted(users, mu_u[keep]), mu_h[keep]), 1)
        tau = np.full(len(users), H); vpos = np.isin(voter, users)
        tau[np.searchsorted(users, voter[vpos])] = tau_all[vpos]
        hh = np.arange(H)
        a_ever = np.c_[np.zeros(len(users), np.int32), np.cumsum(E, 1)[:, :-1]]
        a_new = np.c_[np.zeros(len(users), np.int32), E[:, :-1]]
        m_new = np.c_[np.zeros(len(users), np.int32), Em[:, :-1]]
        alive = hh[None, :] <= tau[:, None]; y = hh[None, :] == tau[:, None]
        soc = alive & (a_ever > 0)
        r, c = np.nonzero(soc)
        key = np.stack([np.full(len(r), si), c, half[users[r]], np.minimum(a_new[r, c], CAP_NEW),
                        np.minimum(a_ever[r, c], CAP_EVER), np.minimum(m_new[r, c], CAP_MUT), y[r, c]], 1)
        uniq, k = np.unique(key, axis=0, return_counts=True)
        out.append(np.c_[uniq, k])
        # zero-exposure users per hour, split by half in proportion to the population
        voted_before = np.array([(tau_all < h).sum() for h in hh]); votes_at = np.array([(tau_all == h).sum() for h in hh])
        soc_n, soc_y = soc.sum(0), (soc & y).sum(0)
        for h in range(H):
            n0, y0 = P - voted_before[h] - soc_n[h], votes_at[h] - soc_y[h]
            for hf, frac in ((1, 0.5), (0, 0.5)):
                y0h = y0 // 2 + (y0 % 2 if hf == 1 else 0)
                out.append(np.array([[si, h, hf, 0, 0, 0, 1, y0h], [si, h, hf, 0, 0, 0, 0, int(n0 * frac) - y0h]]))
        if si % 500 == 0: print(f"  story {si}/{len(stories)}", flush=True)
    return np.concatenate(out), stories


def table(cells):
    df = pd.DataFrame(cells, columns=['s', 'h', 'half', 'a_new', 'a_ever', 'm_new', 'y', 'cnt'])
    df['Y'] = df.cnt * df.y
    return df.groupby(['s', 'h', 'half', 'a_new', 'a_ever', 'm_new'], as_index=False).agg(N=('cnt', 'sum'), Y=('Y', 'sum'))


N_SOCIAL = {'M0': 0, 'M1': 1, 'M2': 1, 'M3': 10, 'M4': 2, 'M5': 2}


def fit(model, d, S, x_init=None, fix_social=False):
    """Maximum likelihood (torch autograd + L-BFGS). fix_social: keep the social parameters of x_init and refit
    only the story backgrounds (used to score a law on held-out users)."""
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float64))
    lam_idx = torch.tensor(d.s.values * H + d.h.values)
    N, Y, a_new, a_ever, m_new = t(d.N.values), t(d.Y.values), t(d.a_new.values), t(d.a_ever.values), t(d.m_new.values)
    sidx = torch.tensor(d.s.values)
    n_soc = 3 * S if model == 'MS' else N_SOCIAL[model]
    x0 = np.r_[np.full(S * H, -8.0), np.full(n_soc, -5.0)] if x_init is None else x_init
    x = torch.tensor(x0[:S * H] if fix_social else x0, requires_grad=True)
    w_fixed = torch.tensor(x0[S * H:]) if fix_social else None

    def loss():
        lam = torch.sigmoid(x[:S * H])[lam_idx]
        w = w_fixed if fix_social else x[S * H:]
        sg = torch.sigmoid
        if model == 'M0': s = torch.ones_like(N)
        elif model == 'M1': s = (1 - sg(w[0])) ** a_new
        elif model == 'M2': s = (1 - sg(w[0])) ** a_ever
        elif model == 'M3':
            g = torch.cat([torch.zeros(1, dtype=torch.float64), sg(w)]); s = 1 - g[torch.clamp(a_ever, max=10).long()]
        elif model == 'M4': s = (1 - sg(w[0])) ** m_new * (1 - sg(w[1])) ** (a_new - m_new)
        elif model == 'M5': s = (1 - sg(w[0])) ** a_new * (1 - sg(w[1])) ** (a_ever - a_new)
        elif model == 'MS':
            g = torch.cat([torch.zeros(S, 1, dtype=torch.float64), sg(w).view(S, 3)], 1)
            s = 1 - g[sidx, torch.clamp(a_ever, max=CAP_MS).long()]
        p = torch.clamp(1 - (1 - lam) * s, 1e-12, 1 - 1e-12)
        return -(Y * torch.log(p) + (N - Y) * torch.log(1 - p)).sum()

    opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50,
                            line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); l = loss(); l.backward(); return l
    for _ in range(5): opt.step(closure)
    with torch.no_grad(): return x.detach().numpy(), float(loss())


if __name__ == '__main__':
    t0 = time.time()
    if os.path.exists(CACHE):
        z = np.load(CACHE); cells, stories = z['cells'], z['stories']
    else:
        W, mutual, votes, n = load()
        print(f"friend links {W.nnz} (mutual {mutual.nnz // 2} pairs), votes {len(votes)}, stories {votes.v.nunique()}  [{time.time()-t0:.0f}s]", flush=True)
        cells, stories = build_cells(W, mutual, votes, n); np.savez_compressed(CACHE, cells=cells, stories=stories)
    tab = table(cells); S = len(stories)
    tr, te = tab[tab.half == 1].reset_index(drop=True), tab[tab.half == 0].reset_index(drop=True)
    print(f"cells {len(tab)}, at-risk user-hours {tab.N.sum():.3g}, votes {tab.Y.sum():.0f}, exposed user-hours {tab[tab.a_ever > 0].N.sum():.3g}  [{time.time()-t0:.0f}s]", flush=True)
    res, base = {}, None
    for m in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'MS']:
        x, _ = fit(m, tr, S); res[m] = x
        # held-out users: social part fixed, story backgrounds refit
        _, nll_te = fit(m, te, S, x, fix_social=True)
        if base is None: base = nll_te
        soc = 1 / (1 + np.exp(-x[S * H:]))
        shown = np.round(soc, 5).tolist() if m != 'MS' else f"median g(1),g(2),g(3+) = {np.round(np.median(soc.reshape(S, 3), 0), 5).tolist()}"
        print(f"{m}: test NLL {nll_te:12.1f} (vs M0 {base - nll_te:+9.1f})  social {shown}  [{time.time()-t0:.0f}s]", flush=True)
    np.savez(os.path.join(RAW, 'fits.npz'), stories=stories, **res)
    # per-story complexity index from MS (stories with enough exposed user-hours at 2 exposures)
    g = (1 / (1 + np.exp(-res['MS'][S * H:]))).reshape(S, 3)
    rho = g[:, 1] / (1 - (1 - g[:, 0]) ** 2)
    n2 = tab[tab.a_ever == 2].groupby('s').N.sum().reindex(range(S)).fillna(0).values
    ok = n2 >= 2000
    print(f"stories with >= 2000 user-hours at 2 exposures: {ok.sum()}; rho quantiles 10/25/50/75/90%: "
          f"{np.round(np.quantile(rho[ok], [.1, .25, .5, .75, .9]), 2).tolist()}; share rho > 1.5: {np.mean(rho[ok] > 1.5):.2f}")
    np.save(os.path.join(RAW, 'rho.npy'), np.c_[rho, n2])
