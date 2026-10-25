"""KT-L0 stage 1: which adoption law fits the Higgs-Twitter cascade (no language model).

Data (SNAP higgs-twitter, in IM/datasets/higgs_raw/): follower graph "A B" = A follows B, so information flows
B -> A; activity "A B t type" = A retweets / mentions / replies to B. A user adopts at its first activity as A,
or (for RT) at the time its tweet is retweeted as B. Time is cut into hourly steps (168 steps).

Laws (hazard of adopting at hour h for a user who has not adopted yet; lam_h = external background per hour):
  M0 background   1 - (1 - lam_h)
  M1 IC-new       1 - (1 - lam_h) (1 - p)^a_new                    a_new  = followees adopted in hour h-1
  M2 IC-ever      1 - (1 - lam_h) (1 - p)^a_ever                   a_ever = followees adopted before h
  M3 exposure     1 - (1 - lam_h) (1 - g(min(a_ever, 10)))         g free per exposure count (complex contagion)
  M4 IC-typed     1 - (1 - lam_h) prod_type (1 - p_type)^a_new,type  type of the followee's adoption
  M5 IC-decay     1 - (1 - lam_h) (1 - p)^a_new (1 - p_old)^(a_ever - a_new)
Fitted by maximum likelihood on half of the users, scored on the other half.
"""
import os, time
import torch  # before numpy/pandas: importing it later fails to load its DLLs on Windows
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', '..', 'IM', 'datasets', 'higgs_raw')
CACHE = os.path.join(RAW, 'cells.npz')
TYPES = ['RT', 'MT', 'RE', 'AU']  # AU: author of a retweeted tweet
CAP_NEW, CAP_EVER, CAP_TYPE = 10, 10, 5


def load():
    soc = pd.read_csv(os.path.join(RAW, 'higgs-social_network.edgelist.gz'), sep=' ', header=None,
                      names=['a', 'b'], dtype=np.int32)
    act = pd.read_csv(os.path.join(RAW, 'higgs-activity_time.txt.gz'), sep=' ', header=None,
                      names=['a', 'b', 't', 'type'])
    n = int(max(soc.a.max(), soc.b.max(), act.a.max(), act.b.max())) + 1
    t0 = act.t.min()
    act['h'] = ((act.t - t0) // 3600).astype(int)
    ev = pd.concat([act[['a', 'h', 'type']].rename(columns={'a': 'u'}),
                    act[act.type == 'RT'][['b', 'h']].rename(columns={'b': 'u'}).assign(type='AU')])
    ev = ev.sort_values(['h', 'type']).drop_duplicates('u')
    adopt = np.full(n, 10 ** 6); adopt[ev.u.values] = ev.h.values
    atype = np.full(n, -1); atype[ev.u.values] = [TYPES.index(x) for x in ev.type.values]
    F = sp.csr_matrix((np.ones(len(soc), np.float32), (soc.a.values, soc.b.values)), shape=(n, n))  # follower x followee
    return F, adopt, atype, n


def build_cells(F, adopt, atype, n, H=168, seed=0):
    """Aggregate (user, hour) at-risk rows into cells of identical covariates, per user half."""
    half = np.random.default_rng(seed).random(n) < 0.5
    rows = []
    for h in range(H):
        risk = adopt >= h
        new = adopt == h - 1
        a_new = np.asarray(F @ new.astype(np.float32)).ravel()
        a_ever = np.asarray(F @ (adopt <= h - 1).astype(np.float32)).ravel()
        typed = [np.asarray(F @ (new & (atype == k)).astype(np.float32)).ravel() for k in range(4)]
        idx = np.flatnonzero(risk)
        key = np.stack([np.full(len(idx), h), half[idx], np.minimum(a_new[idx], CAP_NEW),
                        np.minimum(a_ever[idx], CAP_EVER * 3)]
                       + [np.minimum(t[idx], CAP_TYPE) for t in typed]
                       + [adopt[idx] == h], 1).astype(np.int32)
        uniq, cnt = np.unique(key, axis=0, return_counts=True)
        rows.append(np.c_[uniq, cnt])
    return np.concatenate(rows)


def to_tables(cells):
    """Columns: h, half, a_new, a_ever, n_RT, n_MT, n_RE, n_AU, y -> merge y=0/1 into (N, Y)."""
    df = pd.DataFrame(cells, columns=['h', 'half', 'a_new', 'a_ever', 'RT', 'MT', 'RE', 'AU', 'y', 'cnt'])
    df['Y'] = df.cnt * df.y
    g = df.groupby(['h', 'half', 'a_new', 'a_ever', 'RT', 'MT', 'RE', 'AU'], as_index=False).agg(N=('cnt', 'sum'), Y=('Y', 'sum'))
    return g


def survival(model, w, d):
    a_new, a_ever = d.a_new.values, d.a_ever.values
    if model == 'M0': return np.ones(len(d))
    if model == 'M1': return (1 - expit(w[0])) ** a_new
    if model == 'M2': return (1 - expit(w[0])) ** a_ever
    if model == 'M3':
        g = np.r_[0, expit(w[:CAP_EVER])]
        return 1 - g[np.minimum(a_ever, CAP_EVER)]
    if model == 'M4':
        s = np.ones(len(d))
        for k, t in enumerate(TYPES): s *= (1 - expit(w[k])) ** d[t].values
        return s
    if model == 'M5': return (1 - expit(w[0])) ** a_new * (1 - expit(w[1])) ** (a_ever - a_new)


N_SOCIAL = {'M0': 0, 'M1': 1, 'M2': 1, 'M3': CAP_EVER, 'M4': 4, 'M5': 2}


def nll(params, model, d, H):
    lam = expit(params[:H])[d.h.values]
    s = survival(model, params[H:], d)
    p = np.clip(1 - (1 - lam) * s, 1e-12, 1 - 1e-12)
    return -(d.Y.values * np.log(p) + (d.N.values - d.Y.values) * np.log(1 - p)).sum()


def fit(model, d, H):
    """Maximum likelihood with exact gradients (torch autograd), same objective as nll()."""
    tt = lambda a: torch.tensor(np.asarray(a, dtype=np.float64))
    h, N, Y = torch.tensor(d.h.values), tt(d.N.values), tt(d.Y.values)
    a_new, a_ever = tt(d.a_new.values), tt(d.a_ever.values)
    typed = [tt(d[t].values) for t in TYPES]
    x = torch.tensor(np.r_[np.full(H, -10.0), np.full(N_SOCIAL[model], -5.0)], requires_grad=True)

    def loss():
        lam, w = torch.sigmoid(x[:H])[h], x[H:]
        if model == 'M0': s = torch.ones_like(N)
        elif model == 'M1': s = (1 - torch.sigmoid(w[0])) ** a_new
        elif model == 'M2': s = (1 - torch.sigmoid(w[0])) ** a_ever
        elif model == 'M3':
            g = torch.cat([torch.zeros(1, dtype=torch.float64), torch.sigmoid(w)])
            s = 1 - g[torch.clamp(a_ever, max=CAP_EVER).long()]
        elif model == 'M4':
            s = torch.ones_like(N)
            for k in range(4): s = s * (1 - torch.sigmoid(w[k])) ** typed[k]
        elif model == 'M5': s = (1 - torch.sigmoid(w[0])) ** a_new * (1 - torch.sigmoid(w[1])) ** (a_ever - a_new)
        p = torch.clamp(1 - (1 - lam) * s, 1e-12, 1 - 1e-12)
        return -(Y * torch.log(p) + (N - Y) * torch.log(1 - p)).sum()

    opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(); l = loss(); l.backward(); return l
    for _ in range(5): opt.step(closure)
    return x.detach().numpy()


if __name__ == '__main__':
    t0 = time.time()
    if os.path.exists(CACHE):
        cells = np.load(CACHE)['cells']
    else:
        F, adopt, atype, n = load()
        print(f"users {n}, follow edges {F.nnz}, adopters {(adopt < 10**6).sum()} "
              f"by type {[int((atype == k).sum()) for k in range(4)]}  [{time.time()-t0:.0f}s]", flush=True)
        cells = build_cells(F, adopt, atype, n); np.savez_compressed(CACHE, cells=cells)
    tab = to_tables(cells); H = 168
    train, test = tab[tab.half == 1], tab[tab.half == 0]
    print(f"cells {len(tab)}, at-risk user-hours {tab.N.sum():.3g}, adoptions {tab.Y.sum():.0f}  [{time.time()-t0:.0f}s]", flush=True)
    res = {}
    for m in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
        w = fit(m, train, H); res[m] = w
        ll_tr, ll_te = -nll(w, m, train, H), -nll(w, m, test, H)
        k = H + N_SOCIAL[m]; bic = -2 * ll_tr + k * np.log(train.N.sum())
        soc = expit(w[H:])
        print(f"{m}: test LL {ll_te:12.1f}  (vs M0 {ll_te + nll(res['M0'], 'M0', test, H):+9.1f})  BIC {bic:12.1f}  "
              f"social params {np.round(soc, 5).tolist()}", flush=True)
    np.savez(os.path.join(RAW, 'fits.npz'), **res)
    # empirical exposure curve: adoption rate per hour by number of adopted followees
    e = tab.groupby(np.minimum(tab.a_ever, 15)).agg(N=('N', 'sum'), Y=('Y', 'sum'))
    print("exposure curve P(adopt this hour | a_ever=k):", ' '.join(f"{k}:{y/n:.4f}" for k, (n, y) in e.iterrows()))
