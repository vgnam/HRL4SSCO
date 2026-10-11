"""Belief over the hidden grounding (method Sections 3.3-3.5).

Per consumer, an exact posterior over joint slot hypotheses h = (mu_1, ..., mu_J) (and quorums),
with prior weight  W_v(h; eta) = prod_j pi0(h_j) * lambda(h_j; eta)  and hard elimination by every
observation made while the consumer was up (Proposition 1). Weights are plain float64 products in
slot order, normalised after masking with the surviving hypotheses (as the reference does).
"""
import itertools
import math

import numpy as np


def _product(sizes):
    hyps = list(itertools.product(*[range(k) for k in sizes])) if sizes else [()]
    return np.array(hyps, int).reshape(len(hyps), len(sizes))


class _Consumer:
    """Hypothesis space of one consumer: options per slot are (mask, quorum) pairs."""
    __slots__ = ("kind", "H", "masks", "qs", "kidx", "alive", "fixed")

    def __init__(self, kind, masks, qs, kidx, fixed=None):
        self.kind, self.masks, self.qs, self.kidx, self.fixed = kind, masks, qs, kidx, fixed
        self.H = _product([len(m) for m in masks])
        self.alive = np.ones(len(self.H), bool)

    def copy(self):
        c = _Consumer.__new__(_Consumer)
        c.kind, c.H, c.masks, c.qs, c.kidx, c.fixed = self.kind, self.H, self.masks, self.qs, self.kidx, self.fixed
        c.alive = self.alive.copy()
        return c


def blind_space(slot_lens, p=1 / 3):
    """Unstructured space: every non-empty link subset per slot, independent prior p per link
    (reference probe_supply2.VersionSpace(kind="blind")); quorum 1."""
    masks, fixed = [], []
    for k in slot_lens:
        ms, pr = [], []
        for code in range(1, 2 ** k):
            m = np.array([(code >> b) & 1 for b in range(k)], bool)
            ms.append(m)
            pr.append(p ** m.sum() * (1 - p) ** (k - m.sum()))
        masks.append(ms)
        fixed.append(np.array(pr))
    return _Consumer("blind", masks, [[1] * len(m) for m in masks], [list(range(len(m))) for m in masks], fixed)


class Belief:
    def __init__(self, net, spaces, eta=0.2, quorum=False, use_text=True, form="channel", p_dual=0.3,
                 d_drop=0.15, d_add=0.05, zeta=1e-3, escape=True, blind_p=1 / 3):
        self.net, self.spaces = net, spaces
        self.eta, self.d_drop, self.d_add, self.p_dual = eta, d_drop, d_add, p_dual
        self.quorum, self.use_text, self.form, self.zeta = quorum, use_text, form, zeta
        self.escape_enabled, self.blind_p = escape, blind_p
        self.history = []                                   # every observation (up, op)
        self.cons = sorted(spaces.by_consumer)
        self.sp = {}
        for v in self.cons:
            masks, qs, kidx = [], [], []
            customer = net.layer[v] == 3
            for s in spaces.by_consumer[v]:
                sl = spaces.slots[s]
                ms, q_, k_ = [], [], []
                for k, m in enumerate(sl.masks):
                    ms.append(m); q_.append(1); k_.append(k)
                    if quorum and m.sum() >= 2 and not customer and p_dual > 0:
                        ms.append(m); q_.append(2); k_.append(k)
                masks.append(ms); qs.append(np.array(q_)); kidx.append(np.array(k_))
            self.sp[v] = _Consumer("schema", masks, qs, kidx)

    # ------------------------------------------------------------------ prior
    def _in_S(self, sl):
        """Slots whose text enters the channel and the EM sums."""
        if not self.use_text or sl.stated is None:
            return False
        if self.form == "channel":
            return sl.K >= 2
        return sl.stated >= 0 and len(sl.masks) > 1

    def _factor(self, v, j):
        c = self.sp[v]
        if c.kind == "blind":
            return c.fixed[j]
        sl = self.spaces.slots[self.spaces.by_consumer[v][j]]
        k = c.kidx[j]
        if self.form == "prior" and self.use_text and sl.stated is not None and sl.stated >= 0 and len(sl.masks) > 1:
            rest = sum(p for i, p in enumerate(sl.prior) if i != sl.stated)
            base = np.array([(1 - self.eta) if i == sl.stated else self.eta * sl.prior[i] / rest for i in k])
        else:
            base = sl.prior[k]
            if self.form == "channel" and self._in_S(sl):
                base = base * np.where(k == sl.stated, 1 - self.eta, self.eta / (sl.K - 1))
        if self.quorum:
            sizes = np.array([sl.masks[i].sum() for i in k])
            p2 = np.where((sizes >= 2) & (self.net.layer[v] != 3), self.p_dual, 0.0)
            q = c.qs[j]
            f = np.where(q == 2, p2, 1 - p2)
            if sl.stated is not None and self.use_text:
                if sl.quorum_stated == 2:
                    f = f * np.where(q == 2, 1 - self.d_drop, self.d_add)
                else:
                    f = f * np.where(q == 2, self.d_drop, 1 - self.d_add)
            base = base * f
        return base

    def prior_weights(self, v):
        c = self.sp[v]
        w = np.ones(len(c.H))
        for j in range(c.H.shape[1]):
            w *= self._factor(v, j)[c.H[:, j]]
        return w

    def weights(self, v):
        w = self.prior_weights(v) * self.sp[v].alive
        return w / w.sum()

    # ------------------------------------------------------------------ observations
    def _consistent(self, v, c, op):
        pred = np.ones(len(c.H), bool)
        for j, s in enumerate(self.spaces.by_consumer[v]):
            ol = op[self.net.links_of_slot(s)]
            sat = np.array([int((m & ol).sum()) >= q for m, q in zip(c.masks[j], c.qs[j])])
            pred &= sat[c.H[:, j]]
        return pred == bool(op[v])

    def _apply(self, v, op):
        c = self.sp[v]
        keep = c.alive & self._consistent(v, c, op)
        if keep.any():
            c.alive = keep
        elif c.kind == "schema" and self.escape_enabled:
            self._escape(v)
        # otherwise the observation is ignored (it would eliminate every hypothesis)

    def _escape(self, v):
        """The program is wrong about v: switch permanently to the unstructured space (Section 3.5)."""
        lens = [len(self.net.links_of_slot(s)) for s in self.spaces.by_consumer[v]]
        self.sp[v] = blind_space(lens, self.blind_p)
        for up, op in self.history:
            if up[v]:
                c = self.sp[v]
                keep = c.alive & self._consistent(v, c, op)
                if keep.any():
                    c.alive = keep

    def observe(self, up, op, store=True):
        if store:
            self.history.append((up.copy(), op.copy()))
        for v in self.cons:
            if up[v]:
                self._apply(v, op)

    @property
    def escaped(self):
        return [v for v in self.cons if self.sp[v].kind == "blind"]

    # ------------------------------------------------------------------ EM (Section 3.4)
    def em(self, a=1.0, b=4.0, iters=3):
        for _ in range(iters):
            num, den = a, a + b
            dn, dd, an, ad = 1.5, 10.0, 0.5, 10.0
            for v in self.cons:
                c = self.sp[v]
                if c.kind != "schema":
                    continue
                w = None
                for j, s in enumerate(self.spaces.by_consumer[v]):
                    sl = self.spaces.slots[s]
                    if self._in_S(sl):
                        w = self.weights(v) if w is None else w
                        num += float(w[c.kidx[j][c.H[:, j]] != sl.stated].sum())
                        den += 1.0
                    if self.quorum and self.use_text and sl.stated is not None:
                        w = self.weights(v) if w is None else w
                        b2 = float(w[c.qs[j][c.H[:, j]] == 2].sum())
                        sv = 1.0 if sl.quorum_stated == 2 else 0.0
                        dn += (1 - sv) * b2; dd += b2
                        an += sv * (1 - b2); ad += 1 - b2
            self.eta = num / den
            if self.quorum and self.use_text:
                self.d_drop, self.d_add = dn / dd, an / ad
        return self.eta

    # ------------------------------------------------------------------ model for the planner
    def map_model(self):
        """MAP hypothesis per consumer (first maximiser in enumeration order) as flat arrays."""
        net = self.net
        compat = np.zeros(net.n_links, bool)
        quorum = np.ones(net.n_slots, int)
        for v in self.cons:
            c = self.sp[v]
            h = c.H[int(np.argmax(self.weights(v)))]
            for j, s in enumerate(self.spaces.by_consumer[v]):
                a = net.slot_start[s]
                compat[a:a + net.slot_len[s]] = c.masks[j][h[j]]
                quorum[s] = c.qs[j][h[j]]
        return compat, (quorum if self.quorum else None)

    def slot_posterior(self, v, j):
        """Posterior over the masks of slot j: {mask index: probability} (schema space only)."""
        c = self.sp[v]
        w = self.weights(v)
        out = {}
        for k, p in zip(c.kidx[j][c.H[:, j]], w):
            out[int(k)] = out.get(int(k), 0.0) + float(p)
        return out

    # ------------------------------------------------------------------ marginal likelihood
    def log_marginal(self):
        """L(sigma, eta) = sum_v log[(1 - zeta) m_v^schema + zeta m_v^blind] on all stored observations."""
        total = 0.0
        for v in self.cons:
            c = self.sp[v]
            if c.kind == "schema":
                wb = self.prior_weights(v)
                m_schema = float((wb * c.alive).sum() / wb.sum())
            else:
                m_schema = 0.0
            lens = [len(self.net.links_of_slot(s)) for s in self.spaces.by_consumer[v]]
            bl = blind_space(lens, self.blind_p)
            for up, op in self.history:
                if up[v]:
                    bl.alive &= self._consistent(v, bl, op)
            wb = np.ones(len(bl.H))
            for j in range(bl.H.shape[1]):
                wb *= bl.fixed[j][bl.H[:, j]]
            m_blind = float((wb * bl.alive).sum() / wb.sum())
            total += math.log(max((1 - self.zeta) * m_schema + self.zeta * m_blind, 1e-300))
        return total

    # ------------------------------------------------------------------ revision statistics (Section 4.3)
    def key_statistics(self, a=1.0, b=4.0, n_examples=20):
        rows = []                                           # (key, v, j, r)
        for v in self.cons:
            c = self.sp[v]
            if c.kind != "schema":
                continue
            w = None
            for j, s in enumerate(self.spaces.by_consumer[v]):
                sl = self.spaces.slots[s]
                if self._in_S(sl):
                    w = self.weights(v) if w is None else w
                    rows.append((sl.key, v, j, float(w[c.kidx[j][c.H[:, j]] != sl.stated].sum())))
        esc = set(self.escaped)
        keys = {sl.key for sl in self.spaces.slots if sl.key is not None}
        out = {}
        for k in keys:
            Sk = [r for r in rows if r[0] == k]
            rest = [r for r in rows if r[0] != k]
            ck = sum(r[3] for r in Sk)
            eta_mk = (a + sum(r[3] for r in rest)) / (a + b + len(rest))
            owners = {sl.v for sl in self.spaces.slots if sl.key == k}
            ex_hi = [(r[1], r[2]) for r in Sk if r[3] > 0.9][:n_examples]
            ex_hi += [(sl.v, sl.j) for sl in self.spaces.slots if sl.key == k and sl.v in esc][:max(0, n_examples - len(ex_hi))]
            out[k] = dict(S=len(Sk), c=ck, eta_minus=eta_mk, e=len(owners & esc), N=len(owners),
                          high=ex_hi, low=[(r[1], r[2]) for r in Sk if r[3] < 0.1][:n_examples])
        return out

    def rebuild(self, spaces, em_iters=20) -> "Belief":
        """Belief under a new program, with every stored observation replayed and eta re-estimated."""
        b = Belief(self.net, spaces, self.eta, self.quorum, self.use_text, self.form, self.p_dual,
                   self.d_drop, self.d_add, self.zeta, self.escape_enabled, self.blind_p)
        b.history = list(self.history)
        for up, op in b.history:
            b.observe(up, op, store=False)
        b.em(iters=em_iters)
        return b

    def clone(self):
        b = Belief.__new__(Belief)
        b.__dict__.update(self.__dict__)
        b.history = list(self.history)
        b.sp = {v: c.copy() for v, c in self.sp.items()}
        return b


def flag_keys(stats):
    """Keys flagged for revision (Section 4.3)."""
    flagged = []
    rates = {k: s["e"] / s["N"] for k, s in stats.items() if s["N"]}
    top = max(rates.values()) if rates else 0.0
    for k, s in stats.items():
        contra = (s["S"] >= 20 and s["c"] >= 5 and
                  s["c"] / s["S"] > s["eta_minus"] + 3 * math.sqrt(s["eta_minus"] * (1 - s["eta_minus"]) / s["S"]))
        escapes = s["e"] >= 2 and s["N"] and s["e"] / s["N"] >= top
        if contra or escapes:
            flagged.append(k)
    return flagged
