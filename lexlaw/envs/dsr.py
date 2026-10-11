"""DSR: dependency-constrained supply-chain recovery (method Sections 2.1-2.5).

The generator, random-stream layout and dynamics follow the reference scripts exactly, so that
returns are reproducible to the printed digit (Section 10.3).

Models are flat arrays aligned with the network's link and slot lists:
    compat[e]  bool, link e is compatible with its slot           (length net.n_links)
    quorum[s]  int, number of compatible operating suppliers slot s needs   (length net.n_slots)
"""
from dataclasses import dataclass, field
import hashlib

import numpy as np

LAYERS = (90, 90, 80, 40)
N_CAT, N_GRADE, LINKS = 6, 3, 4
T_DEFAULT, RHO, PHI, DOWN0 = 15, 0.9, 0.01, 0.35


@dataclass
class Network:
    index: int
    n: int
    layer: np.ndarray            # layer of every node
    idx: list                    # node indices per layer
    slots: dict                  # consumer -> [(links, true mask), ...]
    w: np.ndarray                # customer weights (0 for producers)
    grade: np.ndarray            # hidden supplier class (-1 for customers)
    cat: np.ndarray              # hidden supplier category (-1 for customers)
    quorum: dict = None          # consumer -> [q per slot] (variant Q), else None
    # flat structure, built by _index()
    slot_owner: np.ndarray = field(default=None, repr=False)
    slot_start: np.ndarray = field(default=None, repr=False)   # first link of each slot
    slot_len: np.ndarray = field(default=None, repr=False)
    link_node: np.ndarray = field(default=None, repr=False)
    link_slot: np.ndarray = field(default=None, repr=False)
    slots_of: dict = field(default=None, repr=False)           # consumer -> list of flat slot ids
    producers: np.ndarray = field(default=None, repr=False)

    @property
    def n_slots(self):
        return len(self.slot_owner)

    @property
    def n_links(self):
        return len(self.link_node)

    def true_compat(self):
        return np.concatenate([m for v in sorted(self.slots) for _, m in self.slots[v]]).astype(bool)

    def true_quorum(self):
        if self.quorum is None:
            return np.ones(self.n_slots, int)
        return np.array([q for v in sorted(self.slots) for q in self.quorum[v]], int)

    def links_of_slot(self, s):
        a = self.slot_start[s]
        return self.link_node[a:a + self.slot_len[s]]

    def fingerprint(self):
        """md5 of the network (Section 10.3)."""
        h = hashlib.md5()
        for v in sorted(self.slots):
            for links, m in self.slots[v]:
                h.update(np.asarray(links, np.int64).tobytes())
                h.update(np.asarray(m, np.uint8).tobytes())
        h.update(np.round(self.w, 6).astype(np.float64).tobytes())
        return h.hexdigest()


def _index(net):
    owner, start, length, lnode, lslot, slots_of = [], [], [], [], [], {}
    for v in sorted(net.slots):
        slots_of[v] = []
        for links, _ in net.slots[v]:
            s = len(owner)
            slots_of[v].append(s)
            owner.append(v)
            start.append(len(lnode))
            length.append(len(links))
            lnode.extend(int(u) for u in links)
            lslot.extend([s] * len(links))
    net.slot_owner = np.array(owner, int)
    net.slot_start = np.array(start, int)
    net.slot_len = np.array(length, int)
    net.link_node = np.array(lnode, int)
    net.link_slot = np.array(lslot, int)
    net.slots_of = slots_of
    net.producers = np.where(net.layer < 3)[0]
    return net


def make_network(index: int, variant="C", p_dual=0.3) -> Network:
    """Network `index`, generator seed 100 + index (Section 2.1); variant "Q" adds quorums (2.4)."""
    rng = np.random.default_rng(100 + index)
    layer = np.repeat(np.arange(4), LAYERS)
    n = len(layer)
    idx = [np.where(layer == i)[0] for i in range(4)]
    cat = np.full(n, -1)
    grade = np.full(n, -1)
    for i in range(3):
        cat[idx[i]] = rng.integers(0, N_CAT, len(idx[i]))
        grade[idx[i]] = rng.integers(0, N_GRADE, len(idx[i]))
    slots = {}
    for i in range(1, 4):
        for v in idx[i]:
            n_slots = 1 if i == 3 else int(rng.integers(2, 4))
            cats = rng.choice(N_CAT, n_slots, replace=False)
            sl = []
            for c in cats:
                pool = idx[i - 1][cat[idx[i - 1]] == c]
                if len(pool) == 0:
                    continue
                links = rng.choice(pool, min(LINKS, len(pool)), replace=False)
                need = grade[rng.choice(links)]
                sl.append((links, grade[links] == need))
            slots[int(v)] = sl
    w = np.zeros(n)
    w[idx[3]] = rng.uniform(1, 3, LAYERS[3])
    net = _index(Network(index, n, layer, idx, slots, w, grade, cat))
    net.rng_after = rng                                       # the reference draws noisy models from it
    if variant == "Q":
        net.quorum = add_quorum(net, np.random.default_rng(900 + index), p_dual)
    return net


def add_quorum(net, rng, p_dual):
    """Slot quorums of DSR-Q; a random number is drawn only for eligible slots (Section 2.4)."""
    quo = {}
    for v, sl in net.slots.items():
        dual_ok = net.layer[v] in (1, 2)
        quo[v] = [2 if (dual_ok and m.sum() >= 2 and rng.random() < p_dual) else 1 for _, m in sl]
    return quo


def operating(net, up, compat, quorum=None) -> np.ndarray:
    """op = Op_m(up), layer by layer (Section 2.2; with quorum, Section 2.4)."""
    op = np.zeros(net.n, bool)
    op[net.idx[0]] = up[net.idx[0]]
    for i in range(1, 4):
        good = op[net.link_node] & compat
        cnt = np.bincount(net.link_slot, weights=good, minlength=net.n_slots)
        ok = cnt >= (1 if quorum is None else quorum)
        bad = np.bincount(net.slot_owner, weights=~ok, minlength=net.n)
        v = net.idx[i]
        op[v] = up[v] & (bad[v] == 0)
    return op


class Episode:
    """One episode with the reference random streams (Sections 2.2, 2.5).

    rng = default_rng(seed): initial down set, then per day u1 = rng.random(n), u2 = rng.random(n).
    Aftershock (D, f): at the start of day D every producer goes down w.p. f, drawn from
    default_rng(seed + 424243) with one random(#producers) draw.
    """

    def __init__(self, net, seed, K, cap, T=T_DEFAULT, rho=RHO, phi=PHI, aftershock=None,
                 compat=None, quorum=None, down0=DOWN0):
        self.net, self.K, self.cap, self.T, self.rho, self.phi = net, K, cap, T, rho, phi
        self.aftershock = aftershock
        self.seed = seed
        self.compat = net.true_compat() if compat is None else compat
        self.quorum = (net.true_quorum() if net.quorum is not None else None) if quorum is None else quorum
        self.rng = np.random.default_rng(seed)
        self.srng = np.random.default_rng(seed + 424243)
        cand = net.producers
        self.up = np.ones(net.n, bool)
        self.up[self.rng.choice(cand, int(down0 * len(cand)), replace=False)] = False
        self.B, self.t, self.total = K, 0, 0.0
        self.shocked = False
        self._shock()

    def _shock(self):
        if self.aftershock and self.t == self.aftershock[0]:
            cand = self.net.producers
            self.up[cand[self.srng.random(len(cand)) < self.aftershock[1]]] = False
            self.shocked = True

    def op(self):
        return operating(self.net, self.up, self.compat, self.quorum)

    def observe(self):
        """(up, op, remaining budget, day)."""
        return self.up.copy(), self.op(), self.B, self.t

    def step(self, repairs) -> float:
        net = self.net
        repairs = [int(v) for v in repairs]
        assert len(repairs) <= min(self.cap, self.B), "daily cap or budget exceeded"
        assert all(net.layer[v] < 3 and not self.up[v] for v in repairs), "repairs must be down producers"
        assert len(set(repairs)) == len(repairs)
        self.B -= len(repairs)
        u1, u2 = self.rng.random(net.n), self.rng.random(net.n)
        for v in repairs:
            if u1[v] < self.rho:
                self.up[v] = True
        fail = (u2 < self.phi) & (net.layer < 3)
        fail[repairs] = False
        self.up[fail] = False
        r = float(net.w[self.op()].sum())
        self.total += r
        self.t += 1
        self._shock()
        return r

    @property
    def done(self):
        return self.t >= self.T
