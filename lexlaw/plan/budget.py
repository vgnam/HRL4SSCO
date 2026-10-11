"""High level: budget schedule chosen by rollouts of the model (method Section 5.2) [specified, untested].

Schedule family: asap, and (r, mode) for r in {floor(0.15K), floor(0.3K), floor(0.5K), floor(0.75K)},
mode in {hold, uniform}. On day 0, every third day, and on the day an aftershock occurs or is
detected, every schedule is evaluated by n_sim simulated continuations in the model (certainty
equivalence) with common random numbers, and the best one is followed.
"""
import math

import numpy as np

from ..envs.dsr import operating
from .restore import restore_greedy


def family(K):
    fam = [("asap", None)]
    for r in sorted({int(math.floor(x * K)) for x in (0.15, 0.3, 0.5, 0.75)}):
        fam += [(r, "hold"), (r, "uniform")]                 # ties: asap, then smaller r, then hold
    return fam


def room_of(theta, t, B, cap, T, released):
    if theta[0] == "asap":
        return min(cap, B)
    r, mode = theta
    if not released and B > r:
        return min(cap, B - r)
    if mode == "uniform":
        return min(cap, B, int(math.ceil(B / (T - t))))
    return 0 if not released else min(cap, B)               # hold


class BudgetPlanner:
    """variant: "C" (no aftershock), "known" (D, f given), "random" (D ~ U{4..10} unknown, f given)."""

    def __init__(self, net, K, cap, T=15, variant="C", D=None, f=0.0, n_sim=8, seed=0, rho=0.9, phi=None,
                 fixed=None, days=(4, 10)):
        self.net, self.K, self.cap, self.T = net, K, cap, T
        self.variant, self.D, self.f, self.n_sim, self.seed, self.rho = variant, D, f, n_sim, seed, rho
        self.phi_known, self.fixed, self.days = phi, fixed, days
        self.fam = family(K)
        self.choices = []

    def reset(self, ep):
        self.theta, self.prev_up, self.detected, self.ep_seed = None, None, None, ep.seed
        self.event_today = False
        self.choices = []

    # -- aftershock bookkeeping
    def occurred(self, t):
        if self.variant == "known":
            return t >= self.D
        if self.variant == "random":
            return self.detected is not None and t >= self.detected
        return False

    def observe(self, t, up, rates):
        """Call once per day before rates.update; returns True when day t is an aftershock day
        (known D) or the day one is detected."""
        event = False
        if self.variant == "known":
            event = t == self.D
        elif self.variant == "random" and self.detected is None and self.prev_up is not None:
            P = self.net.producers
            was = self.prev_up[P]
            N = int(was.sum())
            drop = int((was & ~up[P]).sum())
            ph = self.phi_known if self.phi_known is not None else rates.phi
            if drop > ph * N + 4 * math.sqrt(ph * N) + 3:
                self.detected, event = t, True
        self.prev_up = up.copy()
        self.event_today = event
        return event

    def released(self, t):
        return self.occurred(t) or t >= self.T - 3

    # -- rollouts
    def simulate(self, theta, t0, up0, B0, compat, quorum, phi, rngs_days, shocks):
        total = 0.0
        up, B = up0.copy(), B0
        net = self.net
        for k, t in enumerate(range(t0, self.T)):
            if shocks is not None and t in shocks and t > t0:
                up[net.producers[shocks[t]]] = False
            rel = self.released(t) or (shocks is not None and any(d <= t for d in shocks))
            room = room_of(theta, t, B, self.cap, self.T, rel)
            rep = restore_greedy(net, up, compat, room, quorum) if room > 0 else []
            B -= len(rep)
            u1, u2 = rngs_days[k]
            for v in rep:
                if u1[v] < self.rho:
                    up[v] = True
            fail = (u2 < phi) & (net.layer < 3)
            fail[rep] = False
            up[fail] = False
            total += float(net.w[operating(net, up, compat, quorum)].sum())
        return total

    def choose(self, t, up, B, compat, quorum, rates):
        phi = self.phi_known if self.phi_known is not None else rates.phi
        net = self.net
        vals = np.zeros(len(self.fam))
        for s in range(self.n_sim):
            rng = np.random.default_rng([self.seed, self.ep_seed, t, s])
            shocks = None
            if self.variant == "known" and t < self.D:
                d = self.D
            elif self.variant == "random" and self.detected is None:
                cand = [d for d in range(self.days[0], self.days[1] + 1) if d > t]
                d = int(rng.choice(cand)) if cand else None
            else:
                d = None
            if d is not None:
                shocks = {d: rng.random(len(net.producers)) < self.f}
            days = [(rng.random(net.n), rng.random(net.n)) for _ in range(t, self.T)]
            for i, theta in enumerate(self.fam):
                vals[i] += self.simulate(theta, t, up, B, compat, quorum, phi, days, shocks)
        best = int(np.argmax(vals))                                # first maximiser: the tie order
        self.choices.append((t, self.fam[best]))
        return self.fam[best]

    def room(self, t, B, up, op, compat, quorum, rates):
        if self.fixed is not None:
            self.theta = self.fixed
        elif self.theta is None or t % 3 == 0 or self.event_today:
            self.theta = self.choose(t, up, B, compat, quorum, rates)
        return room_of(self.theta, t, B, self.cap, self.T, self.released(t))
