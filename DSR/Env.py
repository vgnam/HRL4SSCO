# coding=utf-8
"""DSR (dependency-constrained supply-chain recovery) as an HRL4SSCO environment.

The dynamics, random streams and networks are those of lexlaw.envs.dsr (method.md, Section 2):
variants C (plain), Q (hidden slot quorum) and A (aftershock). The agent sees the graph (links),
the customer weights, and every day which nodes are up and which operate; it does not see which
links are compatible (or the quorums): WS-option has to learn that implicitly.

State (n x F): [up, op, selected today, producer, layer one-hot (4), weight, remaining days / T].
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lexlaw.envs.dsr import Episode, make_network  # noqa: E402

FEATURES = 10


class Env:
    def __init__(self, index=0, variant="C", budget=20, cap=4, T=15, phi=0.01, aftershock=None, p_dual=0.3):
        self.variant, self.cap, self.T, self.phi, self.p_dual = variant, cap, T, phi, p_dual
        self.budget = budget
        self.aftershock = aftershock if variant == "A" else None
        self.set_network(index)

    def set_network(self, index):
        self.net = make_network(index, "Q" if self.variant == "Q" else "C", self.p_dual)
        net = self.net
        self.n = net.n
        # symmetric adjacency over supplier-consumer links, row-normalised (self loops included)
        A = np.eye(self.n)
        for v, sl in net.slots.items():
            for links, _ in sl:
                A[v, links] = 1
                A[links, v] = 1
        self.A = A / A.sum(1, keepdims=True)
        self.static = np.zeros((self.n, FEATURES))
        self.static[:, 3] = net.layer < 3
        self.static[np.arange(self.n), 4 + net.layer] = 1
        self.static[:, 8] = net.w / 3.0
        self.producers = net.producers

    def get_info(self):
        return self.n, self.budget, self.T

    def reset(self, seed):
        self.ep = Episode(self.net, seed, self.budget, self.cap, T=self.T, phi=self.phi, aftershock=self.aftershock)
        self.budget_r = self.budget

    def state(self):
        up, op, B, t = self.ep.observe()
        s = self.static.copy()
        s[:, 0] = up
        s[:, 1] = op
        s[:, 9] = (self.T - t) / self.T
        return s

    @staticmethod
    def set_selected(state, nodes):
        state[nodes, 2] = 1

    def repairable(self, state):
        """Down producers not yet selected today."""
        return np.where((state[:, 3] == 1) & (state[:, 0] == 0) & (state[:, 2] == 0))[0]

    def step(self, action):
        r = self.ep.step(action)
        self.budget_r = self.ep.B
        return r, self.ep.done
