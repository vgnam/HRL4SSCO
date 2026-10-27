"""Flat DQN baseline (model-free, no hierarchy) for IM and RPP.

The episode is unrolled into micro-steps, as in S2V-DQN (Khalil et al., NeurIPS
2017): at each micro-step a single Q-network picks either one node to add to
the current round's set (IM: a seed; RPP: the next POI of the day's route) or
the null action, which ends the round and steps the environment with the set
built so far.  Budget allocation and node selection are therefore one flat
decision -- the counterpart of WS-Option's two layers (option network for the
round budget, action network for the nodes).

The network is a small graph network (two message-passing steps) over per-node
features plus global features; Q(null) reads the pooled embedding.  It is
trained with DQN (replay buffer, target network, epsilon-greedy) on the true
environment of the instance, like the WS-Option baseline, for ``episodes``
training episodes (default 300 = WS-Option's 20 epochs x 15 iterations), on CPU.
"""

from __future__ import annotations

import collections
import random
from typing import List, Optional

import numpy as np

from strata.core.types import STATUS_ACTIVE, STATUS_INACTIVE, STATUS_REMOVED
from strata.domains.im import env as im_env
from strata.domains.rpp import env as rpp_env


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the installation
        raise RuntimeError("flat DQN needs PyTorch: %s" % (exc,))
    return torch


# =========================================================================
# Domain views: features, valid actions and the round transition
# =========================================================================


class _IMView:
    """Micro-step view of an IM instance."""

    n_feat, n_glob = 9, 3

    def __init__(self, instance):
        g = instance.graph
        self.instance, self.graph, self.law = instance, g, instance.law
        self.n = int(g.adj.shape[0])
        self.T, self.B = int(instance.T), int(instance.B)
        A = np.asarray(g.adj.todense(), dtype=np.float32)
        self.out_deg = A.sum(axis=1)
        self.in_deg = np.asarray(g.indeg, dtype=np.float32)
        dmax = max(float(self.out_deg.max()), float(self.in_deg.max()), 1.0)
        self.dmax = dmax
        # mean aggregation over in- and out-neighbours
        self.agg = [(A / np.maximum(A.sum(axis=1, keepdims=True), 1.0)),
                    (A.T / np.maximum(A.T.sum(axis=1, keepdims=True), 1.0))]

    def reset(self):
        return im_env.make_state(self.graph, M=1, budget_left=self.B)

    def budget(self, state):
        return int(state.budget_left)

    def done(self, state):
        return int(state.t) >= self.T

    def valid(self, state, chosen):
        ok = np.asarray(state.status)[0] == STATUS_INACTIVE
        ok[list(chosen)] = False
        return ok

    def features(self, state, chosen):
        status = np.asarray(state.status)[0]
        ch = np.zeros(self.n, dtype=np.float32)
        ch[list(chosen)] = 1.0
        act = ((status == STATUS_ACTIVE) | (ch > 0)).astype(np.float32)
        a = np.asarray(act @ self.graph.adj).ravel()
        inactive = (status == STATUS_INACTIVE).astype(np.float32)
        resid = np.asarray(self.graph.adj @ inactive).ravel()
        x = np.stack([
            inactive, (status == STATUS_ACTIVE).astype(np.float32),
            (status == STATUS_REMOVED).astype(np.float32), ch,
            self.in_deg / self.dmax, self.out_deg / self.dmax,
            a / np.maximum(self.in_deg, 1.0), np.asarray(state.E)[0].astype(np.float32) / self.dmax,
            resid / self.dmax], axis=1).astype(np.float32)
        glob = np.asarray([int(state.t) / self.T, self.budget(state) / self.B,
                           len(chosen) / self.B], dtype=np.float32)
        return x, glob

    def step(self, state, chosen, rng):
        _, r, _ = im_env.step(self.graph, self.law, state, np.asarray(chosen, dtype=np.int64), rng)
        return float(np.asarray(r).reshape(-1)[0])


class _RPPView:
    """Micro-step view of an RPP instance (the chosen list is the day's route)."""

    n_feat, n_glob = 8, 4

    def __init__(self, instance, knn=8):
        env = instance.env
        self.instance, self.env, self.law = instance, env, instance.law
        self.n = int(env.n)
        self.T, self.B = int(instance.T), int(instance.B)
        loc = np.asarray(env.loc, dtype=np.float64)
        span = np.maximum(loc.max(axis=0) - loc.min(axis=0), 1e-9)
        self.loc = ((loc - loc.min(axis=0)) / span).astype(np.float32)
        D = env.dist
        A = np.zeros((self.n, self.n), dtype=np.float32)
        nn = np.argsort(D, axis=1)[:, 1:knn + 1]
        A[np.arange(self.n)[:, None], nn] = 1.0
        self.agg = [A / np.maximum(A.sum(axis=1, keepdims=True), 1.0),
                    A.T / np.maximum(A.T.sum(axis=1, keepdims=True), 1.0)]

    def reset(self):
        return self.env.reset(M=1)

    def budget(self, state):
        return int(np.asarray(state.budget_left).reshape(-1)[0])

    def done(self, state):
        return int(state.t) > self.T

    def valid(self, state, chosen):
        ok = np.asarray(state.profit)[0] > 0
        ok[list(chosen)] = False
        return ok

    def _route_length(self, state, chosen):
        cur, total = int(state.current[0]), 0.0
        for v in chosen:
            total += self.env.dist[cur, v]
            cur = v
        return total / self.env.max_dist, cur

    def features(self, state, chosen):
        env = self.env
        length, cur = self._route_length(state, chosen)
        ch = np.zeros(self.n, dtype=np.float32)
        ch[list(chosen)] = 1.0
        x = np.stack([
            self.loc[:, 0], self.loc[:, 1], np.asarray(state.profit)[0].astype(np.float32),
            np.asarray(state.visited)[0].astype(np.float32), ch,
            (env.dist[cur] / env.max_dist).astype(np.float32),
            (env.dist[env.start_node] / env.max_dist).astype(np.float32),
            env.initial_profit.astype(np.float32)], axis=1).astype(np.float32)
        glob = np.asarray([int(state.t) / self.T, self.budget(state) / self.B,
                           len(chosen) / self.B, length / env.max_daily_distance],
                          dtype=np.float32)
        return x, glob

    def step(self, state, chosen, rng):
        _, r, _, _ = rpp_env.step(self.env, state, np.asarray(chosen, dtype=np.int64),
                                  law=self.law, rng=rng, return_info=True)
        return float(np.asarray(r).reshape(-1)[0])

    def route_value(self, state, chosen):
        """Today's reward of ``chosen`` (deterministic: profits drift after the visit)."""
        if not chosen:
            return 0.0
        env = self.env
        A = np.asarray(chosen, dtype=np.int64)[None, :]
        base = rpp_env.profit_reward(env, state.profit, A)
        dist = rpp_env.total_distance(env, state, A)
        return float(rpp_env.distance_penalty_reward(env, base, dist)[0])


# =========================================================================
# Network
# =========================================================================


def _make_net(torch, n_feat, n_glob, hidden=64, rounds=2):
    nn = torch.nn

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_feat, hidden)
            self.mp = nn.ModuleList([nn.Linear(3 * hidden, hidden) for _ in range(rounds)])
            self.node = nn.Sequential(nn.Linear(2 * hidden + n_glob, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 1))
            self.null = nn.Sequential(nn.Linear(hidden + n_glob, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 1))

        def forward(self, x, glob, agg_out, agg_in):
            # x (B, n, F), glob (B, G); returns (B, n + 1) with the null action last
            h = torch.relu(self.inp(x))
            for layer in self.mp:
                h = torch.relu(layer(torch.cat([h, agg_out @ h, agg_in @ h], dim=-1)))
            pooled = h.mean(dim=1)
            g = glob[:, None, :].expand(-1, h.shape[1], -1)
            q_nodes = self.node(torch.cat([h, pooled[:, None, :].expand_as(h), g], dim=-1))[..., 0]
            q_null = self.null(torch.cat([pooled, glob], dim=-1))
            return torch.cat([q_nodes, q_null], dim=1)

    return QNet()


# =========================================================================
# Policy
# =========================================================================


class FlatDQNPolicy:
    """Flat DQN behind the STRATA policy interface (trained once per instance)."""

    method = "dqn"

    def __init__(self, episodes=300, gamma=0.99, lr=1e-3, batch=32, buffer=10000,
                 target_every=200, eps_end=0.05, eps_frac=0.7, hidden=64, threads=2, seed=0):
        self.episodes, self.gamma, self.lr, self.batch = int(episodes), gamma, lr, int(batch)
        self.buffer, self.target_every = int(buffer), int(target_every)
        self.eps_end, self.eps_frac, self.hidden = eps_end, eps_frac, int(hidden)
        self.threads, self.seed = int(threads), int(seed)
        self._key, self.net, self.view = None, None, None
        self.train_log: List[float] = []

    # ------------------------------------------------------------- helpers
    def _q(self, x, glob):
        torch = self.torch
        with torch.no_grad():
            q = self.net(torch.from_numpy(x)[None], torch.from_numpy(glob)[None], *self.aggs)
        return q[0].numpy()

    def _choose(self, state, chosen, eps, rng):
        """Next micro-action: a node index or ``n`` (end of round)."""
        view = self.view
        valid = view.valid(state, chosen)
        if len(chosen) >= view.budget(state) or not valid.any():
            return view.n
        if rng.random() < eps:
            opts = np.concatenate([np.flatnonzero(valid), [view.n]])
            return int(rng.choice(opts))
        x, glob = view.features(state, chosen)
        q = self._q(x, glob)
        q[:-1][~valid] = -np.inf
        return int(np.argmax(q))

    def _round(self, state, eps, rng, record=None):
        """Build one round's set by micro-steps, then step the environment."""
        chosen: List[int] = []
        view = self.view
        while True:
            x, glob = view.features(state, chosen)
            a = self._choose(state, chosen, eps, rng)
            if a == view.n:
                break
            if record is not None:
                record.append((x, glob, a, 0.0, chosen + [a]))
            chosen.append(a)
        return chosen, x, glob

    # -------------------------------------------------------------- training
    def _train(self, instance):
        torch = self.torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        random.seed(self.seed)
        view = self.view
        self.net = _make_net(torch, view.n_feat, view.n_glob, self.hidden)
        target = _make_net(torch, view.n_feat, view.n_glob, self.hidden)
        target.load_state_dict(self.net.state_dict())
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        replay = collections.deque(maxlen=self.buffer)
        n, steps = view.n, 0
        shaped = getattr(view, "route_value", lambda state, chosen: 0.0)
        for ep in range(self.episodes):
            eps = max(self.eps_end, 1.0 - (1.0 - self.eps_end) * ep / max(1, self.eps_frac * self.episodes))
            state, total = view.reset(), 0.0
            pending = None  # transition waiting for its next micro-state
            while not view.done(state):
                chosen: List[int] = []
                while True:
                    x, glob = view.features(state, chosen)
                    can_add = len(chosen) < view.budget(state)
                    mask = np.concatenate([view.valid(state, chosen) & can_add, [True]])
                    if pending is not None:
                        replay.append(pending + (x, glob, mask, False))
                        pending = None
                    a = self._choose(state, chosen, eps, rng)
                    if a == n:
                        before = shaped(state, chosen)
                        r = view.step(state, chosen, rng)
                        total += r
                        pending = (x, glob, n, r - before)
                        break
                    # RPP: the marginal value of the route so far is paid at once (as
                    # S2V-DQN's incremental objective); IM gets it at the end of the round
                    gain = shaped(state, chosen + [a]) - shaped(state, chosen)
                    pending = (x, glob, a, gain)
                    chosen.append(a)
                    steps += 1
                    self._learn(replay, target, opt, steps)
                steps += 1
                self._learn(replay, target, opt, steps)
            x, glob = view.features(state, [])
            replay.append(pending + (x, glob, np.ones(n + 1, bool), True))
            self.train_log.append(total)

    def _learn(self, replay, target, opt, steps):
        torch = self.torch
        if len(replay) < self.batch:
            return
        batch = random.sample(replay, self.batch)
        x = torch.from_numpy(np.stack([b[0] for b in batch]))
        g = torch.from_numpy(np.stack([b[1] for b in batch]))
        a = torch.as_tensor([b[2] for b in batch])
        r = torch.as_tensor([b[3] for b in batch], dtype=torch.float32)
        x2 = torch.from_numpy(np.stack([b[4] for b in batch]))
        g2 = torch.from_numpy(np.stack([b[5] for b in batch]))
        m2 = torch.from_numpy(np.stack([b[6] for b in batch]))
        d = torch.as_tensor([b[7] for b in batch], dtype=torch.float32)
        q = self.net(x, g, *self.aggs).gather(1, a[:, None])[:, 0]
        with torch.no_grad():
            q2 = target(x2, g2, *self.aggs).masked_fill(~m2, -1e9).max(dim=1).values
            y = r + self.gamma * (1.0 - d) * q2
        loss = torch.nn.functional.smooth_l1_loss(q, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        opt.step()
        if steps % self.target_every == 0:
            target.load_state_dict(self.net.state_dict())

    # -------------------------------------------------------------- policy API
    def reset(self, instance, rng) -> None:
        if self._key == instance.name:
            return
        self.torch = _torch()
        self.torch.set_num_threads(self.threads)
        self.view = _IMView(instance) if instance.domain == "im" else _RPPView(instance)
        self.aggs = tuple(self.torch.from_numpy(m) for m in self.view.agg)
        self._train(instance)
        self.net.eval()
        self._key = instance.name

    def act(self, state, t: int) -> np.ndarray:
        chosen, _, _ = self._round(state, 0.0, np.random.default_rng(0))
        return np.asarray(chosen, dtype=np.int64)

    def observe(self, state_before, action, state_after, info) -> None:
        return None
