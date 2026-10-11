# coding=utf-8
"""WS-option (wake-sleep option) for DSR, following the IM / RPP agents of this repository.

Higher layer: Q^I(s, o) with o = [t_r, B_r, b] (action-in), Monte Carlo targets.
Lower layer: Q^II(s, u) for every node u plus a null action (action-out), Double-DQN targets.
Wake-sleep: during the first half of the epochs the higher layer follows the average policy.

Difference from IM: there is no reward decomposition by simulation, because simulating DSR needs
the hidden compatibility relation. Each repair gets reward 0 and the day's reward is paid on the
null action that closes the day; credit reaches the repairs through the TD targets (discount 1
within a day, gamma across days).
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Env import FEATURES
from Network import ActionNetwork, OptionNetwork
from ReplayBuffer import ReplayBuffer, ReplayBuffer_sub, Transition, Transition_sub

CAPACITY = 20000
BATCH_SIZE = 32
# the graph has 300 nodes: per-step CPU inference is faster than many small GPU launches
device = torch.device(os.environ.get("DSR_DEVICE", "cpu"))


class Agent(nn.Module):
    def __init__(self, env, lr_high=1e-3, lr_low=1e-4, discount=0.99):
        super().__init__()
        self.env = env
        self.discount = discount
        self.OptNet = OptionNetwork(FEATURES).to(device)
        self.OptNet_target = OptionNetwork(FEATURES).to(device)
        self.ActNet = ActionNetwork(FEATURES).to(device)
        self.ActNet_target = ActionNetwork(FEATURES).to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.OptNet.parameters(), lr=lr_high)
        self.optimizer_sub = optim.Adam(self.ActNet.parameters(), lr=lr_low)
        self.memory = ReplayBuffer(CAPACITY)
        self.memory_sub = ReplayBuffer_sub(CAPACITY)
        self.learning_step = [0, 0]
        self.target_update = 10
        self.epsilon_decay, self.epsilon_min = 0.98, 0.1
        self.epsilon = [0.9, 0.9]
        self.set_adj()

    def set_adj(self):
        self.adj = torch.tensor(self.env.A, dtype=torch.float32, device=device)

    def set_network(self, index):
        self.env.set_network(index)
        self.set_adj()

    def update_target(self):
        if self.learning_step[0] % self.target_update == 0:
            self.OptNet_target.load_state_dict(self.OptNet.state_dict())
        if self.learning_step[1] % self.target_update == 0:
            self.ActNet_target.load_state_dict(self.ActNet.state_dict())

    # ------------------------------------------------------------------ value functions
    def option_features(self, t_r, B_r, b):
        return [t_r / self.env.T, B_r / self.env.budget, b / self.env.cap]

    def Q(self, states, options, target=False):
        net = self.OptNet_target if target else self.OptNet
        x = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        o = torch.tensor(np.array(options), dtype=torch.float32, device=device)
        return net(x, o, self.adj)

    def Q_sub(self, states, mask=None, target=False):
        net = self.ActNet_target if target else self.ActNet
        x = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        m = None if mask is None else torch.tensor(np.array(mask), dtype=torch.bool, device=device)
        return net(x, self.adj, m)

    def action_mask(self, state, force_node):
        """True = not allowed. The null action (index n) is allowed only when no node must be chosen."""
        valid = np.zeros(self.env.n + 1, bool)
        valid[self.env.repairable(state)] = True
        if not force_node or not valid[:-1].any():
            valid[-1] = True
        return ~valid

    # ------------------------------------------------------------------ policies
    def average_policy(self, t_r, B_r):
        if B_r == 0:
            return 0
        if t_r == 1:
            return min(self.env.cap, B_r)
        return min(self.env.cap, B_r, max(int(self.env.budget / self.env.T), 1))

    def get_option(self, state, t_r, B_r, epsilon=0.0):
        if B_r == 0:
            return 0
        top = min(self.env.cap, B_r)
        if t_r == 1:
            return top
        opts = list(range(0, top + 1))
        if np.random.rand() < epsilon:
            return int(np.random.choice(opts))
        with torch.no_grad():
            q = self.Q([state] * len(opts), [self.option_features(t_r, B_r, b) for b in opts]).cpu().numpy()
        return opts[int(np.argmax(q))]

    def get_action(self, state, budget, epsilon=0.0):
        """Greedy one-at-a-time selection; returns the actions and the intermediate states."""
        state = state.copy()
        actions, states = [], [state.copy()]
        for _ in range(budget):
            cand = self.env.repairable(state)
            if len(cand) == 0:
                break
            if np.random.rand() < epsilon:
                u = int(np.random.choice(cand))
            else:
                with torch.no_grad():
                    q = self.Q_sub([state], [self.action_mask(state, True)]).cpu().numpy()[0]
                u = int(np.argmax(q[:-1]))
            actions.append(u)
            self.env.set_selected(state, [u])
            states.append(state.copy())
        return actions, states

    def run_episode(self, seed, budget_policy="agent", epsilon=(0.0, 0.0)):
        env = self.env
        env.reset(seed)
        S, O, A, R, SS = [], [], [], [], []
        for t in range(env.T):
            s = env.state()
            t_r, B_r = env.T - t, env.budget_r
            b = self.average_policy(t_r, B_r) if budget_policy == "average" else self.get_option(s, t_r, B_r, epsilon[0])
            acts, states = self.get_action(s, b, epsilon[1])
            r, _ = env.step(acts)
            # rewards are learned in units of the total customer weight (a day serving everyone = 1)
            S.append(s); O.append(self.option_features(t_r, B_r, len(acts))); A.append(acts)
            R.append(r / env.net.w.sum()); SS.append(states)
        S.append(env.state())
        return S, O, A, R, SS, env.ep.total

    # ------------------------------------------------------------------ learning
    def update_q_function(self, seed, budget_policy="agent"):
        self.train()
        S, O, A, R, SS, total = self.run_episode(seed, budget_policy, self.epsilon)
        T = len(R)
        G = [0.0] * T
        G[-1] = R[-1]
        for i in range(T - 2, -1, -1):
            G[i] = R[i] + self.discount * G[i + 1]
        for i in range(T):
            self.memory.push(S[i], O[i], G[i], S[i + 1], i == T - 1)
            k = len(A[i])
            next_day_select = 1 if (i + 1 < T and len(A[i + 1]) > 0) else 0
            for j in range(k):
                more = 1 if j + 1 < k else 0                 # within the day: must pick another node
                self.memory_sub.push(SS[i][j], np.array([1]), np.array(A[i][j]), 0.0, SS[i][j + 1],
                                     np.array([more]), False)
            # closing the day with the null action; the next state is the next day's
            self.memory_sub.push(SS[i][k], np.array([0]), np.array(-1), R[i], S[i + 1],
                                 np.array([next_day_select]), i == T - 1)
            loss, loss_sub = self.learn()
        if budget_policy == "agent":
            self.epsilon[0] = max(self.epsilon[0] * self.epsilon_decay, self.epsilon_min)
        return loss, loss_sub, total

    def learn(self):
        batch = self.memory.sample(min(BATCH_SIZE, len(self.memory)))
        b = Transition(*zip(*batch))
        pred = self.Q(b.state, b.option)
        loss = self.loss_fn(pred, torch.tensor(b.gain, dtype=torch.float32, device=device))
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.learning_step[0] += 1

        batch = self.memory_sub.sample(min(BATCH_SIZE, len(self.memory_sub)))
        b = Transition_sub(*zip(*batch))
        n = self.env.n
        act = torch.tensor([int(a) if int(a) >= 0 else n for a in b.action], device=device).view(-1, 1)
        q = self.Q_sub(b.state).gather(1, act).squeeze(1)
        only_null = np.ones(n + 1, bool)
        only_null[-1] = False
        # next_option 1: the next decision picks a node; 0: the next decision is the null action
        masks = [self.action_mask(s, True) if int(o[0]) == 1 else only_null for s, o in zip(b.next_state, b.next_option)]
        with torch.no_grad():
            a_star = self.Q_sub(b.next_state, masks).argmax(1, keepdim=True)        # Double DQN
            q_next = self.Q_sub(b.next_state, masks, target=True).gather(1, a_star).squeeze(1)
            done = torch.tensor(b.done, dtype=torch.bool, device=device)
            q_next[done] = 0.0
            within = torch.tensor([int(x[0]) == 1 for x in b.option], dtype=torch.bool, device=device)
            disc = torch.where(within, torch.ones_like(q_next), torch.full_like(q_next, self.discount))
            target = torch.tensor(b.reward, dtype=torch.float32, device=device) + disc * q_next
        loss_sub = self.loss_fn(q, target)
        self.optimizer_sub.zero_grad(); loss_sub.backward(); self.optimizer_sub.step()
        self.learning_step[1] += 1
        self.update_target()
        self.epsilon[1] = max(self.epsilon[1] * self.epsilon_decay, self.epsilon_min)
        return loss.item(), loss_sub.item()

    def train_model(self, train_nets, num_epochs=20, num_iterations=10, rng=None, log=print):
        rng = rng or np.random.default_rng(0)
        history = []
        for epoch in range(num_epochs):
            policy = "average" if epoch < np.floor(num_epochs / 2) else "agent"   # sleep, then wake
            self.set_network(int(rng.choice(train_nets)))
            losses, rets = [], []
            for it in range(num_iterations):
                l, ls, r = self.update_q_function(int(rng.integers(1 << 30)), policy)
                losses.append((l, ls)); rets.append(r)
            self.eval()
            test = self.run_episode(int(rng.integers(1 << 30)))[-1]
            history.append((np.mean(rets), test))
            log(f"Epoch {epoch} ({policy}), loss {np.mean([x[0] for x in losses]):.2f}, "
                f"loss_sub {np.mean([x[1] for x in losses]):.3f}, train return {np.mean(rets):.1f}, test return {test:.1f}")
        return history
