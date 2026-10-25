"""Adapter for the original WS-Option agents (``IM/Agent.py``, ``RPP/Agent.py``).

The paper baselines are two-layer option networks implemented in PyTorch.  They
are *not* modified here: the module imports the original files with their own
directory on ``sys.path`` and supplies an environment adapter that exposes the
small API the agents expect (``Env`` in ``IM/`` and ``RPP/``) on top of the
STRATA environments.  This makes it possible to

* train on the true environment of an instance (the upper-bound variant), or
* train on a simulator built from a fitted law by passing ``law=...``.

The paper hyper-parameters are the defaults below (``IM/param_sets.py``:
20 epochs, 15 iterations; ``RPP/Main.py``: 20 epochs, 10 iterations); the
training budget can be overridden through ``epochs``/``iterations``.

PyTorch is optional.  ``available()`` reports whether the agents can be
imported; :class:`WSOptionUnavailable` is raised otherwise, so the benchmark
driver can skip this baseline and still run the rest.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Optional

import numpy as np

import networkx as nx

from strata.core.types import STATUS_ACTIVE, STATUS_INACTIVE, STATUS_REMOVED, Graph
from strata.domains.im import env as im_env
from strata.domains.im import graph as im_graph
from strata.domains.rpp import env as rpp_env

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IM_DIR = os.path.join(REPO_ROOT, "IM")
RPP_DIR = os.path.join(REPO_ROOT, "RPP")

# Paper defaults (IM/param_sets.py, RPP/Main.py).
IM_WS_DEFAULTS = {"epochs": 20, "iterations": 15}
RPP_WS_DEFAULTS = {"epochs": 20, "iterations": 10}

_AGENT_MODULES = ("Env", "Network", "ReplayBuffer", "Agent")


class WSOptionUnavailable(RuntimeError):
    """Raised when PyTorch (or the original agent files) cannot be imported."""


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _ensure_tensorboard_stub():
    """Give ``Agent.py`` a no-op ``SummaryWriter`` when TensorBoard is absent."""
    try:
        import torch.utils.tensorboard  # noqa: F401

        return
    except Exception:
        pass
    module = types.ModuleType("torch.utils.tensorboard")

    class SummaryWriter:  # minimal API used by train_model
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

    module.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = module
    try:
        import torch.utils

        torch.utils.tensorboard = module
    except Exception:
        pass


def _import_agent(domain: str):
    """Import the original ``Agent`` module of ``domain`` in isolation.

    ``IM/`` and ``RPP/`` both contain modules called ``Env``, ``Network`` and
    ``ReplayBuffer``, so the temporary modules are removed from ``sys.modules``
    after the import (and any previous ones restored) to avoid cross-talk.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the installation
        raise WSOptionUnavailable("PyTorch is not importable: %s" % (exc,))
    _ensure_tensorboard_stub()

    directory = IM_DIR if str(domain).lower() == "im" else RPP_DIR
    saved = {
        name: sys.modules.pop(name) for name in _AGENT_MODULES if name in sys.modules
    }
    sys.path.insert(0, directory)
    try:
        module = importlib.import_module("Agent")
    except Exception as exc:  # pragma: no cover - depends on the installation
        raise WSOptionUnavailable("cannot import %s/Agent.py: %s" % (directory, exc))
    finally:
        try:
            sys.path.remove(directory)
        except ValueError:
            pass
        for name in _AGENT_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return module, torch


def available() -> bool:
    """Whether the original WS-Option agents can be imported in this process."""
    try:
        _import_agent("im")
    except WSOptionUnavailable:
        return False
    return True


# ---------------------------------------------------------------------------
# IM environment adapter (mimics IM/Env.py)
# ---------------------------------------------------------------------------


def _to_im_state(status_row: np.ndarray) -> np.ndarray:
    """``(n,)`` status -> the ``(n, 3)`` one-hot layout of ``IM/Env.py``."""
    n = int(status_row.size)
    state = np.zeros((n, 3), dtype=np.float64)
    state[:, 0] = status_row == STATUS_INACTIVE
    state[:, 1] = status_row == STATUS_ACTIVE
    state[:, 2] = status_row == STATUS_REMOVED
    return state


class IMEnvAdapter:
    """``IM/Env.py``-compatible view of one STRATA IM instance under a law.

    The API (method names, shapes, return values) mirrors ``IM/Env.py`` exactly;
    the actual propagation is delegated to :mod:`strata.domains.im.env`, which
    samples the seeded law instead of hard-coding the weighted cascade.  Passing
    ``law=WC`` therefore reproduces the reference environment in distribution.
    """

    def __init__(self, graph, law, budget: int = 20, T: int = 10, seed: int = 0):
        # networkx graphs also expose .adj, so test for the STRATA type itself
        if not isinstance(graph, Graph):
            graph = im_graph.build_graph(graph, community=False)
        self.graph_obj = graph
        self.law = law
        self.n = int(graph.adj.shape[0])
        self.nodes = list(range(self.n))
        # IM/Env.py uses a weighted adjacency (edge u -> v has weight
        # 1 / in_degree(v)); the agents only feed it to the Q network.
        dense = np.asarray(graph.adj.todense(), dtype=np.float64)
        coeff = 1.0 / np.maximum(graph.indeg.astype(np.float64), 1.0)
        self.A = dense * coeff[None, :]
        self.weighted = True
        self.T = int(T)
        self.budget = int(budget)
        self.budget_r = int(budget)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._sim = im_env.make_state(graph, M=1, budget_left=0)
        self.state = _to_im_state(self._sim.status[0])
        # Kept for API parity with IM/Env.py (not used by the agent code).
        self.graph = nx.from_scipy_sparse_array(graph.adj, create_using=nx.DiGraph())

    # ------------------------------------------------------------- lifecycle
    def get_info(self):
        return self.n, self.budget, self.T

    def reset(self):
        self._sim = im_env.make_state(self.graph_obj, M=1, budget_left=0)
        self.state = _to_im_state(self._sim.status[0])
        self.budget_r = self.budget

    def step(self, state, action):
        """One round from ``state`` (as ``IM/Env.py`` does) under ``self.law``."""
        seeds = np.asarray(list(action), dtype=np.int64).reshape(-1)
        # ``run_simulation`` passes earlier states, so build a throw-away path
        # from the supplied state rather than from the adapter's own state.
        status = np.empty((1, self.n), dtype=np.int8)
        status[0] = np.where(
            state[:, 0] == 1,
            STATUS_INACTIVE,
            np.where(state[:, 1] == 1, STATUS_ACTIVE, STATUS_REMOVED),
        )
        # The 3-column state has no exposure; carry it over when stepping the
        # adapter's own trajectory so memory laws (rho > 0, e.g. FATIGUE) see
        # their past exposure. Throw-away states start from zero exposure.
        own = np.array_equal(state, self.state)
        sim = im_env.SimState(
            status=status,
            E=self._sim.E.copy() if own else np.zeros((1, self.n), dtype=np.float32),
            t=0,
            budget_left=0,
        )
        _, reward, info = im_env.step(
            self.graph_obj, self.law, sim, seeds, self.rng, return_info=True
        )
        if own:
            self._sim = sim
        self.state = _to_im_state(sim.status[0])
        is_null = bool(len(seeds) + int(np.asarray(info["new"][0]).sum()) == 0)
        done = self.is_stop(self.state)
        return (
            self.state.copy(),
            float(np.asarray(reward).reshape(-1)[0]),
            done,
            is_null,
        )

    # ------------------------------------------------------------ state utils
    def set_state_index(self, state_, nodes, state_index: int):
        # ``IM/Env.py`` accepts a scalar node id as well as a list/array.
        idx = np.asarray(nodes, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return state_
        state_[idx, :] = 0.0
        state_[idx, int(state_index)] = 1.0
        return state_

    def get_nodes_type(self, state, type):
        column = {"inactive": 0, "active": 1, "removed": 2}.get(type)
        if column is None:
            raise ValueError("type must be one of 'inactive', 'active', 'removed'")
        return [int(v) for v in np.flatnonzero(state[:, column] == 1)]

    def get_num_type(self, state, type):
        return len(self.get_nodes_type(state, type))

    def get_inactive_edges(self, state=None, nodes=None):
        if state is None:
            state = self.state
        if nodes is None:
            nodes = self.nodes
        nodes = [int(u) for u in np.asarray(nodes, dtype=np.int64).reshape(-1)]
        node_set = set(nodes)
        adj = self.graph_obj.adj
        indptr, indices = adj.indptr, adj.indices
        edges = []
        for u in nodes:
            for v in indices[indptr[u] : indptr[u + 1]]:
                v = int(v)
                if state[v, 0] == 1 and v not in node_set:
                    edges.append((u, v))
        return edges

    def get_inactive_neighbors(self, state=None, nodes=None):
        return [v for _, v in self.get_inactive_edges(state, nodes)]

    def get_edge_prob(self, u, v):
        # The STRATA graphs keep topology only, so the weighted-cascade
        # probability is the only per-edge weight available (``IM/Env.py``
        # would prefer a stored ``p`` attribute first).
        return 1.0 / max(int(self.graph_obj.indeg[v]), 1)

    def is_stop(self, state):
        return self.get_num_type(state, "active") == 0 and self.budget == 0

    def get_node_value(self, nodes, mode="score"):
        values = {}
        for v in nodes:
            neighbors = self.get_inactive_neighbors(nodes=[v])
            if mode == "score":
                values[v] = sum(self.get_edge_prob(v, u) for u in neighbors)
            else:
                values[v] = float(len(neighbors))
        return values

    def sort_by_value(self, nodes, mode="score"):
        values = self.get_node_value(nodes, mode)
        return [
            v for v, _ in sorted(values.items(), key=lambda kv: kv[1], reverse=True)
        ]


# ---------------------------------------------------------------------------
# RPP environment adapter (mimics RPP/Env.py)
# ---------------------------------------------------------------------------


class RPPEnvAdapter:
    """``RPP/Env.py``-compatible view of one STRATA RPP instance under a law."""

    def __init__(
        self,
        loc,
        initial_profit=None,
        budget: int = 20,
        T: int = 10,
        law=None,
        seed: int = 0,
        profit_factor: float = 0.05,
        max_daily_distance: float = 1.5,
        distance_penalty: float = 5.0,
        start_node: int = 0,
    ):
        self.loc = np.asarray(loc, dtype=np.float64)
        self.initial_profit = (
            np.random.default_rng(seed).uniform(0.5, 1.0, len(self.loc))
            if initial_profit is None
            else np.asarray(initial_profit, dtype=np.float64).copy()
        )
        self.budget = int(budget)
        self.T = int(T)
        self.law = law
        self.seed = int(seed)
        self.profit_factor = float(profit_factor)
        self.max_daily_distance = float(max_daily_distance)
        self.distance_penalty = float(distance_penalty)
        self.start_node = int(start_node)
        self.profit_min = 0.3
        self.eta = 0.1
        self.rng = np.random.default_rng(self.seed)
        self._make_env()
        self.reset()

    def _make_env(self):
        self.n = int(self.loc.shape[0])
        self.dist = np.sqrt(
            ((self.loc[:, None, :] - self.loc[None, :, :]) ** 2).sum(axis=2)
        )
        self._env = rpp_env.RPPEnv(
            loc=self.loc,
            initial_profit=self.initial_profit,
            budget=self.budget,
            T=self.T,
            max_daily_distance=self.max_daily_distance,
            distance_penalty=self.distance_penalty,
            start_node=self.start_node,
            profit_factor=self.profit_factor,
            eta=self.eta,
            seed=self.seed,
            empty_day_advances=True,
        )

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        self._make_env()
        self._state = self._env.reset(M=1)
        self.t = int(self._state.t)
        self.profit = self._state.profit[0].copy()
        self.remaining_budget = int(self._state.budget_left[0])
        self.current_node = int(self._state.current[0])

    def get_state(self):
        loc = self.loc / np.max(self.loc)
        return np.concatenate([loc, self.profit.reshape(-1, 1)], axis=1)

    def get_context(self):
        return np.array([self.current_node, self.start_node])

    def get_not_visited(self):
        return np.flatnonzero(self.profit > 0)

    def cal_total_distance(self, action, context=None):
        if context is None:
            context = self.get_context()
        action = [int(v) for v in action]
        if len(action) == 0:
            if self.t < self.T:
                return 0.0
            # RPP/Env.py returns the raw (unnormalised) distance here.
            return float(self.dist[context[0], context[1]])
        total = self.dist[context[0], action[0]]
        total += sum(
            self.dist[action[i], action[i + 1]] for i in range(len(action) - 1)
        )
        if self.t == self.T:
            total += self.dist[action[-1], context[1]]
        return float(total / np.max(self.dist))

    def cal_reward(self, nodes):
        nodes = [int(v) for v in nodes]
        if not nodes:
            return 0.0
        reward = sum(self.profit[node] for node in nodes)
        reward -= self.eta * (max(len(nodes) - 1, 0) ** 2)
        return float(reward)

    def step(self, action):
        action = np.asarray(list(action), dtype=np.int64)
        self._state, reward, done, _ = rpp_env.step(
            self._env,
            self._state,
            action,
            law=self.law,
            rng=self.rng,
            return_info=True,
        )
        self.t = int(self._state.t)
        self.profit = self._state.profit[0].copy()
        self.remaining_budget = int(self._state.budget_left[0])
        self.current_node = int(self._state.current[0])
        return self.get_state(), float(np.asarray(reward).reshape(-1)[0]), bool(done)


# ---------------------------------------------------------------------------
# Training / running
# ---------------------------------------------------------------------------


def _nx_graph(graph) -> nx.DiGraph:
    out = nx.from_scipy_sparse_array(graph.adj, create_using=nx.DiGraph())
    return out


def _set_seeds(seed: int, torch) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_im(instance, module, torch, epochs, iterations, law, seed):
    graph = instance.graph
    nxg = _nx_graph(graph)
    train_law = instance.law if law is None else law

    def factory(graph, budget=20, T=10, seed=seed, **kwargs):
        g = (
            graph
            if isinstance(graph, Graph)
            else im_graph.build_graph(graph, community=False)
        )
        return IMEnvAdapter(g, train_law, budget=budget, T=T, seed=seed)

    module.Env = factory
    adapter = IMEnvAdapter(graph, train_law, budget=instance.B, T=instance.T, seed=seed)
    agent = module.Agent(env=adapter)
    agent.graph_sampler = lambda: nxg.copy()
    agent.train_model(num_epochs=epochs, num_iterations=iterations, test=False)

    eval_adapter = IMEnvAdapter(
        graph, instance.law, budget=instance.B, T=instance.T, seed=seed + 1
    )
    agent.env = eval_adapter
    agent.A = torch.Tensor(np.asarray(eval_adapter.A)).to(agent.A.device)
    return agent, eval_adapter


def _train_rpp(instance, module, torch, epochs, iterations, law, seed):
    cache = instance.build()
    loc = np.asarray(cache["loc"], dtype=np.float64)
    initial = np.asarray(cache["initial_profit"], dtype=np.float64)
    factor = float(cache["profit_factor"])
    train_law = instance.law if law is None else law

    adapter = RPPEnvAdapter(
        loc,
        initial_profit=initial,
        budget=instance.B,
        T=instance.T,
        law=train_law,
        seed=seed,
        profit_factor=factor,
    )
    agent = module.Agent(env=adapter)
    agent.instance_sampler = lambda: (loc.copy(), initial.copy(), factor)
    agent.train_model(num_epochs=epochs, num_iterations=iterations, test=False)

    eval_adapter = RPPEnvAdapter(
        loc,
        initial_profit=initial,
        budget=instance.B,
        T=instance.T,
        law=instance.law,
        seed=seed + 1,
        profit_factor=factor,
    )
    agent.env = eval_adapter
    return agent, eval_adapter


def run_ws_option(
    instance,
    epochs: Optional[int] = None,
    iterations: Optional[int] = None,
    episodes: int = 1,
    law=None,
    seed: int = 0,
) -> float:
    """Train and run WS-Option on ``instance``; return the mean reward.

    ``law`` selects the training simulator: ``None`` trains on the true law of
    the instance (upper-bound variant), any law object trains on that simulator.
    The agent is always evaluated under the true law of the instance.
    """
    module, torch = _import_agent(instance.domain)
    defaults = IM_WS_DEFAULTS if instance.domain == "im" else RPP_WS_DEFAULTS
    epochs = defaults["epochs"] if epochs is None else int(epochs)
    iterations = defaults["iterations"] if iterations is None else int(iterations)
    _set_seeds(seed, torch)

    if instance.domain == "im":
        agent, _ = _train_im(instance, module, torch, epochs, iterations, law, seed)
        agent.eval()
        values = [
            float(
                agent.run_episode(
                    budget_policy="agent",
                    seeding_policy="agent",
                    beam_search=False,
                    epsilon=0,
                )[-1]
            )
            for _ in range(int(episodes))
        ]
    else:
        agent, _ = _train_rpp(instance, module, torch, epochs, iterations, law, seed)
        agent.eval()
        values = [
            float(
                agent.run_episode(
                    budegt_policy="agent", beam_search=False, epsilon=[0, 0]
                )[-1]
            )
            for _ in range(int(episodes))
        ]
    return float(np.mean(values))


class WSOptionPolicy:
    """WS-Option behind the STRATA policy interface (train once per instance).

    Training happens on the first :meth:`reset`; subsequent resets reuse the
    trained networks.  ``act`` is a greedy option/action choice (no exploration)
    on the state passed by the evaluator, so the policy can be paired with any
    STRATA environment rollout.
    """

    def __init__(
        self,
        epochs: Optional[int] = None,
        iterations: Optional[int] = None,
        law=None,
        seed: int = 0,
    ):
        self.epochs = epochs
        self.iterations = iterations
        self.law = law
        self.seed = int(seed)
        self._cache_key = None
        self.agent = None
        self.module = None
        self.eval_adapter = None

    def reset(self, instance, rng: np.random.Generator) -> None:
        if self._cache_key == instance.name:
            return
        module, torch = _import_agent(instance.domain)
        self.module = module
        defaults = IM_WS_DEFAULTS if instance.domain == "im" else RPP_WS_DEFAULTS
        epochs = defaults["epochs"] if self.epochs is None else int(self.epochs)
        iterations = (
            defaults["iterations"] if self.iterations is None else int(self.iterations)
        )
        _set_seeds(self.seed, torch)
        if instance.domain == "im":
            self.agent, self.eval_adapter = _train_im(
                instance, module, torch, epochs, iterations, self.law, self.seed
            )
        else:
            self.agent, self.eval_adapter = _train_rpp(
                instance, module, torch, epochs, iterations, self.law, self.seed
            )
        self.agent.eval()
        self._cache_key = instance.name

    def act(self, state, t: int) -> np.ndarray:
        if self.agent is None:
            raise RuntimeError("reset(instance, rng) must be called before act")
        if getattr(state, "profit", None) is None:  # IM SimState
            state3 = _to_im_state(state.status[0])
            budget = int(state.budget_left)
            budget_cur = self.agent.get_option(
                state3, self.agent.env.T - int(t), budget, True, epsilon=0
            )
            action = self.agent.get_action(state3, budget_cur, epsilon=0)
            return np.asarray(action, dtype=np.int64)
        # RPP RPPState
        adapter = self.eval_adapter
        loc = adapter.loc / np.max(adapter.loc)
        state_feat = np.concatenate(
            [loc, np.asarray(state.profit[0]).reshape(-1, 1)], axis=1
        )
        context = np.array([int(state.current[0]), adapter.start_node])
        budget = int(np.asarray(state.budget_left).reshape(-1)[0])
        budget_cur = self.agent.get_option(
            state_feat, context, adapter.T - int(t), budget, epsilon=0
        )
        action = self.agent.get_action(state_feat, context, budget_cur, epsilon=0)
        return np.asarray(action, dtype=np.int64)

    def observe(self, state_before, action, state_after, info) -> None:
        return None
