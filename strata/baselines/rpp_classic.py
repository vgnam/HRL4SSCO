"""The RPP baselines of the WS-Option paper (``RPP/Baselines.py``) on the STRATA env.

* :class:`GreedyPolicy` -- every day, from the current position, repeatedly
  add the most profitable unvisited POI that keeps the day's route within the
  daily distance limit, until the budget runs out (closed loop: it reads the
  current profits each day).
* :class:`GAPolicy` -- the paper's genetic algorithm: an individual is a
  ``T x n`` 0/1 plan (POIs of each day, routed in index order), fitness is one
  simulated episode under the instance's true law, roulette selection,
  one-point crossover over days with budget repair, add/remove mutation.  The
  plan is optimised at :meth:`reset` and played open loop; POIs that are no
  longer visitable are skipped.

Both are direct ports of ``RPP/Baselines.py`` (same operators, population 100,
100 generations, mutation rate 0.01).  The only change is that the GA evaluates
a whole population in one vectorised ``M``-path step instead of one rollout at
a time, which gives the same fitness distribution.  Like ``oracle``, the GA
uses the true law as its simulator.
"""

from __future__ import annotations

import copy
import random
from typing import List

import numpy as np

from strata.domains.rpp import env as rpp_env


def _pad(plans: List[List[int]]) -> np.ndarray:
    k = max((len(p) for p in plans), default=0)
    out = -np.ones((len(plans), k), dtype=np.int64)
    for i, p in enumerate(plans):
        out[i, : len(p)] = p
    return out


class GreedyPolicy:
    """``RPP/Baselines.Greedy`` behind the STRATA policy interface."""

    method = "greedy"

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.env = None

    def reset(self, instance, rng) -> None:
        self.env = instance.env

    def act(self, state, t: int) -> np.ndarray:
        env = self.env
        remaining = int(np.asarray(state.budget_left).reshape(-1)[0])
        current = int(state.current[0])
        profit = state.profit[0]
        available = [v for v in range(env.n) if profit[v] > 0]
        plan, total = [], 0.0
        while remaining > 0 and available:
            best, best_profit = None, -np.inf
            for v in available:
                if (total + env.dist[current, v]) / env.max_dist <= env.max_daily_distance:
                    if profit[v] > best_profit:
                        best, best_profit = v, profit[v]
            if best is None:
                break
            plan.append(best)
            total += env.dist[current, best]
            remaining -= 1
            available.remove(best)
            current = best
        return np.asarray(plan, dtype=np.int64)

    def observe(self, state_before, action, state_after, info) -> None:
        return None


class GAPolicy:
    """``RPP/Baselines.GA`` behind the STRATA policy interface (open-loop plan)."""

    method = "ga"

    def __init__(self, population_size=100, num_generations=100, mutation_rate=0.01, seed=0):
        self.population_size = int(population_size)
        self.num_generations = int(num_generations)
        self.mutation_rate = float(mutation_rate)
        self.seed = int(seed)
        self.plan = None

    # ----------------------------------------------------------- GA operators
    def _decode(self, day):
        return [i for i in range(len(day)) if day[i] == 1]

    def _count(self, individual):
        return sum(sum(day) for day in individual)

    def _init_population(self, env, state0):
        population = []
        for _ in range(self.population_size):
            individual, chosen, remaining = [], set(), env.budget
            for _ in range(env.T):
                available = list(set(rpp_env.not_visited(state0).tolist()) - chosen)
                k = self.rand.randint(0, min(len(available), remaining))
                picked = self.rand.sample(available, k)
                chosen.update(picked)
                remaining -= len(picked)
                day = [0] * env.n
                for v in picked:
                    day[v] = 1
                individual.append(day)
            population.append(individual)
        return population

    def _fitness(self, env, law, population, rng):
        """One simulated episode per individual, all individuals in one batch."""
        P = len(population)
        valid = np.asarray([self._count(ind) <= env.budget for ind in population])
        state = env.reset(M=P)
        total = np.zeros(P, dtype=np.float64)
        for d in range(env.T):
            plans = [self._decode(ind[d]) if ok else [] for ind, ok in zip(population, valid)]
            _, reward, done = rpp_env.step(env, state, _pad(plans), law=law, rng=rng)
            total += reward
            if done:
                break
        return np.where(valid, total, -np.inf).tolist()

    def _select(self, population, fitness):
        finite = [f for f in fitness if f != -np.inf]
        low = min(finite) if finite else 0.0
        shifted = [f - low + 1e-6 if f != -np.inf else 0.0 for f in fitness]
        pick = self.rand.uniform(0, sum(shifted))
        current = 0.0
        for individual, score in zip(population, shifted):
            current += score
            if current > pick:
                return individual
        return population[-1]

    def _crossover(self, env, p1, p2):
        p1, p2 = copy.deepcopy(p1), copy.deepcopy(p2)
        cut = self.rand.randint(1, len(p1) - 1)
        c1, c2 = p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]

        def repair(child):
            total = self._count(child)
            if total > env.budget:
                for day in child:
                    for i in range(len(day)):
                        if total <= env.budget:
                            break
                        if day[i] == 1:
                            day[i] = 0
                            total -= 1
            return child

        return repair(c1), repair(c2)

    def _mutate(self, env, individual):
        total = self._count(individual)
        for d in range(len(individual)):
            if self.rand.random() < self.mutation_rate:
                if total < env.budget:
                    used = set(np.where(np.array(individual).sum(axis=0))[0])
                    free = list(set(range(env.n)) - used)
                    if free:
                        individual[d][self.rand.choice(free)] = 1
                        total += 1
                elif total > 0:
                    on = [i for i, x in enumerate(individual[d]) if x == 1]
                    if on:
                        individual[d][self.rand.choice(on)] = 0
                        total -= 1
        return individual

    # -------------------------------------------------------------- policy API
    def reset(self, instance, rng) -> None:
        env, law = instance.env, instance.law
        seed = int(rng.integers(2**31)) if rng is not None else self.seed
        self.rand = random.Random(seed)
        sim_rng = np.random.default_rng(seed)
        population = self._init_population(env, env.reset(M=1))
        best, best_fitness = None, -np.inf
        for _ in range(self.num_generations):
            fitness = self._fitness(env, law, population, sim_rng)
            new_population = []
            for _ in range(self.population_size // 2):
                c1, c2 = self._crossover(env, self._select(population, fitness),
                                         self._select(population, fitness))
                new_population.extend([self._mutate(env, c1), self._mutate(env, c2)])
            i = int(np.argmax(fitness))
            if fitness[i] > best_fitness:
                best_fitness, best = fitness[i], copy.deepcopy(population[i])
            population = new_population
        self.plan = [self._decode(day) for day in best]

    def act(self, state, t: int) -> np.ndarray:
        if self.plan is None:
            raise RuntimeError("reset(instance, rng) must be called before act")
        day = int(state.t) - 1
        if day >= len(self.plan):
            return np.zeros(0, dtype=np.int64)
        budget = int(np.asarray(state.budget_left).reshape(-1)[0])
        plan = [v for v in self.plan[day] if state.profit[0, v] > 0][:budget]
        return np.asarray(plan, dtype=np.int64)

    def observe(self, state_before, action, state_after, info) -> None:
        return None
