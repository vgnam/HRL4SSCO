"""Kill test 2 (RPP): value of adaptivity. Same simulated-annealing planner, run once at day 0 (open loop)
or every day on the observed profits (closed loop, execute the first day). Evaluated in RPP/Env.py."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, time, math, random, numpy as np
sys.path.insert(0, os.path.join(HERE, '..', '..', 'RPP'))
from Env import Env
from real_data import instance_sampler

K, T, E, ITERS = 20, 10, 30, 3000

def plan_value(env, plan, profit, cur, t0):
    val, maxd, pos = 0.0, np.max(env.dist), cur
    for d, day in enumerate(plan):
        if not day: continue
        dist = env.dist[pos, day[0]] + sum(env.dist[day[i], day[i + 1]] for i in range(len(day) - 1))
        if t0 + d == env.T: dist += env.dist[day[-1], env.start_node]
        dist /= maxd
        val += sum(profit[v] for v in day) - env.eta * (len(day) - 1) ** 2
        if dist > env.max_daily_distance: val -= env.distance_penalty * (dist - env.max_daily_distance)
        pos = day[-1]
    return val

def plan(env, rng, informed=True):
    """Plan the remaining days on the current profits (no forecast of the zero-mean drift).
    informed=False plans on the day-0 profits of the unvisited nodes (replanning without new information)."""
    D, B, cur, t0 = env.T - env.t + 1, env.remaining_budget, env.current_node, env.t
    profit = env.profit.copy() if informed else np.where(env.profit > 0, env.initial_profit, 0); cand = [v for v in np.argsort(-profit) if profit[v] > 0]
    days = [[] for _ in range(D)]
    for i, v in enumerate(cand[:B]): days[i % D].append(int(v))  # round-robin init
    used = set(v for d in days for v in d); best = [d[:] for d in days]
    val = bval = plan_value(env, days, profit, cur, t0)
    for it in range(ITERS):
        temp = 0.2 * (1 - it / ITERS) + 1e-3
        new = [d[:] for d in days]; m = rng.random(); planned = [v for d in new for v in d]
        free = [v for v in cand if v not in used]
        if m < 0.3 and free and len(planned) < B:
            d = rng.randrange(D); new[d].insert(rng.randint(0, len(new[d])), rng.choice(free))
        elif m < 0.45 and planned:
            v = rng.choice(planned); [d.remove(v) for d in new if v in d]
        elif m < 0.8 and planned:
            v = rng.choice(planned); [d.remove(v) for d in new if v in d]
            d = rng.randrange(D); new[d].insert(rng.randint(0, len(new[d])), v)
        elif planned and free:
            v, w = rng.choice(planned), rng.choice(free)
            for d in new:
                if v in d: d[d.index(v)] = w
        else:
            continue
        nv = plan_value(env, new, profit, cur, t0)
        if nv >= val or rng.random() < math.exp((nv - val) / temp):
            days, val = new, nv; used = set(v for d in days for v in d)
            if val > bval: best, bval = [d[:] for d in days], val
    return best

def episode(inst, drift, mode, seed):
    loc, p0, _ = inst
    env = Env(loc=loc, initial_profit=p0, budget=K, T=T, profit_factor=drift)
    rng = random.Random(seed); np.random.seed(seed); total = 0.0
    fixed = plan(env, rng)
    for d in range(T):
        day = fixed[d] if mode == 'open' or d == 0 else plan(env, rng, informed=(mode == 'closed'))[0]
        day = [v for v in day if env.profit[v] > 0][:env.remaining_budget]
        _, r, _ = env.step(day); total += r
    return total

