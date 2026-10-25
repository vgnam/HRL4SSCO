"""Training-free adaptive greedy run inside IM/Env.py, discounted return with gamma=0.99 as
IM/Agent.run_episode. K=20, T=10; a schedule gives the seeds per step (None: spend what is left)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'IM'))
import numpy as np
from kt_common import simple_greedy

K, T, GAMMA, ROUNDS, GRAPHS, R = 20, 10, 0.99, 30, 5, 30
SCHEDULES = {'average': [2] * 9 + [None], 'static': [10, 0, 0, 0, 0, 10, 0, 0, 0, None],
             'quarter': [5, 2, 2, 2, 2, 2, 2, 1, 1, None]}  # None: spend what is left

def greedy_episode(env, P, sched, rng):
    env.reset(); total, left = 0.0, K
    for t in range(T):
        k = left if sched[t] is None else min(sched[t], left); left -= k
        status = env.state.argmax(1).astype(np.int8)
        seeds = simple_greedy(P, status, k, T - t, R, rng)
        nxt, r, _, _ = env.step(env.state, seeds); env.state = nxt.copy(); total += r * GAMMA ** t
    return total

