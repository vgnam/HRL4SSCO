"""Training-free adaptive greedy on the exact test graphs of `IM/Main.py 0 --train-dataset D --test-dataset D`
(test graph sampled with seed 1, independent of training), 100 rounds, gamma = 0.99. Usage: same_graph.py D [D ...]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'IM'))
import numpy as np
from kt_common import graph_arrays
from utils.datasets import dataset_sampler
from Env import Env
from ws_compare_lib import greedy_episode, SCHEDULES

for name in sys.argv[1:]:
    g = dataset_sampler(name, 200, method='rw', seed=1)()
    env = Env(graph=g, budget=20, T=10); _, P = graph_arrays(g)
    for sched in ['average', 'quarter']:
        rng = np.random.default_rng(0)
        vals = [greedy_episode(env, P, SCHEDULES[sched], rng) for _ in range(100)]
        print(f"{name} {sched}-greedyMC: {np.mean(vals):.2f} +- {np.std(vals) / 10:.2f}", flush=True)
