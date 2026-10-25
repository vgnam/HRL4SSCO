"""Training-free adaptive greedy vs the WS-option heuristics, inside IM/Env.py, on the paper's test setting
(ER n=200 p=0.01 directed, K=20, T=10), discounted return with gamma=0.99 as IM/Agent.run_episode."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys, io, contextlib, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'IM'))
import numpy as np, networkx as nx, random
from kt_common import graph_arrays
from Env import Env
from Agent import Agent

from ws_compare_lib import K, T, GAMMA, ROUNDS, GRAPHS, R, SCHEDULES, greedy_episode

res = {}
for gi in range(GRAPHS):
    random.seed(gi); np.random.seed(gi)
    g = nx.erdos_renyi_graph(200, 0.01, seed=gi, directed=True)
    env = Env(graph=g, budget=K, T=T); _, P = graph_arrays(g)
    with contextlib.redirect_stdout(io.StringIO()):
        model = Agent(env=env); model.eval()
    for bp, sp in [('average', 'score'), ('static', 'score'), ('normal', 'score'), ('static', 'degree')]:
        vals = [model.run_episode(budget_policy=bp, seeding_policy=sp)[-1] for _ in range(ROUNDS)]
        res.setdefault(f'{bp}-{sp}', []).append(np.mean(vals))
    rng = np.random.default_rng(gi)
    for name, sched in SCHEDULES.items():
        vals = [greedy_episode(env, P, sched, rng) for _ in range(ROUNDS)]
        res.setdefault(f'{name}-greedyMC', []).append(np.mean(vals))
    print(f"graph {gi}: " + '  '.join(f"{k}={v[-1]:.1f}" for k, v in res.items()), flush=True)

ref = np.array(res['static-score'])
print("mean over graphs (vs static-score): " + '  '.join(f"{k}={np.mean(v):.1f} ({100*np.mean(np.array(v)/ref-1):+.1f}%)" for k, v in res.items()))
print("reference on the paper-reproduction test graph (results/, seed 0): WS-option agent-agent 108.86, static-score 119.99, average-score 117.26, normal-score 114.60")
