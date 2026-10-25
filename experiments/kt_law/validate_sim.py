"""T3 sanity check: my vectorised simulator must reproduce IM/Env.py under the repo law.

Both are run with the same action sequence on the same graph; the means over many
episodes should agree within Monte-Carlo error.
"""

import random
import sys

import numpy as np

from common import REPO, build_arrays, er_graph, load_subgraph
from laws import get_law
from planner import plan_episode, plan_episode_heuristic
from sim import rollout, value

sys.path.insert(0, REPO + r'\IM')
from Env import Env  # noqa: E402


def strip_weights(G):
    """Remove dataset edge probabilities so Env falls back to the weighted cascade."""
    H = G.copy()
    for u, v in H.edges():
        H[u][v].pop('p', None)
        H[u][v].pop('weight', None)
    return H


def env_episode(G, actions, seed):
    random.seed(seed)
    env = Env(graph=G.copy(), budget=sum(len(a) for a in actions), T=len(actions))
    env.reset()
    total = 0.0
    for a in actions:
        next_state, reward, done, is_null = env.step(env.state, list(a))
        env.state = next_state.copy()
        total += reward
    return total


def compare(name, G, actions, n_episodes=1500, seed=7):
    law = get_law('WC')
    adj, indeg, _ = build_arrays(G)
    env_vals = np.array([env_episode(G, actions, seed * 100000 + i) for i in range(n_episodes)])
    R = rollout(law, adj, indeg, [np.asarray(a, dtype=np.int64) for a in actions],
                M=n_episodes, seed=seed)
    sim_vals = R.sum(axis=1)
    e_mean, e_se = env_vals.mean(), env_vals.std(ddof=1) / np.sqrt(n_episodes)
    s_mean, s_se = sim_vals.mean(), sim_vals.std(ddof=1) / np.sqrt(n_episodes)
    diff = s_mean - e_mean
    pooled = np.hypot(e_se, s_se)
    flag = 'OK' if abs(diff) <= 3 * pooled else 'MISMATCH'
    print(f'{name:34s} env {e_mean:8.3f} ± {e_se:.3f}   sim {s_mean:8.3f} ± {s_se:.3f}   '
          f'diff {diff:+7.3f} ({diff / max(pooled, 1e-9):+5.1f} SE)  {flag}', flush=True)
    return flag == 'OK'


if __name__ == '__main__':
    ok = True
    law = get_law('WC')

    # 1. ER graph, plan from the 'even' heuristic (includes re-seeding of removed nodes)
    G = er_graph(40, 120, seed=3)
    adj, indeg, _ = build_arrays(G)
    for alloc, rule in [('even', 'prob'), ('normal', 'prob'), ('static', 'indeg')]:
        actions = plan_episode_heuristic(law, adj, indeg, T=10, budget=20, rule=rule,
                                         alloc=alloc, seed=11)
        ok &= compare(f'ER40 {alloc}/{rule}', G, actions, n_episodes=1200)

    # 2. tiny instance where the whole budget lands in one round
    G2 = er_graph(25, 60, seed=5)
    adj2, indeg2, _ = build_arrays(G2)
    actions = plan_episode_heuristic(law, adj2, indeg2, T=4, budget=8, rule='outdeg',
                                     alloc='normal', seed=13)
    ok &= compare('ER25 normal/outdeg T=4 B=8', G2, actions, n_episodes=1500)

    # 3. real dataset subgraph (probabilities stripped so both use the weighted cascade)
    G3 = strip_weights(load_subgraph('wiki', 60, seed=21))
    adj3, indeg3, _ = build_arrays(G3)
    actions = plan_episode_heuristic(law, adj3, indeg3, T=10, budget=20, rule='prob',
                                     alloc='even', seed=17)
    ok &= compare('wiki60 even/prob', G3, actions, n_episodes=1200)

    print('\nVALIDATION:', 'PASS' if ok else 'FAIL')
