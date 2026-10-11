import networkx as nx
import numpy as np
import random
import torch
import sys
from Agent import Agent
from Env import Env

params = {
    'index': 1,
    'n': 100,     # number of nodes (paper: 50-100; replay buffer needs a fixed size)
    'test_n': 200,
    'budget': 20,  # number of budget
    'test_budget': 20,
    'T': 10,     # number of time steps
    'num_epoch': 20,
    'num_iteration': 10,
    'num_test': 100,
    'seed': 0,
}
if len(sys.argv) > 1:
    params['seed'] = int(sys.argv[1])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_job_name(params):
    return f"job{params['index']}-n={params['n']}-test_n={params['test_n']}-budget={params['budget']}-test_budget={params['test_budget']}-T={params['T']}-epoch={params['num_epoch']}-iteration={params['num_iteration']}-seed={params['seed']}"


# run the testing rounds and write results
def run_test_rounds(model, params, job_name, policy_pairs):
    for policy in policy_pairs:
        budget_policy, seeding_policy = policy
        txt_name = f"results/{job_name}-{budget_policy}-{seeding_policy}.txt"
        with open(txt_name, 'w', encoding='utf-8') as f:
            sys.stdout = f
            cumulative_reward_test = []
            for i in range(params['num_test']):
                print(f"test round: {i}")
                _, _, _, _, _, cumulative_reward = model.run_episode(
                    budget_policy=budget_policy,
                    seeding_policy=seeding_policy,
                    beam_search=('agent' in seeding_policy)
                )
                print(f"cumulative reward of round {i}: {cumulative_reward}")
                cumulative_reward_test.append(cumulative_reward)
            avg_reward = np.average(cumulative_reward_test)
            std_dev = np.std(cumulative_reward_test)
            print(f"average_cumulative_reward: {avg_reward}, std: {std_dev}")


if __name__ == "__main__":
    set_seed(params['seed'])
    job_name = generate_job_name(params)

    # Training phase
    txt_name = f"results/{job_name}-training.txt"
    with open(txt_name, 'w', encoding='utf-8') as f:
        sys.stdout = f
        g = nx.erdos_renyi_graph(params['n'], 0.01, directed=True)
        env = Env(graph=g, budget=params['budget'], T=params['T'])
        model = Agent(env=env)

        train_cumulative_reward_list, test_cumulative_reward_list = model.train_model(
            num_epochs=params['num_epoch'], num_iterations=params['num_iteration']
        )

        print("------------------------------ Total Results ------------------------------")
        print("train_cumulative_reward_list: ", train_cumulative_reward_list)
        print("test_cumulative_reward_list: ", test_cumulative_reward_list)

    # Testing phase
    g = nx.erdos_renyi_graph(params['test_n'], 0.01, directed=True)
    model.set_graph(g)
    model.env.budget = params['test_budget']
    model.env.reset()
    model.eval()

    # Define different budget and seeding policy pairs
    policy_pairs = [
        ('agent', 'agent'), ('average', 'score'), ('average', 'degree'),
        ('average', 'agent'), ('static', 'score'), ('static', 'degree'),
        ('static', 'agent'), ('normal', 'score'), ('normal', 'degree'),
        ('normal', 'agent'), ('agent', 'score'), ('agent', 'degree')
    ]

    # Run test rounds for all policy pairs
    run_test_rounds(model, params, job_name, policy_pairs)
