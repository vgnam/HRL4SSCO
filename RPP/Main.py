import argparse
import networkx as nx
import numpy as np
import random
import torch
import sys
from Agent import Agent
from Env import Env
from Baselines import *
from real_data import DATASETS, instance_sampler

params = {
    'index': 1,
    'n': 100,     # number of nodes
    'test_n': 100,
    'budget': 20,  # number of budget
    'T': 10,     # number of time steps
    'num_epoch': 20,
    'num_iteration': 10,
    'num_test': 100,
    'seed': 0,
    'train_dataset': None,  # None: random instances
    'test_dataset': None,   # None: random instance; flickr-* uses all POIs of the city as test_n
    'sampling': 'knn',      # venue sampling for foursquare-*: 'knn' or 'uniform'
    'drift': 'default',     # daily profit drift: 'default' (0.05) or 'fitted' from the dataset
}
parser = argparse.ArgumentParser()
parser.add_argument('seed', nargs='?', type=int, default=params['seed'])
parser.add_argument('--train-dataset', choices=DATASETS, default=params['train_dataset'])
parser.add_argument('--test-dataset', choices=DATASETS, default=params['test_dataset'])
parser.add_argument('--sampling', choices=['knn', 'uniform'], default=params['sampling'])
parser.add_argument('--drift', choices=['default', 'fitted'], default=params['drift'])
args = parser.parse_args()
params['seed'] = args.seed
params['train_dataset'] = args.train_dataset
params['test_dataset'] = args.test_dataset
params['sampling'] = args.sampling
params['drift'] = args.drift


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_job_name(params):
    job_name = f"job{params['index']}-n={params['n']}-text_n={params['test_n']}-budget={params['budget']}-T={params['T']}-epoch={params['num_epoch']}-iteration={params['num_iteration']}-seed={params['seed']}"
    if params['train_dataset'] is not None:
        job_name += f"-train={params['train_dataset']}"
    if params['test_dataset'] is not None:
        job_name += f"-test={params['test_dataset']}"
    if params['train_dataset'] is not None or params['test_dataset'] is not None:
        job_name += f"-{params['sampling']}-drift={params['drift']}"
    return job_name


def dataset_instance_sampler(name, n, seed):
    # flickr-* instances are whole cities, so n is not used
    return instance_sampler(name, None if name.startswith('flickr') else n,
                            method=params['sampling'], drift=params['drift'], seed=seed)


# run the testing rounds and write results
def run_test_rounds(model, params, job_name):

    txt_name = f"results/{job_name}-test.txt"
    with open(txt_name, 'w', encoding='utf-8') as f:
        sys.stdout = f
        cumulative_reward_test = []
        for i in range(params['num_test']):
            print(f"test round: {i}")
            _, _, _, _, _, _, cumulative_reward = model.run_episode(
                beam_search=True)
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
        if params['train_dataset'] is None:
            loc = np.random.rand(params['n'], 2) * 100
            env = Env(loc=loc, budget=params['budget'], T=params['T'])
        else:
            train_sampler = dataset_instance_sampler(params['train_dataset'], params['n'], params['seed'])
            loc, initial_profit, profit_factor = train_sampler()
            env = Env(loc=loc, initial_profit=initial_profit, budget=params['budget'],
                      T=params['T'], profit_factor=profit_factor)
        model = Agent(env=env)
        if params['train_dataset'] is not None:
            model.instance_sampler = train_sampler

        train_cumulative_reward_list, test_cumulative_reward_list = model.train_model(
            num_epochs=params['num_epoch'], num_iterations=params['num_iteration']
        )

        print("------------------------------ Total Results ------------------------------")
        print("train_cumulative_reward_list: ", train_cumulative_reward_list)
        print("test_cumulative_reward_list: ", test_cumulative_reward_list)

    # Testing phase
    if params['test_dataset'] is None:
        loc = np.random.rand(params['test_n'], 2) * 100
        model.set_loc(loc)
    else:
        model.set_loc(*dataset_instance_sampler(params['test_dataset'], params['test_n'], params['seed'] + 1)())
    model.eval()
    model.discount = 1.0

    # Run test rounds for all policy pairs
    run_test_rounds(model, params, job_name)

    # baseline
    txt_name = f"results/{job_name}-greedy.txt"
    with open(txt_name, 'w', encoding='utf-8') as f:
        sys.stdout = f
        cumulative_reward_test = []
        greedy = Greedy(env=model.env)
        for i in range(params['num_test']):
            print(f"test round: {i}")
            seeding_policy, cumulative_reward = greedy.run()
            print(f"cumulative reward of round {i}: {cumulative_reward}")
            print(f"seeding_policy: {seeding_policy}")
            cumulative_reward_test.append(cumulative_reward)
        avg_reward = np.average(cumulative_reward_test)
        std_dev = np.std(cumulative_reward_test)
        print(f"average_cumulative_reward: {avg_reward}, std: {std_dev}")

    txt_name = f"results/{job_name}-ga.txt"
    with open(txt_name, 'w', encoding='utf-8') as f:
        sys.stdout = f
        cumulative_reward_test = []
        ga = GA(env=model.env)
        for i in range(params['num_test']):
            print(f"test round: {i}")
            seeding_policy, cumulative_reward = ga.run()
            print(f"cumulative reward of round {i}: {cumulative_reward}")
            print(f"seeding_policy: {seeding_policy}")
            cumulative_reward_test.append(cumulative_reward)
        avg_reward = np.average(cumulative_reward_test)
        std_dev = np.std(cumulative_reward_test)
        print(f"average_cumulative_reward: {avg_reward}, std: {std_dev}")
