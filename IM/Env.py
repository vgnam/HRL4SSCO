# coding=utf-8
import numpy as np
import networkx as nx
import random


class Env:

    def __init__(self, graph, budget=20, T: int = 10, p_active=1, weighted=True):
        # graph
        self.graph = graph
        self.n = len(graph.nodes())
        self.nodes = list(range(self.n))
        self.A = np.asmatrix(nx.to_numpy_array(graph, nodelist=self.nodes))   # n * n
        self.weighted = weighted

        if weighted:
            for u, v in self.graph.edges():
                self.graph[u][v]['weight'] = self.get_edge_prob(u, v)
            self.A = np.asmatrix(nx.to_numpy_array(graph, nodelist=self.nodes, weight='weight'))

        # parameters
        self.T = T
        self.budget = budget
        self.budget_r = budget
        self.p_active = p_active

        # initial state
        self.state = np.zeros((self.n, 3))
        self.state[:, 0] = 1

    def get_info(self):
        return self.n, self.budget, self.T

    def reset(self):
        self.state = np.zeros((self.n, 3))
        self.state[:, 0] = 1
        self.budget_r = self.budget

    def step(self, state, action):
        state = state.copy()
        activated_seeds = self.activate_seeds(action)

        num_inactive_old = self.get_num_type(state, "inactive")

        self.set_state_index(state, activated_seeds, 1)

        # propogation
        active_nodes = self.get_nodes_type(state, "active")
        target_edges = self.get_inactive_edges(nodes=active_nodes)
        new_activated_nodes = self.activate_nodes(target_edges)
        new_activated_nodes = list(set(new_activated_nodes))

        # update state
        self.set_state_index(state, active_nodes, 2)
        self.set_state_index(state, new_activated_nodes, 1)
        next_state = state.copy()

        num_inactive_new = self.get_num_type(next_state, "inactive")
        reward = (num_inactive_old - num_inactive_new)
        done = self.is_stop(next_state)
        is_null = (len(activated_seeds) + len(new_activated_nodes) == 0)

        return next_state, reward, done, is_null

    def set_state_index(self, state_, nodes, state_index: int):
        state = state_
        for i in range(3):
            if i == state_index:
                state[nodes, i] = 1
            else:
                state[nodes, i] = 0

    def activate_seeds(self, seeds):
        activated_seeds = [
            v for v in seeds if random.uniform(0, 1) < self.p_active]
        return activated_seeds

    def activate_nodes(self, edges):
        # given a list of (active, inactive neighbor) edges, return active nodes
        active_nodes = [v for u, v in edges if random.uniform(
            0, 1) < self.get_edge_prob(u, v)]
        return active_nodes

    def get_nodes_type(self, state, type):
        if type == "inactive":
            return [v for v in self.nodes if state[v, 0] == 1]
        if type == "active":
            return [v for v in self.nodes if state[v, 1] == 1]
        if type == "removed":
            return [v for v in self.nodes if state[v, 2] == 1]
        else:
            raise ValueError(
                "type must be one of 'inactive', 'active', 'removed'")

    def get_num_type(self, state, type):
        nodes = self.get_nodes_type(state, type)
        num = len(nodes)
        return num

    def get_inactive_neighbors(self, state=None, nodes=None):
        # output is a 1-d list
        return [v for _, v in self.get_inactive_edges(state, nodes)]

    def get_inactive_edges(self, state=None, nodes=None):
        if state is None:
            state = self.state
        all_edges = []
        for u in nodes:
            all_edges.extend((u, v) for v in self.graph.neighbors(u))
        inactive_edges = [
            (u, v) for u, v in all_edges if state[v, 0] == 1 and v not in nodes]
        # output is a list of (source, target) edges
        return inactive_edges

    def get_edge_prob(self, u, v):
        # activation probability of edge u -> v: dataset probability if given,
        # otherwise the weighted cascade 1 / in_degree(v)
        return self.graph[u][v].get('p', 1/self.graph.in_degree(v))

    def is_stop(self, state):
        # since removed nodes can become inactive again.
        # if self.get_num_type(state, "inactive") == 0:
        #     return True
        if self.get_num_type(state, "active") == 0 and self.budget == 0:
            return True
        return False

    def sort_by_value(self, nodes, mode='score'):
        value_dict = self.get_node_value(nodes, mode)
        value_list_sorted = sorted(
            value_dict.items(), key=lambda x: x[1], reverse=True)
        nodes_sorted = [v[0] for v in value_list_sorted]
        return nodes_sorted

    # nodes: list
    def get_node_value(self, nodes, mode='score'):
        value_dict = dict()
        for v in nodes:
            inactive_neighbors = self.get_inactive_neighbors(nodes=[v])
            value = 0
            for u in inactive_neighbors:
                value += self.get_edge_prob(v, u) if mode == 'score' else 1
            value_dict[v] = value
        return value_dict
