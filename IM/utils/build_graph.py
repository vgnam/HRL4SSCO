import networkx as nx


def build_graph(file_path):
    G = nx.Graph()

    with open(file_path, 'r') as file:
        for line in file:
            start, end = map(int, line.strip().split())
            G.add_edge(start, end)

    return G


def build_directed_graph(file_path):
    G = nx.DiGraph()

    with open(file_path, 'r') as file:
        for line in file:
            start, end = map(int, line.strip().split())
            G.add_edge(start, end)

    return G
