import numpy as np

def random_path(graph, start = None, stop_prob = 0.1):
    """
    Generate random path on graph G with lenght at least 2
    """
    n = len(graph.nodes)
    if not start:
        start = np.random.choice(n)
    path = [start]
    while True:
        nxt = np.random.choice(graph.neighbors(start))
        path.append(nxt)
        start = nxt
        if np.random.rand() < stop_prob:
            break
    return np.array(path)

def random_path_not_back(graph, start = None, stop_prob = 0.1):
    """
    Generate random path on road graph with lenght at least 2. (not go back version)
    """
    n = len(graph.nodes)
    if not start:
        start = np.random.choice(n)
    path = [start]
    prev = None
    while True:
        neighbors = list(graph.neighbors(start))
        if prev in neighbors:
            neighbors.remove(prev)
        nxt = np.random.choice(neighbors)
        path.append(nxt)
        prev = start
        start = nxt
        if np.random.rand() < stop_prob:
            break
    return np.array(path)

def random_walk_data(graph, size, stop_prob = 0.1, go_back = True):
    if go_back:
        return [random_path(graph, stop_prob = stop_prob) for _ in range(size)]
    else:
        return [random_path_not_back(graph, stop_prob = stop_prob) for _ in range(size)]
