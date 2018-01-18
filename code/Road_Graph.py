import numpy as np
import networkx as nx

class Road_Graph():
    def __init__(self, graph_dict = {}):
        self.graph = self._build_graph(graph_dict)
    def _build_graph(self, graph_dict):
        nodes = list(graph_dict.keys())
        edges = [(m, n) for m in graph_dict.keys() for n in graph_dict[m]]
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G
    @property
    def nodes(self):
        return np.array(self.graph.nodes)
    @property
    def edges(self):
        return np.array(self.graph.edges)
    def neighbors(self, x):
        if type(x) == list or type(x) == np.ndarray:
            return [list(self.graph.neighbors(i)) for i in x]
        return np.array(list(self.graph.neighbors(x)))
    def read_pathes(self, pathes):
        for p in pathes:
            self.graph.add_path(p)
