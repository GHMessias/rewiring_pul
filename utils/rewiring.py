#TODO - consertar o erro na função de rewiring
#TODO - trocar o dijkstra médio pela profundidade

import networkx as nx
from utils.utils import bfs

def rewiring(graph, L, P, ro):
    rewiring_graphs = list()
    # print(len(graph.edges))
    for l in range(L):
        for node in P:
            bfs(graph, node, P, ro)
            # print(len(graph.edges()))
        rewiring_graphs.append(graph.copy())
    return rewiring_graphs
        