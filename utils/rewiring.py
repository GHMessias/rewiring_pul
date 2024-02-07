import networkx as nx
from utils.utils import bfs, bfs_distance, rng

def rewiring(graph, L, P):
    '''
    Aplica a função de rewiring

    Parameters
    graph: Grafo, objeto data do pytorch_geometric
    L: Quantidade de grafos para atribuir ao rewiring
    P: Vértices positivos
    ro: Quantidade de vértices que devem ser ligados a um vértice positivo

    Returns
    List: Lista de grafos com L elementos feitos rewiring.

    '''
    print(len(graph.edges))
    rewiring_graphs = list()
    for l in range(L):
        edges_to_add = list()
        for v1 in P:
            for v2 in P:
                if v1 != v2:
                    prob = round(prob_rewiring(v1, v2, graph, P, 0.3, 0.5), 4)
                    if rng(prob):
                        edges_to_add.append((v1,v2))
        graph.add_edges_from(edges_to_add)
        print(len(graph.edges))
        rewiring_graphs.append(graph.copy())
    return rewiring_graphs

def prob_rewiring(v1, v2, graph, P, beta, gamma):
    '''
    Calcula a probabilidade de um vértice se ligar com outro, através da equação
    phi(v1,v2) = 1 / (BFS(v1,v2)^beta * |P|^gamma)

    Parameters
    graph: Grafo, objeto data pytorch_geometric
    P: Lista de vértices positivos
    v1: vértice 
    v2: vértice
    alpha, beta, gamma: float
    '''
    prob = 1 / ((bfs_distance(graph, v1, v2) ** beta) * (len(P) ** gamma))

    if prob == 0:
        return 1
    else:
        return prob
        