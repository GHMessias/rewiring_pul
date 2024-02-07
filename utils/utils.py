import json
import networkx as nx
from collections import deque
import random

def remove_elements(x,y):
    '''
    remove os elementos de x que estão em y

    Parameters
    x: list
    y: list

    Returns
    list: Lista com os elementos de y que não estão em x
    '''
    for element in y:
        if y in x:
            x.remove(y)
    return x

def bfs(graph, start_node, positives, vertex_to_add = 1):
    '''
    Faz a busca em profundidade. Dado um nó de partida, adiciona uma aresta aos primeiros "vertex_to_add" vertices encontrados

    Parameters
    graph: Grafo, objeto data pytorch_geometric
    start_node: Nó inicial
    positives: Espaço de busca para o algoritmo BFS
    vertex_to_add: Quantidade de vértices para criar uma aresta que liga o start node em vertices positivos.

    returns
    Nonetype: Adiciona diretamente as arestas
    '''
    visited = set()
    queue = [start_node]

    positives = remove_elements(positives, [start_node] + list(graph.neighbors(start_node)))
    
    added_vertex = 0

    while queue  and added_vertex <= vertex_to_add :
        current_node = queue.pop(0)
        if current_node not in visited:
            visited.add(current_node)
            neighbors = list(graph.neighbors(current_node))
            queue.extend(neighbors)
        if current_node in positives:
            graph.add_edge(start_node, current_node)
            added_vertex += 1

     
def bfs_distance(graph, start, end):
    if start == end:
        return 0
    
    visited = set()
    queue = deque([(start, 0)])

    while queue:
        current_node, distance = queue.popleft()

        if current_node == end:
            return distance

        if current_node not in visited:
            visited.add(current_node)
            neighbors = graph.neighbors(current_node)
            queue.extend((neighbor, distance + 1) for neighbor in neighbors if neighbor not in visited)

    # Se não for possível alcançar o vértice de destino a partir do vértice de origem
    return float('inf')

def rng(prob):
    # Gera um número aleatório entre 0 e 1
    numero_aleatorio = random.random()
    return numero_aleatorio <= prob