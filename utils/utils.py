
import json
import networkx as nx

def json_writer(path, key, value):

    try:
        with open(path, 'r') as file:
            positive_nodes = json.load(file)

    except FileNotFoundError:
        positive_nodes = dict()

    positive_nodes[key] = value

    with open(path, 'w') as file:
        json.dump(positive_nodes, file, indent = 2)

def remove_elements(x,y):
    for element in y:
        if y in x:
            x.remove(y)
    return x

def bfs(graph, start_node, positives, vertex_to_add = 1):
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