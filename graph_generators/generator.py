import random
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import numpy as np
import sys

sys.path.insert(0, '')
from utils.rewiring import rewiring
from utils.utils import json_writer

def generator(data, samples, L_range, P_rate_range, ro_range):
    random.seed(100)

    for sample in range(samples):

        json_path = 'rewiring_results/positive_nodes/positive_nodes.json'

        G = to_networkx(data)

        true_labels = [1 if y == 3 else 0 for y in data.y]

        all_positives = [x for x in list(G.nodes) if true_labels[x] == 1]


        # Initializing graph generation
        for P_rate in P_rate_range:
            for L in L_range:
                for ro in ro_range:
                    G_0 = G.copy()
                    print(f'initial edges: {len(G_0.edges)}')
                    P = random.sample(all_positives, int(P_rate * len(all_positives)))
                    print(P)

                    print(P_rate, L, ro, end = '\r')
                    graph_rewiring = rewiring(G_0, L, P, ro)

                    for i in range(len(graph_rewiring)):
                        nx.write_graphml(graph_rewiring[i], f'rewiring_results/graphs/graph_L_{L}_P_{P_rate}_ro_{ro}_sample_{sample}_{i}.graphml')
                
                    json_writer(json_path, f'L_{L}_P_{P_rate}_ro_{ro}_sample_{sample}', P)