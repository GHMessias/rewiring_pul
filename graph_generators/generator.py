import argparse
import random
import json
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import numpy as np
import torch
import sys

sys.path.insert(0, '')
from utils.rewiring import rewiring
from utils.utils import json_writer

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type = str, help = 'Rewiring Graph')
    args = parser.parse_args()
    name = args.name
    random.seed(42)

    json_path = 'rewiring_results/positive_nodes/positive_nodes.json'

    dataset = Planetoid(root = 'datasets', name = name)

    data = dataset[0]
    G = to_networkx(data)

    true_labels = [1 if y == 3 else 0 for y in data.y]

    all_positives = [x for x in list(G.nodes) if true_labels[x] == 1]

    L_range = [2,3,4,5]
    P_rate_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25]
    ro_range = [0.25, 0.5, 0.75, 1]


    # Initializing graph generation
    for P_rate in P_rate_range:
        for L in L_range:
            for ro in ro_range:
                G_0 = G.copy()
                print(f'initial edges: {len(G_0.edges)}')
                P = random.sample(all_positives, int(P_rate * len(all_positives)))
                print(P)

                graph_rewiring = rewiring(G_0, L, P, ro)

                for i in range(len(graph_rewiring)):
                    nx.write_graphml(graph_rewiring[i], f'rewiring_results/graphs/graph_L_{L}_P_{P_rate}_ro_{ro}_{i}.graphml')
            
                json_writer(json_path, f'{P_rate}_{L}_{ro}', P)



if __name__ == '__main__':
    main()
