import numpy as np
from datasets.FakeBr.FakeBr import FakeBr
import random
from utils.evaluate import evaluate_model
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import accuracy_score, f1_score
def data_to_adjacency_matrix(data):
    num_nodes = data.num_nodes
    edge_index = data.edge_index.numpy()
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    
    return adjacency_matrix

def PULP(P, U, m, l, A, alpha):
    RP = list()
    P_line = P.copy()
    U_line = U.copy()

    # Calculando W = (I - alpha*A)**-1 - I
    I = np.eye(len(A))
    W = np.linalg.inv(I - alpha * A) - I

    for k in range(m):
        rank_dict = dict()
        for vi in U_line:
            S_vi = 0
            for vj in P_line:
                S_vi += W[vi, vj]
            S_vi /= len(P_line)
            rank_dict[vi] = S_vi
        
        rank_dict = sorted(rank_dict.items(), key=lambda x:x[1], reverse=True)
        rank_dict = [tupla[0] for tupla in rank_dict]
        # print(rank_dict)
        RP_line = rank_dict[:int((l / m) * len(P))]
        P_line = P_line + RP_line
        U_line = list(set(U_line) - set(RP_line))
        RP = RP + RP_line

    rank_dict = dict()
    for vi in list(set(U) - set(RP_line)):
        S_vi = 0
        for vj in P + RP:
            S_vi += W[vi, vj]
        S_vi /= len(P_line)
        rank_dict[vi] = S_vi
    rank_dict = sorted(rank_dict.items(), key=lambda x:x[1])
    rank_dict = [tupla[0] for tupla in rank_dict]
    RN = rank_dict[:len(P + RP)]
    return RN, RP

dataset = FakeBr(root = 'datasets/FakeBr')
data = dataset.get()

true_labels = np.array([1 if x == 1 else 0 for x in data.y])
all_positives = [x for x in range(data.x.shape[0]) if true_labels[x] == 1]
P = random.sample(all_positives, int(0.1 * len(all_positives)))
U = [x for x in range(data.x.shape[0]) if x not in P]

RN, RP = PULP(P = P, U = U, m = 3, l = 1, alpha = 0.1, A = data_to_adjacency_matrix(data))
print(len(RN))
print(len(RP + P))
print(len(data.y))

G = to_networkx(data, to_undirected=True)

for x in P+RP:
    G.nodes[x]['label'] = 1

for x in RN:
    G.nodes[x]['label'] = 0

result = nx.algorithms.node_classification.local_and_global_consistency(G)

print(accuracy_score(true_labels, result))
print(f1_score(true_labels, result))