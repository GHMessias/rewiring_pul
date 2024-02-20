from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import random
from utils.rewiring import prob_rewiring
from datasets.FakeBr.FakeBr import FakeBr

def rng(prob):
    # Gera um número aleatório entre 0 e 1
    numero_aleatorio = random.random()
    return numero_aleatorio <= prob

dataset = FakeBr(root = 'datasets/FakeBr')
data = dataset.get()
# data = dataset[0]

G = to_networkx(data, to_undirected = True)
y = [1 if data.y[x] == 2 else 0 for x in range(len(G.nodes))]
all_positives = [x for x in range(len(G.nodes)) if y[x] == 1]
P = random.sample(all_positives, int(0.05 * len(all_positives)))

print(len(G.edges))
for l in range(3):
    edges_to_add = list()
    for v1 in P:
        for v2 in P:
            if v1 != v2:
                print(f'source {v1} \t | target {v2} \t : {round(prob_rewiring(v1, v2, G, P, 0.3, 0.5), 4)}')
                prob = round(prob_rewiring(v1, v2, G, P, 0.3, 0.5), 4)
                if rng(prob):
                    edges_to_add.append((v1,v2))

    G.add_edges_from(edges_to_add)
    print(len(G.edges))
