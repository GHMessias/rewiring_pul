import json
import random
import torch
import networkx as nx
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

def train(data, model, optimizer, epochs, L, ro, P_rate, samples, rewiring = True):
    random.seed(100)
    torch.manual_seed(100)

    model.train()
    
    # opening json with informations about the graph and positive nodes
    json_path = 'rewiring_results/positive_nodes/positive_nodes.json'

    with open(json_path, 'r') as file:
        positive_dict = json.load(file)

    for sample in range(samples):
        P = positive_dict[f'L_{L}_P_{P_rate}_ro_{ro}_sample_{sample}']

        graphs = list()
        
        if rewiring:
            # In case of rewiring, we need to read the .graphml objects to create a list of graphs
            for i in range(L - 1):
                aux_graph = nx.read_graphml(f'rewiring_results/graphs/graph_L_{L}_P_{P_rate}_ro_{ro}_sample_{sample}_{i}.graphml')
                pyg_graph = from_networkx(aux_graph)
                graphs.append(pyg_graph)
            graphs = [data] + graphs
        else:
            # In case of not using rewiring, the list of graphs has only a single graph
            graphs = [data]

        # mask for training
        mask = torch.zeros(graphs[0].x.shape[0], dtype = torch.bool)

        for element in P:
            mask[element] = True

        losses = list()

        for epoch in range(epochs):
            optimizer.zero_grad()
            H_L = model.encode(graphs[0].x.float(), graphs)
            out = model.decode(H_L)

            loss = F.binary_cross_entropy(out, graphs[0].x.float())
            print(f'epoch: {epoch} | loss: {loss.item()}')
            loss.backward()
            optimizer.step()

            losses.append(loss)
        return losses






