import json
import random
import torch
import networkx as nx
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

def train(model, optimizer, epochs, P, rewiring = True, graph_list = None, graph = None):

    if rewiring:
        # mask for training
        mask = torch.zeros(graph_list[0].x.shape[0], dtype = torch.bool)

        for element in P:
            mask[element] = True

        losses = list()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            H_L = model.encode(graph_list[0].x.float(), graph_list)
            out = model.decode(H_L)

            loss = F.binary_cross_entropy(out, graph_list[0].x.float())
            print(f'epoch: {epoch} | loss: {loss.item()}')
            loss.backward()
            optimizer.step()

            losses.append(loss)
        return losses

        
    else:
        # mask for training
        mask = torch.zeros(graph.x.shape[0], dtype = torch.bool)

        for element in P:
            mask[element] = True

        losses = list()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            H_L = model.encode(graph.x.float(), graph)
            out = model.decode(H_L)

            loss = F.binary_cross_entropy(out, graph.x.float())
            print(f'epoch: {epoch} | loss: {loss.item()}')
            loss.backward()
            optimizer.step()

            losses.append(loss)

            return losses






