import json
import random
import torch
import networkx as nx
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, to_networkx
from utils.rewiring import rewiring
from utils.evaluate import negative_inference, evaluate_model
import pandas as pd


def train(graph, model, optimizer, epochs, true_labels, P, P_rate, ro = None , L = None , rewiring_usage = True, num_neg = 100, return_dataframe = False):
    '''
    A partir dos dados positivos, gera os grafos para valores de L e ro. Esses grafos são utilizados para treinar o modelo de GCN
    '''

    if rewiring_usage:
        # mask for training
        mask = torch.zeros(graph.x.shape[0], dtype = torch.bool)

        for element in P:
            mask[element] = True

        losses = list()
        accs = list()
        f1s = list()

        model.train()

        graph_list = rewiring(to_networkx(graph), L, P, ro)
        graph_list = [graph] + [from_networkx(G) for G in graph_list]
        # print(len(graph_list)) 

        for epoch in range(epochs):
            optimizer.zero_grad()
            H_L = model.encode(graph.x.float(), graph_list)
            out = model.decode(H_L)

            loss = F.binary_cross_entropy(out, graph_list[0].x.float())
            print(f'epoch: {epoch + 1} | loss: {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            _negatives = negative_inference(model, graph_list,num_neg)
            acc, f1 = evaluate_model(_negatives, true_labels)
            accs.append(acc)
            f1s.append(f1)

        if not return_dataframe:
            return {
                'losses': losses,
                'acc_per_epoch': accs,
                'f1_per_epoch': f1s,
            }
        
        if return_dataframe:
            df = pd.DataFrame({
                'L': [L],
                'ro': [ro],
                'P': [P_rate],
                'final_acc': [accs[-1]],
                'final_f1': [f1s[-1]],
                'max_acc': [max(accs)],
                'max_f1': [max(f1s)],
                'acc_per_epoch': [accs],
                'f1_per_epoch': [f1s],
                'losses': [losses]
                })
            
            return df

    
    else:
        # mask for training
        mask = torch.zeros(graph.x.shape[0], dtype = torch.bool)

        for element in P:
            mask[element] = True

        losses = list()
        accs = list()
        f1s = list()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            H_L = model.encode(graph.x.float(), graph.edge_index)
            out = model.decode(H_L)

            loss = F.binary_cross_entropy(out, graph.x.float())
            print(f'epoch: {epoch + 1} | loss: {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()

            losses.append(loss)

            _negatives = negative_inference(model, graph ,num_neg)
            acc, f1 = evaluate_model(_negatives, true_labels)
            accs.append(acc)
            f1s.append(f1)

            if not return_dataframe:
                return {
                    'losses': losses,
                    'acc_per_epoch': accs,
                    'f1_per_epoch': f1s,
                }
            
            if return_dataframe:
                df = pd.DataFrame({
                    'L': [L],
                    'ro': [ro],
                    'P': [P_rate],
                    'final_acc': [accs[-1]],
                    'final_f1': [f1s[-1]],
                    'max_acc': [max(accs)],
                    'max_f1': [max(f1s)],
                    'acc_per_epoch': [accs],
                    'f1_per_epoch': [f1s],
                    'losses': [losses]
                    })
                
                return df
            



