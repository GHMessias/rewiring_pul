import json
import random
import torch
import networkx as nx
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, to_networkx
from utils.rewiring import rewiring
from utils.evaluate import negative_inference, evaluate_model, negative_inference_AE
import pandas as pd


def train(graph, model, optimizer, epochs, true_labels, P, P_rate, L = None , rewiring_usage = True, num_neg = 200, return_dataframe = False, scheduler = None):
    '''
    Treina os modelos que utilizam rewiring ou não

    Parameters
    graph: Grafo, objeto data do pytorch_geometric
    model: Modelo GAE a ser treinado
    optimizer: Otimizador pytorch
    epochs: Quantidade de épocas que o modelo será treinado
    true_labels: labels para computar a acurácia em cada iteração
    ro: Parametro do rewiring
    L: parametro do rewiring
    rewiring_usage: Variável para verificar se utiliza o rewiring ou não
    num_neg: Quantidade de vértices negativos inferidos
    return_dataframe: Variável para retornar o dataframe
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

        graph_list = rewiring(to_networkx(graph, to_undirected = True), L, P)
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

            if scheduler is not None:
                scheduler.step()
                

            losses.append(loss.item())
            _negatives = negative_inference(model, graph_list,num_neg)
            acc, f1 = evaluate_model(_negatives, true_labels)
            accs.append(acc)
            f1s.append(f1)

        if not return_dataframe:
            return {
                'losses': losses,
                'acc_per_epoch': accs,
                'f1_per_epoch': f1s
            }
        
        if return_dataframe:
            df = pd.DataFrame({
                'L': [L],
                'P': [P_rate],
                'acc': [accs[-1]],
                'f1': [f1s[-1]],
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
            
            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())

            _negatives = negative_inference(model, graph ,num_neg)
            acc, f1 = evaluate_model(_negatives, true_labels)
            accs.append(acc)
            f1s.append(f1)


        if not return_dataframe:
            return {
                'losses': losses,
                'acc_per_epoch': accs,
                'f1_per_epoch': f1s
            }
        
        if return_dataframe:
            df = pd.DataFrame({
                'L': [L],
                'P': [P_rate],
                'acc': [accs[-1]],
                'f1': [f1s[-1]],
                'acc_per_epoch': [accs],
                'f1_per_epoch': [f1s],
                'losses': [losses]
                })
            
            return df
            
def train_AE(data, model, optimizer, epochs, true_labels, P, P_rate,scheduler = None, num_neg = 200, return_dataframe = True):
    # mask for training
    mask = torch.zeros(data.shape[0], dtype = torch.bool)

    for element in P:
        mask[element] = True

    losses = list()
    accs = list()
    f1s = list()
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        H_L = model.encode(data)
        out = model.decode(H_L)

        loss = F.binary_cross_entropy(out, data)
        print(f'epoch: {epoch + 1} | loss: {loss.item()}', end = '\r')
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

        _negatives = negative_inference_AE(model, data)
        acc, f1 = evaluate_model(_negatives, true_labels)
        accs.append(acc)
        f1s.append(f1)

    if not return_dataframe:
        return {
            'losses': losses,
            'acc_per_epoch': accs,
            'f1_per_epoch': f1s
            }
        
    if return_dataframe:
        df = pd.DataFrame({
                'P': [P_rate],
                'acc': [accs[-1]],
                'f1': [f1s[-1]],
                'acc_per_epoch': [accs],
                'f1_per_epoch': [f1s],
                'losses': [losses]
            })
        
        return df


