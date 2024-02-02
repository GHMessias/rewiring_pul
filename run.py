import argparse
from utils.rewiring import rewiring
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, from_networkx
from models.RGCN_model import RGCN_model
from models.GCN_model import GCN_model
from models.MLP_model import MLP_model
import torch
from torch_geometric.nn import GAE
from utils.training import train
from utils.evaluate import negative_inference, evaluate_model
import json
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser()

# Setting variables for all_training
parser.add_argument('--training_type', type = str, default = 'general')
parser.add_argument("--name", type = str, help = 'Cora')
parser.add_argument("--samples", type = int, default = 1)
parser.add_argument('--model', type = str, default = "RGCN, GCN, random") #RGCN, GCN
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--L2norm', type = float, default=0)

# If training_type is general
parser.add_argument('--ro_range', nargs= '+', type = int, default = [1,3,5,7,9])
parser.add_argument('--L_range', nargs='+', type = int, default = [2,3,4,5])
parser.add_argument('--P_rate_range', type=float, nargs='+', default = [n/100 for n in range(1,26)])

# If training_type is pontual
parser.add_argument('--L', type = int, default=3)
parser.add_argument('--ro', type = float, default=3.0)
parser.add_argument('--P_rate', type = float, default=0.05)

args = parser.parse_args()

if __name__ == '__main__':

    dataset = Planetoid(root = 'datasets', name = args.name)
    data = dataset[0]
    random.seed(100)
    torch.manual_seed(100)

    if args.training_type == 'general':
        # Defining NN parameters
        in_channels = data.x.shape[0]
        # Arrumar isso pro args
        hidden_channels = 64
        out_channels = 16
        
        true_labels = np.array([1 if x == 3 else 0 for x in data.y])
        all_positives = np.array([x for x in range(data.x.shape[0]) if true_labels[x] == 1])

        df = pd.DataFrame()
        # for sample in range(args.samples):
        for prate in args.P_rate_range:
            P = random.sample(all_positives.tolist(), int(prate * len(all_positives)))
            for lrate in args.L_range:
                for rorate in args.ro_range:
                    if "RGCN" in args.model:
                        
                        print(f'\n P = {prate}, L = {lrate}, ro = {rorate}')
                        RGCN_encoder = RGCN_model(data.x.shape[1], hidden_channels, out_channels, L = lrate)
                        RGCN_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
                        GAE_RGCN = GAE(encoder = RGCN_encoder, decoder = RGCN_decoder)
                        optimizer_RGCN = torch.optim.Adam(GAE_RGCN.parameters(), lr = args.lr, weight_decay = args.L2norm)
                        GAE_RGCN.float()
                        results = train(graph = data,
                                        model = GAE_RGCN,
                                        optimizer = optimizer_RGCN,
                                        epochs = args.epochs,
                                        P = P,
                                        P_rate = prate,
                                        true_labels = true_labels,
                                        ro = rorate,
                                        L = lrate,
                                        rewiring_usage = True,
                                        return_dataframe = True)
                        results['model'] = 'RGCN'
                        df = pd.concat([df, results])

                    if "GCN" in args.model:
                        GCN_encoder = GCN_model(data.x.shape[1], hidden_channels, out_channels)
                        GCN_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
                        GAE_GCN = GAE(encoder = GCN_encoder, decoder = GCN_decoder)
                        optimizer_GCN = torch.optim.Adam(GAE_GCN.parameters(), lr = args.lr, weight_decay = args.L2norm)
                        GAE_GCN.float()
                        results = train(graph = data,
                                        model = GAE_GCN,
                                        optimizer = optimizer_GCN,
                                        epochs = args.epochs,
                                        P = P,
                                        P_rate = prate,
                                        true_labels = true_labels,
                                        rewiring_usage=False,
                                        return_dataframe=True
                        )
                        results['model'] = 'GCN'
                    df.to_csv('results/dataframe_results.csv',index = False)
                    
    
    if args.training_type == 'pontual':
        true_labels = np.array([1 if x == 3 else 0 for x in data.y])
        all_positives = np.array([x for x in range(data.x.shape[0]) if true_labels[x] == 1])
        in_channels = data.x.shape[0]
        # Arrumar isso pro args
        hidden_channels = 64
        out_channels = 16

        P = random.sample(all_positives.tolist(), int(args.P_rate * len(all_positives)))
        
        RGCN_encoder = RGCN_model(data.x.shape[1], hidden_channels, out_channels, L = args.L)
        RGCN_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
        GAE_RGCN = GAE(encoder = RGCN_encoder, decoder = RGCN_decoder)
        optimizer_RGCN = torch.optim.Adam(GAE_RGCN.parameters(), lr = args.lr, weight_decay = args.L2norm)
        GAE_RGCN.float()

        results = train(graph = data,
                            model = GAE_RGCN,
                            optimizer=optimizer_RGCN,
                            epochs = args.epochs,
                            P = P,
                            true_labels=true_labels,
                            ro = args.ro,
                            L = args.L,
                            rewiring_usage = True)
        
        print(results['acc_per_epoch'][-1], results['f1_per_epoch'][-1], max(results['acc_per_epoch']), max(results['f1_per_epoch']))
        

