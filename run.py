'''
Função principal para determinar os treinamentos de cada rede.
--training_type general -> faz o teste paramétrico nos modelo para identificar os melhores parâmetros para cada valor de dados positivos
                           salva esses valores no arquivo dataframe_results.csv
--training_type pontual -> faz o treino para um determinados parâmetros
'''

import argparse
from utils.rewiring import rewiring
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, from_networkx
from models.RGCN_model import RGCN_model
from models.GCN_model import GCN_model
from models.MLP_model import MLP_model
from models.CCRNE import CCRNE
from models.MCLS import MCLS
from models.LP_PUL import LP_PUL
from models.PU_LP import PU_LP
from models.RCSVM_RN import RCSVM_RN
import torch
from torch_geometric.nn import GAE
from utils.training import train, train_AE
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
parser.add_argument("--samples", type = int, default = 5)
parser.add_argument('--model', type = str, default = "RGCN GCN LPPUL PULP MCLS RCSVM CCRNE AE RANDOM") #RGCN, GCN
parser.add_argument('--epochs', type = int, default = 20)
parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--L2norm', type = float, default=1e-4)
parser.add_argument('--lr_scheduler', type = float, default =0.9)
parser.add_argument('--positive_class', type = int, default = 3)

# If training_type is general
parser.add_argument('--L_range', nargs='+', type = int, default = [2,3,4,5])
parser.add_argument('--P_rate_range', type=float, nargs='+', default = [n/100 for n in range(1,26)])

# If training_type is pontual
parser.add_argument('--L', type = int, default=5)
parser.add_argument('--P_rate', type = float, default=0.05)

args = parser.parse_args()

if __name__ == '__main__':

    dataset = Planetoid(root = 'datasets', name = args.name)
    data = dataset[0]
    # random.seed(2)
    # torch.manual_seed(2)

    if args.training_type == 'general':  
        # Defining NN parameters
        in_channels = data.x.shape[0]
        # Arrumar isso pro args
        hidden_channels = 64
        out_channels = 16
        
        true_labels = np.array([1 if x == args.positive_class else 0 for x in data.y])
        all_positives = np.array([x for x in range(data.x.shape[0]) if true_labels[x] == 1])
        

        df = pd.DataFrame()
        for sample in range(args.samples):
            print(f'################ SAMPLE {sample} ################ \n')
            for prate in args.P_rate_range:
                P = random.sample(all_positives.tolist(), int(prate * len(all_positives)))
                U = np.array([x for x in range(data.x.shape[0]) if x not in P])
                if "RGCN" in args.model:
                    for lrate in args.L_range:
                        print(f'\n P = {prate}, L = {lrate}')
                        RGCN_encoder = RGCN_model(data.x.shape[1], hidden_channels, out_channels, L = lrate)
                        RGCN_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
                        GAE_RGCN = GAE(encoder = RGCN_encoder, decoder = RGCN_decoder)
                        optimizer_RGCN = torch.optim.Adam(GAE_RGCN.parameters(), lr = args.lr, weight_decay = args.L2norm)
                        scheduler_RGCN = torch.optim.lr_scheduler.StepLR(optimizer_RGCN, step_size = 1, gamma = args.lr_scheduler)
                        GAE_RGCN.float()
                        results = train(graph = data,
                                        model = GAE_RGCN,
                                        optimizer = optimizer_RGCN,
                                        epochs = args.epochs,
                                        P = P,
                                        P_rate = prate,
                                        true_labels = true_labels,
                                        L = lrate,
                                        scheduler=scheduler_RGCN,
                                        rewiring_usage = True,
                                        return_dataframe = True)
                        results['model'] = 'RGCN'
                        df = pd.concat([df, results])

                if "GCN" in args.model:
                    GCN_encoder = GCN_model(data.x.shape[1], hidden_channels, out_channels)
                    GCN_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
                    GAE_GCN = GAE(encoder = GCN_encoder, decoder = GCN_decoder)
                    optimizer_GCN = torch.optim.Adam(GAE_GCN.parameters(), lr = args.lr, weight_decay = args.L2norm)
                    scheduler_GCN = torch.optim.lr_scheduler.StepLR(optimizer_GCN, step_size = 1, gamma = args.lr_scheduler)
                    GAE_GCN.float()
                    results = train(graph = data,
                                    model = GAE_GCN,
                                    optimizer = optimizer_GCN,
                                    epochs = args.epochs,
                                    P = P,
                                    P_rate = prate,
                                    true_labels = true_labels,
                                    scheduler=scheduler_GCN,
                                    rewiring_usage=False,
                                    return_dataframe=True
                    )
                    results['model'] = 'GCN'
                    df = pd.concat([df, results])
                
                if "CCRNE" in args.model:
                    print('CCRNE Classifier')
                    ccrne_classifier = CCRNE(data.x, P, U.tolist())
                    ccrne_classifier.train()
                    negatives = ccrne_classifier.negative_inference(200)
                    print(len(negatives))
                    acc, f1 = evaluate_model(negatives, true_labels)
                    results = pd.DataFrame({'model': ['CCRNE'], 'acc': [acc], 'f1': [f1], 'P': prate})
                    df = pd.concat([df, results])

                if "MCLS" in args.model:
                    mcls_classifier = MCLS(data.x, P, 7, 0.3)
                    mcls_classifier.train()
                    negatives = mcls_classifier.negative_inference(200)
                    # print(negatives)
                    acc, f1 = evaluate_model(negatives, true_labels)
                    results = pd.DataFrame({'model': ['MCLS'], 'acc': [acc], 'f1': [f1], 'P' : prate})
                    df = pd.concat([df, results])

                if "LPPUL" in args.model:
                    lp_pul_classifier = LP_PUL(to_networkx(data, to_undirected = True), data.x, P, U)
                    lp_pul_classifier.train()
                    negatives = lp_pul_classifier.negative_inference(200)
                    # print(negatives)
                    acc, f1 = evaluate_model(negatives, true_labels)
                    results = pd.DataFrame({'model': ['LP_PUL'], 'acc': [acc], 'f1': [f1], 'P': prate})
                    df = pd.concat([df, results])

                if "PULP" in args.model:
                    pulp_classifier = PU_LP(data.x, P, U, alpha=0.3, m = 3, l = 1)
                    pulp_classifier.train()
                    negatives = pulp_classifier.negative_inference(200)
                    # print(negatives)
                    acc, f1 = evaluate_model(negatives, true_labels)
                    results = pd.DataFrame({'model': ['PU_LP'], 'acc': [acc], 'f1': [f1], 'P':prate})
                    df = pd.concat([df, results])

                if "RCSVM" in args.model:
                    rcsvm_classifier = RCSVM_RN(data.x, P, U, alpha = 0.1, beta = 0.9)
                    rcsvm_classifier.train()
                    negatives = rcsvm_classifier.negative_inference(200)
                    print(negatives)
                    acc, f1 = evaluate_model(negatives, true_labels)
                    results = pd.DataFrame({'model': ['RCSVM'], 'acc': [acc], 'f1': [f1], 'P': prate})
                    df = pd.concat([df, results])

                if "AE" in args.model:
                    AE_encoder = MLP_model(data.x.shape[1], hidden_channels, out_channels)
                    AE_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
                    AE = GAE(encoder = AE_encoder, decoder = AE_decoder)
                    optimizer_AE = torch.optim.Adam(AE.parameters(), lr = args.lr, weight_decay = args.L2norm)
                    scheduler_AE = torch.optim.lr_scheduler.StepLR(optimizer_AE, step_size = 1, gamma = args.lr_scheduler)
                    AE.float()
                    results = train_AE(data.x,
                                       model = AE,
                                       optimizer =  optimizer_AE,
                                       epochs = args.epochs,
                                       true_labels = true_labels,
                                       P = P,
                                       P_rate = prate,
                                       scheduler = scheduler_AE,
                                       return_dataframe = True)
                    results['model'] = 'AE'
                    df = pd.concat([df, results])
                df.to_csv(f'results/dataframe_{args.name}_results.csv',index = False)
                    
    
    if args.training_type == 'pontual':
        true_labels = np.array([1 if x == 3 else 0 for x in data.y])
        all_positives = np.array([x for x in range(data.x.shape[0]) if true_labels[x] == 1])
        in_channels = data.x.shape[0]
        # Arrumar isso pro args
        hidden_channels = 64
        out_channels = 16

        P = random.sample(all_positives.tolist(), int(args.P_rate * len(all_positives)))

        if "RGCN" in args.model:
            RGCN_encoder = RGCN_model(data.x.shape[1], hidden_channels, out_channels, L = args.L)
            RGCN_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
            GAE_RGCN = GAE(encoder = RGCN_encoder, decoder = RGCN_decoder)
            optimizer_RGCN = torch.optim.Adam(GAE_RGCN.parameters(), lr = args.lr, weight_decay = args.L2norm)
            scheduler_RGCN = torch.optim.lr_scheduler.StepLR(optimizer_RGCN, step_size = 1, gamma = args.lr_scheduler)
            GAE_RGCN.float()

            results = train(graph = data,
                            model = GAE_RGCN,
                            optimizer=optimizer_RGCN,
                            epochs = args.epochs,
                            P = P,
                            P_rate = args.P_rate,
                            true_labels=true_labels,
                            L = args.L,
                            scheduler=scheduler_RGCN,
                            rewiring_usage = True,
                            return_dataframe=False)
            
            print('RGCN', results['acc_per_epoch'][-1], round(results['f1_per_epoch'][-1], 4), max(results['acc_per_epoch']), round(max(results['f1_per_epoch']), 4))

        if "GCN" in args.model:
            GCN_encoder = GCN_model(data.x.shape[1], hidden_channels, out_channels)
            GCN_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
            GAE_GCN = GAE(encoder = GCN_encoder, decoder = GCN_decoder)
            optimizer_GCN = torch.optim.Adam(GAE_GCN.parameters(), lr = args.lr, weight_decay = args.L2norm)
            scheduler_GCN = torch.optim.lr_scheduler.StepLR(optimizer_GCN, step_size = 1, gamma = args.lr_scheduler)
            GAE_GCN.float()

            results = train(
                graph = data,
                model = GAE_GCN,
                optimizer = optimizer_GCN,
                epochs = args.epochs,
                P = P,
                P_rate = args.P_rate,
                true_labels = true_labels,
                rewiring_usage = False,
                scheduler = scheduler_GCN,
                return_dataframe = False)
            
            print('GCN', results['acc_per_epoch'][-1], round(results['f1_per_epoch'][-1], 4), max(results['acc_per_epoch']), round(max(results['f1_per_epoch']), 4))
        
        if 'RANDOM' in args.model:
            negatives = random.sample(list(range(data.x.shape[0])), 200)
            acc, f1 = evaluate_model(negatives, true_labels)
            print('random', acc, f1)

