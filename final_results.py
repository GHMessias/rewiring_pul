'''
Arquivo responsável por fazer os treinamentos em duas etapas de todos os modelos:

Algoritmos testados na fase final:
OCSVM OK
LP_PUL
PU_LP
CCRNE + SVM
Distance Mean OK
nnPU
Bagging-Based PU_learning (https://github.com/pulearn/pulearn)

Algoritmos de Embedding:
GAE OK
rGAE OK
Node2Vec OK
No embedding OK
'''

import argparse
import pandas as pd
import numpy as np
from torch_geometric.utils import from_networkx, to_networkx
from utils.embedding_generators import *
from torch_geometric.nn import GAE
from models.RGCN_model import RGCN_model
from models.GCN_model import GCN_model
from sklearn import svm
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
from node2vec import Node2Vec
import random
from utils.rewiring import *
# Temporary libs
from sklearn.metrics import accuracy_score, f1_score
from models.LP_PUL import LP_PUL
from models.PU_LP import PU_LP
from networkx.algorithms import node_classification
from datasets.FakeBr.FakeBr import FakeBr
from datasets.FactCheckedNews.FactCheckedNews import FactCheckedNews
from datasets. FakeNewsNet.FakeNewsNet import FakeNewsNet
from pulearn import WeightedElkanotoPuClassifier
from sklearn.svm import SVC

parser = argparse.ArgumentParser()

parser.add_argument("--name", type = str, default='')
parser.add_argument('--dataset', type = str, default = 'Planetoid')
parser.add_argument('--positive_class', type = int)
parser.add_argument('--samples', type = int, default = 5)
parser.add_argument('--hid_channels', type = int, default = 64)
parser.add_argument('--out_channels', type = int, default = 16)
parser.add_argument('--P_rate_range', type=float, nargs='+', default = [0.05, 0.1, 0.15, 0.2, 0.25])
parser.add_argument('--model', type=str, default = 'OCSVM LP_PUL RLP_PUL PU_LP RPU_LP Distance_Mean') #args.model = 'OCSVM LP_PUL RLP_PUL'

args = parser.parse_args()

models = args.model.split()
print(models)

if args.dataset == 'Planetoid':
    dataset = Planetoid(root = 'datasets', name = args.name, transform = NormalizeFeatures())
    data = dataset[0]

if args.dataset == 'FakeBr':
    dataset = FakeBr(root = 'datasets/FakeBr', transform=NormalizeFeatures())
    data = dataset.get()
    print(data)

if args.dataset == 'FakeNewsNet':
    dataset = FakeNewsNet(root = 'datasets/FakeNewsNet', transform=NormalizeFeatures())
    data = dataset.get()
    print(data)

if args.dataset == 'FactCheckedNews':
    dataset = FactCheckedNews(root = 'datasets/FactCheckedNews', transform=NormalizeFeatures())
    data = dataset.get()
    print(data)

true_labels = np.array([1 if x == args.positive_class else 0 for x in data.y])
all_positives = np.array([x for x in range(data.x.shape[0]) if true_labels[x] == 1])

df = pd.DataFrame()

for sample in range(args.samples):
    print(f'################ SAMPLE {sample} ################ \n')

    # Gerando os embeddings não supervisionados

    # GAE Embedding
    GCN_encoder = GCN_model(data.x.shape[1], args.hid_channels, args.out_channels)
    GAE_GCN = GAE(encoder = GCN_encoder)
    optimizer_GAE = torch.optim.Adam(GAE_GCN.parameters(), lr = 0.001)
    GAE_GCN.float()
    X_gae = GAE_embeddings(gae_model = GAE_GCN, gae_optim = optimizer_GAE, data = data)

    # # Node2Vec
    X_node2vec = node2vec_embeddings(to_networkx(data))

    for prate in args.P_rate_range:
        P = random.sample(all_positives.tolist(), int(prate * len(all_positives)))
        U = np.array([x for x in range(data.x.shape[0]) if x not in P])

        # Gerando os elementos necessários para todos os embeddings
        graph_list = rewiring(to_networkx(data, to_undirected=True), 3, P)
        graph_list = [data] + [from_networkx(G) for G in graph_list]

        # Gerando os embeddings semi-supervisionados

        # rGAE Embedding
        RGCN_encoder = RGCN_model(data.x.shape[1], args.hid_channels, args.out_channels, L = 3)
        GAE_RGCN = GAE(encoder = RGCN_encoder)
        optimizer_RGAE = torch.optim.Adam(GAE_RGCN.parameters(), lr = 0.001)
        GAE_RGCN.float()
        X_rgae = rGAE_embeddings(rgae_model=GAE_RGCN, rgae_optim=optimizer_RGAE, graph_list=graph_list)
        print(X_rgae.shape)

        embeddings = {'None': data.x, 'GAE': X_gae, 'node2vec': X_node2vec, 'RGAE': X_rgae}
        # Treinando os modelos

        # OCSVM
        if 'OCSVM' in models:
            # if 'OCSVM' in args.model:
            ocsvm = svm.OneClassSVM(kernel = 'rbf', nu = 0.1)
            for name, embedding in embeddings.items():
                ocsvm.fit(embedding[P].detach().numpy())
                classification = [0 if x == -1 else 1 for x in ocsvm.predict(embedding[U].detach().numpy())] # 0 is negative_class 1 is positive_class
                results = pd.DataFrame({
                    'model': ['ocsvm'],
                    'embedding': [name],
                    'sample': [sample],
                    'f1_score': [f1_score(true_labels[U], classification, pos_label = 1)],
                    'acc_score': [accuracy_score(true_labels[U], classification)],
                    'P': [prate]
                })
                df = pd.concat([df, results])
        
        # LP_PUL
        if 'LP_PUL' in models:
            G = to_networkx(data, to_undirected=True)
            lp_pul_classifier = LP_PUL(G, data.x, P, U)
            lp_pul_classifier.train()
            N = lp_pul_classifier.negative_inference(len(P))
            for element in P:
                G.nodes[element]['label'] = 1
            for element in N:
                G.nodes[element]['label'] = 0
            classification = node_classification.local_and_global_consistency(G)
            results = pd.DataFrame({
                    'model': ['LP_PUL'],
                    'embedding': ['None'],
                    'sample': [sample],
                    'f1_score': [f1_score(np.array(true_labels)[U], np.array(classification)[U], pos_label = 1)],
                    'acc_score': [accuracy_score(np.array(true_labels)[U], np.array(classification)[U])],
                    'P': [prate]
                })
            df = pd.concat([df, results])

        if 'RLP_PUL' in models:
        # RLP_PUL
            G = to_networkx(graph_list[-1], to_undirected=True)
            lp_pul_classifier = LP_PUL(G, data.x, P, U)
            lp_pul_classifier.train()
            N = lp_pul_classifier.negative_inference(len(P))
            for element in P:
                G.nodes[element]['label'] = 1
            for element in N:
                G.nodes[element]['label'] = 0
            classification = node_classification.local_and_global_consistency(G)
            results = pd.DataFrame({
                    'model': ['RLP_PUL'],
                    'embedding': ['None'],
                    'sample': [sample],
                    'f1_score': [f1_score(np.array(true_labels)[U], np.array(classification)[U], pos_label = 1)],
                    'acc_score': [accuracy_score(np.array(true_labels)[U], np.array(classification)[U])],
                    'P': [prate]
                })
            df = pd.concat([df, results])

        #PU_LP
        if 'PU_LP' in models:
            G = to_networkx(data, to_undirected=True)
            pu_lp_classifier = PU_LP(data = data.x, positives = P, unlabeled = U, precomputed_graph = True, A = nx.to_numpy_matrix(G))
            pu_lp_classifier.train()
            N = pu_lp_classifier.negative_inference(len(P))
            for element in pu_lp_classifier.RP:
                G.nodes[element]['label'] = 1
            for element in N:
                G.nodes[element]['label'] = 0
            classification = node_classification.local_and_global_consistency(G)
            results = pd.DataFrame({
                    'model': ['PU_LP'],
                    'embedding': ['None'],
                    'sample': [sample],
                    'f1_score': [f1_score(np.array(true_labels)[U], np.array(classification)[U], pos_label = 1)],
                    'acc_score': [accuracy_score(np.array(true_labels)[U], np.array(classification)[U])],
                    'P': [prate]
                })
            df = pd.concat([df, results])

        #RPU_LP
        if 'RPU_LP' in models:
            G = to_networkx(graph_list[-1], to_undirected=True)
            pu_lp_classifier = PU_LP(data = data.x, positives = P, unlabeled = U, precomputed_graph = True, A = nx.to_numpy_matrix(G))
            pu_lp_classifier.train()
            N = pu_lp_classifier.negative_inference(len(P))
            for element in pu_lp_classifier.RP:
                G.nodes[element]['label'] = 1
            for element in N:
                G.nodes[element]['label'] = 0
            classification = node_classification.local_and_global_consistency(G)
            results = pd.DataFrame({
                    'model': ['RPU_LP'],
                    'embedding': ['None'],
                    'sample': [sample],
                    'f1_score': [f1_score(np.array(true_labels)[U], np.array(classification)[U], pos_label = 1)],
                    'acc_score': [accuracy_score(np.array(true_labels)[U], np.array(classification)[U])],
                    'P': [prate]
                })
            df = pd.concat([df, results])

        # #Elkanoto
        # if args.model == 'Elkanoto':
        #     svc = SVC(C = 10, kernel = 'rbf', gamma = 0.4, probability=True)
        #     pu_estimator = WeightedElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.8, labeled = len(P), unlabeled = len(U))
        #     for name, embedding in embeddings.items():
        #         pu_estimator.fit(embedding[P].detach().numpy(), true_labels[P])
        #         classification = [0 if x == -1 else 1 for x in pu_estimator.predict(embedding[U].detach().numpy())] # 0 is negative_class 1 is positive_class
        #         results = pd.DataFrame({
        #             'model': ['Elkanoto'],
        #             'embedding': [name],
        #             'sample': [sample],
        #             'f1_score': [f1_score(true_labels[U], classification, pos_label = 1)],
        #             'acc_score': [accuracy_score(true_labels[U], classification)]
        #         })
        #         df = pd.concat([df, results])

        #Distance_Mean
        if 'Distance_Mean' in models:
            for name, embedding in embeddings.items():
                inference_dict = dict()
                classification = torch.zeros_like(data.y)
                classification[P] = 1
                for element in U:
                    dist = torch.cdist(embedding[element].unsqueeze(0), embedding[P])
                    value = dist.mean()
                    inference_dict[element] = value
                all_distances = dict(sorted(inference_dict.items(),reverse=True, key=lambda item: item[1]))
                N = list(all_distances.keys())[:len(P)]
                # new unlabeleds
                RU = torch.tensor([x for x in U if x not in N])
                for element in RU:
                    if torch.mean(torch.cdist(embedding[element].unsqueeze(0), embedding[P])) <= torch.mean(torch.cdist(embedding[element].unsqueeze(0), embedding[N])):
                        classification[element] = 1
                results = pd.DataFrame({
                    'model': ['Distance Mean'],
                    'embedding': [name],
                    'sample': [sample],
                    'f1_score': [f1_score(np.array(true_labels)[U], np.array(classification)[U], pos_label = 1)],
                    'acc_score': [accuracy_score(np.array(true_labels)[U], np.array(classification)[U])],
                    'P': [prate]
                })
                df = pd.concat([df, results])
                print(df)

        df.to_csv(f'results/results_{args.dataset}_{args.name}.csv')


        


