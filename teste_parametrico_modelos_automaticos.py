import argparse
from models.RGCN_model import RGCN_model
from models.GCN_model import GCN_model
from models.MLP_model import MLP_model
from models.CCRNE import CCRNE
from models.MCLS import MCLS
from models.LP_PUL import LP_PUL
from models.PU_LP import PU_LP
from models.RCSVM_RN import RCSVM_RN
from datasets.FactCheckedNews.FactCheckedNews import FactCheckedNews
from datasets.FakeBr.FakeBr import FakeBr
from datasets.FakeNewsNet.FakeNewsNet import FakeNewsNet
import pandas as pd
import networkx as nx
import random
import numpy as np
from utils.evaluate import negative_inference, evaluate_model
from torch_geometric.transforms import NormalizeFeatures

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str)
parser.add_argument('--model', type = str)
parser.add_argument('--P_rate_range', type=float, nargs='+', default = [n/100 for n in range(1,26)])
parser.add_argument('--ccrne_ratio', type = float, nargs = '+', default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
parser.add_argument('--positive_class', type = int, default=1)

args = parser.parse_args()

if __name__ == '__main__':

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

    for prate in args.P_rate_range:
        P = random.sample(all_positives.tolist(), int(prate * len(all_positives)))
        U = np.array([x for x in range(data.x.shape[0]) if x not in P])

        if "CCRNE" in args.model:
            for ratio in args.ccrne_ratio:
                ccrne_classifier = CCRNE(data.x, P, U.tolist(), ratio = ratio)
                ccrne_classifier.train()
                negatives = ccrne_classifier.negative_inference(len(data.y))
                print(len(negatives))
                acc, f1 = evaluate_model(negatives, true_labels)
                results = pd.DataFrame({'model': ['CCRNE'], 'acc': [acc], 'f1': [f1], 'P': prate, 'neg': len(negatives), 'ratio': ratio})
                df = pd.concat([df, results])

        if "RCSVM" in args.model:
            for alpha in [0.2, 0.4, 0.6, 0.8]:
                beta = 1 - alpha
                rcsvm_classifier = RCSVM_RN(data.x, P, U, alpha = alpha, beta = beta)
                rcsvm_classifier.train()
                negatives = rcsvm_classifier.negative_inference(len(data.y))
                # print(negatives)
                acc, f1 = evaluate_model(negatives, true_labels)
                results = pd.DataFrame({'model': ['RCSVM'], 'acc': [acc], 'f1': [f1], 'P': prate, 'neg': len(negatives), 'alpha':alpha, 'beta': beta})
                df = pd.concat([df, results])

        if "PULP" in args.model:
            for alpha in [0.02, 0.04, 0.06, 0.08, 0.1]:
                for l in [0.2, 0.6, 0.8, 1, 1.2, 1.4, 2]:
                    for m in [1,2,3]:
                        print(f'computing alpha {alpha} l {l} m {m}')
                        pulp_classifier = PU_LP(data.x, P, U, alpha=alpha, m = m, l = l)
                        pulp_classifier.train()
                        negatives = pulp_classifier.negative_inference(len(data.y))
                        # print(negatives)
                        acc, f1 = evaluate_model(negatives, true_labels)
                        results = pd.DataFrame({'model': ['PU_LP'], 'acc': [acc], 'f1': [f1], 'P':prate, 'neg': len(negatives), 'alpha':alpha, 'l': l, 'm': m})
                        df = pd.concat([df, results])

        df.to_csv(f'df_parametrico_{args.model}_{args.dataset}.csv')