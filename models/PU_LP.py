import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph


class PU_LP:
    def __init__(self, data_graph, positives, unlabeled, alpha=0.1, m = 3, l = 1):
        self.positives = positives
        self.data_graph = data_graph
        self.unlabeled = unlabeled
        self.alpha = alpha
        self.m = m
        self.l = l

    # def train(self):
    #     pul_mask = torch.tensor([1 if x in self.positives else 0 for x in range(len(self.data))])
    #     if not self.precomputed_graph:
    #         self.A = kneighbors_graph(self.data, 3, mode='connectivity', metric='minkowski', include_self=False).todense()
    #     eigenvalues, eigenvectors = np.linalg.eig(self.A)
    #     largest_eigenvalue = np.max(eigenvalues)
    #     if self.alpha < largest_eigenvalue:
    #         W = torch.inverse(torch.eye(len(self.A)) - self.alpha * self.A) - torch.eye(len(self.A))
    #     else:
    #         print(self.alpha, largest_eigenvalue)
    #         raise Exception('alpha > largest_eigenvalue')
        
    #     self.RP = list()
    #     positives_ = self.positives.copy()
    #     unlabeled_ = self.unlabeled.copy()
    #     S = list()

    #     for k in range(self.m):
    #         for v_i in unlabeled_:
    #             #Sv_i = torch.sum(W[v_i][pul_mask]) / len(positives_)
    #             Sv_i = torch.mean(W[v_i][pul_mask])
    #             S.append(Sv_i.item())
    #         _, RP_ = zip(*sorted(zip(S, unlabeled_), reverse=True))
    #         RP_ = list(RP_)[:int(self.l / self.m)]
    #         positives_ = positives_ + RP_
    #         unlabeled_ = [element for element in unlabeled_ if element not in RP_]
    #         self.RP = self.RP + RP_
        
    #     P_RP_mask = [1 if x in self.positives + self.RP else 0 for x in range(len(self.data))]
    #     S = list()
    #     for v_i in [element for element in self.unlabeled if element not in self.RP]:
    #         #Sv_i = torch.sum(W[v_i][P_RP_mask]) / len(positives_)
    #         Sv_i = torch.mean(W[v_i][P_RP_mask])
    #         S.append(Sv_i.item())
    #     _, self.RN = zip(*sorted(zip(S, [element for element in self.unlabeled if element not in self.RP]), reverse=True))
    #     self.RN = list(self.RN)
        
    def train(self):
        self.RP = list()
        P_line = self.positives.copy()
        U_line = self.unlabeled.copy()

        # Calculando W = (I - alpha*A)**-1 - I
        I = np.eye(len(self.data_graph))
        W = np.linalg.inv(I - self.alpha * self.data_graph) - I

        for k in range(self.m):
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
        RP_line = rank_dict[:int((self.l / self.m) * len(self.positives))]
        P_line = P_line + RP_line
        U_line = list(set(U_line) - set(RP_line))
        self.RP = self.RP + RP_line

        rank_dict = dict()
        for vi in list(set(self.unlabeled) - set(RP_line)):
            S_vi = 0
            for vj in self.positives + self.RP:
                S_vi += W[vi, vj]
            S_vi /= len(P_line)
            rank_dict[vi] = S_vi
        rank_dict = sorted(rank_dict.items(), key=lambda x:x[1])
        rank_dict = [tupla[0] for tupla in rank_dict]
        self.RN = rank_dict[:len(self.positives + self.RP)]

    def negative_inference(self, num_neg = None):
        #return self.RN[-len(self.positives + self.RP):][:num_neg]
        if not num_neg:
            num_neg = len(self.RN) + len(self.RP)
        return self.RN[-num_neg:]
    
    
    def positive_inference(self):
        return self.RP
