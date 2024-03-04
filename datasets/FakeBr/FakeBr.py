from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import torch
import numpy as np
from torch_cluster import knn_graph
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os.path as osp
import os

class FakeBr(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['pre-processed.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        print('processing FakeBr dataset')

        df = pd.read_csv(self.raw_paths[0])

        corpus = []

        for index, row in df.iterrows():
            corpus.append(
                TaggedDocument(row['preprocessed_news'], tags = [f'doc{index}'])
                )

        model = Doc2Vec(documents=corpus, 
                        vector_size = 500, 
                        window=8, 
                        epochs = 20, 
                        alpha = 0.025, 
                        min_alpha = 0.0001, 
                        min_count = 1)

        # Salvar os vetores em um formato PyTorch
        vectors = [model.docvecs[tag] for tag in model.docvecs.index_to_key]

        # Converter os vetores para tensores do PyTorch
        X = torch.tensor(np.array(vectors))
        y = torch.tensor([1 if x == 'fake' else -1 for x in df.label])
        edge_index = knn_graph(X, 3)

        # Shuffling
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]

        data = Data(x = X, edge_index=edge_index, y = y)
        torch.save(data, osp.join(self.processed_dir, f'data.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self):
        data = torch.load(osp.join(self.processed_dir, f'data.pt'))
        return data

# dataset = FakeBr(root = 'datasets/FakeBr')
# dataset.process()