'''
Arquivo com as funções de geração de embeddings
'''
from node2vec import Node2Vec
import torch

def GAE_embeddings(gae_model, gae_optim, data, epochs = 500):
    for e in range(epochs):
        gae_optim.zero_grad()
        H_L = gae_model.encode(data.x.float(), data.edge_index)
        loss = gae_model.recon_loss(H_L, data.edge_index)
        print(f'epoch: {e + 1} | loss: {loss.item()}', end = '\r')
        loss.backward()
        gae_optim.step()
    return gae_model.encode(data.x.float(), data.edge_index)                   

def rGAE_embeddings(rgae_model, rgae_optim, graph_list, epochs = 500):
    for e in range(epochs):
        rgae_optim.zero_grad()
        H_L = rgae_model.encode(graph_list[0].x.float(), graph_list)
        loss = rgae_model.recon_loss(H_L, graph_list[-1].edge_index)
        print(f'epoch: {e + 1} | loss: {loss.item()}', end = '\r')
        loss.backward()
        rgae_optim.step()
    return rgae_model.encode(graph_list[0].x.float(), graph_list)

def node2vec_embeddings(G, dimensions=64, walk_length=30, num_walks=200, workers=4):
    # Criar um objeto Node2Vec com parâmetros desejados
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)

    # Treinar o modelo (pode demorar algum tempo)
    model = node2vec.fit(window=10, min_count=1)

    return torch.tensor(model.wv.vectors)