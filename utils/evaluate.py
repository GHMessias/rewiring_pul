from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import torch.nn.functional as F

def negative_inference(model, rewiring_graphs, num_neg, weights = None):
    inference_dict = dict()
    
    if isinstance(rewiring_graphs, list):
        model.eval()
        H_L = model.encode(rewiring_graphs[0].x, rewiring_graphs)
        out = model.decode(H_L)
        for element in range(rewiring_graphs[0].x.shape[0]):
            inference_dict[element] = F.binary_cross_entropy(out[element], rewiring_graphs[0].x[element], weight=weights).item()
    else:
        model.eval()
        H_L = model.encode(rewiring_graphs.x.float(), rewiring_graphs.edge_index)
        out = model.decode(H_L)
        for element in range(rewiring_graphs.x.shape[0]):
            inference_dict[element] = F.binary_cross_entropy(out[element], rewiring_graphs.x[element], weight=weights).item()
    
    dicionario_ordenado = dict(sorted(inference_dict.items(), key=lambda item: item[1]))
    # print(dicionario_ordenado)

    return list(dicionario_ordenado.keys())[:num_neg]

def evaluate_model(negative, true_labels):
    y_pred = np.zeros_like(negative)
    y_true = np.array([true_labels[x] for x in negative])

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, pos_label = 0)