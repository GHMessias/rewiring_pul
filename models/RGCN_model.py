import torch.nn
from torch_geometric.nn import GCNConv, GAE
import torch.nn.functional as F

class RGCN_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, L):
        super(RGCN_model, self).__init__()
        self.L = L
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.parameter_list = torch.nn.ParameterList()

        # Criando o primeiro layer de rewiring
        self.rewiring_gcn1 = list()
        for i in range(L):
            sublayer1 = GCNConv(self.in_channels, self.hidden_channels)
            # torch.nn.init.xavier_uniform(sublayer1.weight)
            self.rewiring_gcn1.append(sublayer1)
            for index, param in enumerate(self.rewiring_gcn1[i].parameters()):
                if index == 1:
                    self.parameter_list.append(torch.nn.init.xavier_uniform_(param))
                else:
                    self.parameter_list.append(param)


        # Criando a segunda layer de rewiring
        self.rewiring_gcn2 = list()
        for i in range(self.L):
            sublayer2 = GCNConv(self.hidden_channels, self.out_channels)
            # torch.nn.init.xavier_uniform(sublayer2.weight)
            self.rewiring_gcn2.append(sublayer2)
            for index, param in enumerate(self.rewiring_gcn2[i].parameters()):
                if index == 1:
                    self.parameter_list.append(torch.nn.init.xavier_uniform_(param))
                else:
                    self.parameter_list.append(param)

    def forward(self, x, rewiring_graph_list):
        # Forward da primeira camada
        output_tensor_1 = torch.zeros((x.shape[0], self.hidden_channels))
        for i in range(self.L):
            # print((rewiring_graph_list[i]))
            _x = self.rewiring_gcn1[i](x, rewiring_graph_list[i].edge_index)
            output_tensor_1 += _x
        output_tensor_1 = F.relu(output_tensor_1)

        # Forward da segunda camada
        output_tensor_2 = torch.zeros((output_tensor_1.shape[0], self.out_channels))
        for i in range(self.L):
            _x = self.rewiring_gcn2[i](output_tensor_1, rewiring_graph_list[i].edge_index)
            output_tensor_2 += _x

        return output_tensor_2
