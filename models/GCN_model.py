import torch.nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_model, self).__init__()
        self.layer1 = GCNConv(in_channels, hidden_channels)
        self.layer2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        x = self.layer2(x, edge_index)

        return x