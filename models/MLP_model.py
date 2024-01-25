import torch.nn
import torch.nn.functional as F

class MLP_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP_model, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.sigmoid(x)