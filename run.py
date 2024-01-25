import argparse
from utils.rewiring import rewiring
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from graph_generators.generator import generator
from models.RGCN_model import RGCN_model
from models.GCN_model import GCN_model
from models.MLP_model import MLP_model
import torch
from torch_geometric.nn import GAE
from utils.training import train

parser = argparse.ArgumentParser()

# Setting variables for generator
parser.add_argument('--generator', action='store_true', help = 'generating data using rewiring')
parser.add_argument("--name", type = str, help = 'Rewiring Graph')
parser.add_argument("--samples", type = int, default = 1)

# Setting variables for graph generator
parser.add_argument('--ro_range', type = str, default = "0.2 0.4 0.6")
parser.add_argument('--L_range', type = str, default = "2 3 4 5")
parser.add_argument('--P_rate_range', type=str, default = "0.01 0.02 0.03 0.04 0.05 0.1 0.15 0.2 0.25")

# Setting variables for training
parser.add_argument('--training', action = 'store_true', help = 'starting train the model')
parser.add_argument('--model', type = str, default = "RGCN, GCN, random") #RGCN, GCN
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--lr', type = float, default=1e-6)
parser.add_argument('--L2norm', type = float, default=1e-7)
parser.add_argument('--L', type = int, default=3)
parser.add_argument('--ro', type = float, default=0.2)
parser.add_argument('--P_rate', type = float, default=0.05)
args = parser.parse_args()


if __name__ == '__main__':

    dataset = Planetoid(root = 'datasets', name = args.name)
    data = dataset[0]

    if args.generator:
        L_range = [int(x) for x in args.L_range.split()]
        ro_range = [float(x) for x in args.ro_range.split()]
        P_rate_range = [float(x) for x in args.P_rate_range.split()]

        generator(data, args.samples, L_range, P_rate_range, ro_range)
    
    if args.training:
        models = list()
        in_channels = data.x.shape[0]
        hidden_channels = 64
        out_channels = 16
        model_losses = dict()
        
        # if 'RGCN' in args.model:
        #     RGCN_encoder = RGCN_model(data.x.shape[1], hidden_channels, out_channels, L = args.L)
        #     RGCN_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
        #     GAE_RGCN = GAE(encoder = RGCN_encoder, decoder = RGCN_decoder)
        #     optimizer_RGCN = torch.optim.Adam(GAE_RGCN.parameters(), lr = args.lr, weight_decay = args.L2norm)
        #     GAE_RGCN.float()

        #     losses_RGCN = train(data, GAE_RGCN, optimizer_RGCN, args.epochs, args.L, args.ro, args.P_rate, args.samples, rewiring=True)
        #     model_losses['loss RGCN'] = losses_RGCN

        if 'GCN' in args.model:
            GCN_encoder = GCN_model(data.x.shape[1], hidden_channels, out_channels)
            GCN_decoder = MLP_model(out_channels, hidden_channels, data.x.shape[1])
            GAE_GCN = GAE(encoder = GCN_encoder, decoder = GCN_decoder)
            optimizer_GCN = torch.optim.Adam(GAE_GCN.parameters(), lr = args.lr, weight_decay = args.L2norm)
            
            losses_GCN = train(data, GAE_GCN, optimizer_GCN, args.epochs, args.L, args.ro, args.P_rate, args.samples, rewiring=False)
            model_losses['loss GCN'] = losses_GCN


        
        


        
        

