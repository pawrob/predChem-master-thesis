import warnings

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GATv2Conv as GAT, global_mean_pool as gap

warnings.filterwarnings("ignore")


class Embedder(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, layers=2, type='SAGE', is_bias=True):
        super(Embedder, self).__init__()
        torch.manual_seed(torch.initial_seed())
        self.layers = layers
        if type == 'SAGE':
            self.in_layer = SAGEConv(input_dim, 64, bias=is_bias)
            self.hidden_layer = SAGEConv(64, 64, bias=is_bias)
            self.out_layer = SAGEConv(64, embed_dim, bias=is_bias)
            self.in_out_layer = SAGEConv(input_dim, embed_dim, bias=is_bias)
        elif type == 'GAT':
            self.in_layer = GAT(input_dim, 64, heads=1, bias=is_bias)
            self.hidden_layer = GAT(64, 64, heads=1, bias=is_bias)
            self.out_layer = GAT(64, embed_dim, heads=1, bias=is_bias)
            self.in_out_layer = GAT(input_dim, embed_dim, heads=1, bias=is_bias)

    def forward(self, x, edge_index, batch_index):
        if self.layers == 1:
            x = self.in_out_layer(x, edge_index)
        elif self.layers == 2:
            x = self.in_layer(x, edge_index)
            x = F.relu(x)
            x = self.out_layer(x, edge_index)
        else:
            x = self.in_layer(x, edge_index)
            x = F.relu(x)
            for i in range(self.layers - 2):
                x = self.hidden_layer(x, edge_index)
            x = F.relu(x)
            x = self.out_layer(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = gap(x, batch_index)
        return x


class Classifier(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, layers=3, is_bias=True):
        super(Classifier, self).__init__()
        torch.manual_seed(123)

        self.embedder = Embedder(input_dim, embed_dim, layers, is_bias=is_bias)
        self.lin = Linear(embed_dim, output_dim)

    def forward(self, x, edge_index, batch_index):
        e = self.embedder(x, edge_index, batch_index)
        x = self.lin(e)
        x = F.sigmoid(x)
        return x, e
