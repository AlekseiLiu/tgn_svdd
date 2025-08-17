#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import torch
import torch.nn as nn

from torch.nn import Linear

from torch_geometric.nn import TransformerConv




# implementation form PyG examples
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 8, heads=8,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)
        
        
class GraphAttentionEmbeddingVanilla(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

    
    
    
# My implementation
class RegressionModel(nn.Module):
    def __init__(self, dim_n, hidden_layers=1, hidden_units=32):
        super().__init__()
        self.fc1 = nn.Linear(dim_n, hidden_units)
        self.relu = nn.ReLU()
        self.hidden_layers = hidden_layers
        if hidden_layers > 1:
            self.hidden = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for i in range(hidden_layers - 1)])
        self.fc2 = nn.Linear(hidden_units, dim_n)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        for i in range(self.hidden_layers - 1):
            x = self.hidden[i](x)
            x = self.relu(x)
        x = self.fc2(x)
        return x





class DeepSVDD(nn.Module):
    def __init__(self, output_shape):
        super(DeepSVDD, self).__init__()
        self.c = nn.Parameter(torch.randn(output_shape), requires_grad=True)

    def forward(self, z):
        distance = torch.sum((z - self.c) ** 2, dim=1)
        return distance



class DimRedMlp(nn.Module):
    def __init__(self, dim_in, dim_out, factor_dim_in=0.7):
        super(DimRedMlp, self).__init__()
        
        self.fc1 = nn.Linear(dim_in, int(dim_in*factor_dim_in))  # First fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(int(dim_in*factor_dim_in), dim_out)  # Second fully connected layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LinearRed(nn.Module):
    def __init__(self, dim_1, dim_2):
        super(LinearRed, self).__init__()
        self.linear = nn.Linear(dim_1, dim_2)

    def forward(self, x):
        x = self.linear(x)
        return x

















