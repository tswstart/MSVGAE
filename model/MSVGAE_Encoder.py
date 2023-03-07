# -*- coding: utf-8 -*-
# @Author  : sw t
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GAT_Encoder(nn.Module):
    def __init__(self, num_heads, in_channels, hidden_dims, latent_dim, dropout):
        super(GAT_Encoder, self).__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # initialize GAT layer
        self.hidden_layer_1 = GATConv(
            in_channels=in_channels, out_channels=hidden_dims[0],
            heads=self.num_heads['first'],
            dropout=dropout[0],
            concat=True)
        in_dim2 = hidden_dims[0] * self.num_heads['first'] + in_channels * 2
        # in_dim2 = hidden_dims[0] * self.num_heads['first']

        self.hidden_layer_2 = GATConv(
            in_channels=in_dim2, out_channels=hidden_dims[1],
            heads=self.num_heads['second'],
            dropout=dropout[1],
            concat=True)

        in_dim_final = hidden_dims[-1] * self.num_heads['second'] + in_channels
        # in_dim_final = hidden_dims[-1] * self.num_heads['second']

        self.out_mean_layer = GATConv(in_channels=in_dim_final, out_channels=self.latent_dim,
                                      heads=self.num_heads['mean'], concat=False, dropout=0.2)
        self.out_logstd_layer = GATConv(in_channels=in_dim_final, out_channels=self.latent_dim,
                                        heads=self.num_heads['std'], concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        hidden_out1 = self.hidden_layer_1(x, edge_index)
        hidden_out1 = F.relu(hidden_out1)
        # add Gaussian noise being the same shape as x and concat
        hidden_out1 = torch.cat([x, torch.randn_like(x), hidden_out1], dim=1)
        hidden_out2 = self.hidden_layer_2(hidden_out1, edge_index)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        last_out = hidden_out2
        # concat x with last_out
        last_out = torch.cat([x, last_out], dim=1)
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)

        return z_mean, z_logstd


class GCN_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super(GCN_Encoder, self).__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        # initialize GCN layer
        self.hidden_layer_1 = GCNConv(in_channels=in_channels, out_channels=hidden_dims[0])
        self.hidden_layer_2 = GCNConv(in_channels=hidden_dims[0] + 2 * in_channels, out_channels=hidden_dims[1])
        self.out_mean_layer = GCNConv(in_channels=hidden_dims[-1] + in_channels, out_channels=latent_dim)
        self.out_logstd_layer = GCNConv(in_channels=hidden_dims[-1] + in_channels, out_channels=latent_dim)

    def forward(self, x, edge_index):
        hidden_out1 = self.hidden_layer_1(x, edge_index)
        hidden_out1 = F.relu(hidden_out1)
        # add Gaussian noise being the same shape as x and concat
        hidden_out1 = torch.cat([x, torch.randn_like(x), hidden_out1], dim=1)
        hidden_out2 = self.hidden_layer_2(hidden_out1, edge_index)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        last_out = hidden_out2
        # concat x with last_out
        last_out = torch.cat([x, last_out], dim=1)
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)

        return z_mean, z_logstd