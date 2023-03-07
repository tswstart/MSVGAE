# -*- coding: utf-8 -*-
# @Author  : sw t
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)

EPS = 1e-15
MAX_LOGSTD = 10


# based on torch.nn.module class in torch

class MSVGAE(torch.nn.Module):
    def __init__(self, encoder_gat1, encoder_gat2, line_decoder_hid_dim=128):
        super(MSVGAE, self).__init__()

        # initialize parameter
        self.mu_gat2 = self.logstd_gat2 = None
        self.mu_gat1 = self.logstd_gat1 = None
        # encoder
        self.encoder_gat1 = encoder_gat1
        self.encoder_gat2 = encoder_gat2
        # use inner product decoder by default
        self.decoder = InnerProductDecoder()
        # liner decoder
        self.liner_decoder = Sequential(
            Linear(in_features=self.encoder_gat1.latent_dim * 2, out_features=line_decoder_hid_dim),
            BatchNorm1d(line_decoder_hid_dim),
            ReLU(),
            Dropout(0.4),
            Linear(in_features=line_decoder_hid_dim, out_features=self.encoder_gat1.in_channels),
        )

    def encode(self, *args, **kwargs):
        """ encode """
        # GAT encoder
        self.mu_gat2, self.logstd_gat2 = self.encoder_gat2(*args, **kwargs)
        # GCN encoder
        self.mu_gat1, self.logstd_gat1 = self.encoder_gat1(*args, **kwargs)
        # fix range
        self.logstd_gat2 = self.logstd_gat2.clamp(max=MAX_LOGSTD)
        self.logstd_gat1 = self.logstd_gat1.clamp(max=MAX_LOGSTD)
        # reparameter
        z_gat2 = self.reparametrize(self.mu_gat2, self.logstd_gat2)
        z_gat1 = self.reparametrize(self.mu_gat1, self.logstd_gat1)
        z = torch.concat([z_gat1, z_gat2], dim=1)
        return z

    def reparametrize(self, mu, log_std):
        if self.training:
            return mu + torch.randn_like(log_std) * torch.exp(log_std)
        else:
            return mu

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """

        loss_kl = 0.0
        loss_kl = -0.5 * torch.mean(torch.sum(1 + 2 * self.logstd_gat2 - self.mu_gat2 ** 2 - self.logstd_gat2.exp()**2, dim=1))
        loss_kl += -0.5 * torch.mean(
            torch.sum(1 + 2 * self.logstd_gat1 - self.mu_gat1 ** 2 - self.logstd_gat1.exp() ** 2, dim=1))
        return loss_kl / 2

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss
