"""Kiến trúc mô hình Graph Convolutional Network (GCN) 2 lớp."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden_dim)
        self.c2 = GCNConv(hidden_dim, out_dim)
        self.p = dropout
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.c1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.p, training=self.training)
        return self.c2(x, edge_index, edge_weight=edge_weight)
