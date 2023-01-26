import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch_scatter import scatter_mean
from torch_geometric.utils import remove_self_loops, add_self_loops, degree, scatter_, dropout_adj, softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import GATConv

class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_index, num_node, aggr='add', bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.num_node = num_node
        row, col = edge_index
        deg = degree(row, num_node)
        deg_in_sqrt = deg.pow(0.5)
        norm = deg_in_sqrt[row] * deg_in_sqrt[col]
        self.norm = norm.view(-1, 1)

    def forward(self, x):
        return self.propagate(edge_index=self.edge_index, size=None, x=x, edge_weight=None, res_n_id=None)

    def message(self, x_i, x_j):
        return self.norm * x_j

    def update(self, aggr_out, x):
        return aggr_out
