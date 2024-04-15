# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2021/10/08 16:59
# @Author  : dongZheX
# @Version : python3.7
# @Desc    : $END$
import torch
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU, ModuleList, LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor, matmul
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DeepGCNLayer, ChebConv, GINConv, JumpingKnowledge, GatedGraphConv, GENConv
from torch_geometric.utils import to_undirected, add_self_loops
from models.mlp import MLP
import random
import numpy as np
from models.MPNCOV import Triuvec

class FOGConv(MessagePassing):
    def __init__(self, input_channel, hidden_channel1, hidden_channel2, output_channel, bias=False, **kwargs):
        super().__init__(aggr='add')
        kwargs.setdefault('aggr', 'add')
        super(FOGConv, self).__init__(**kwargs)
        self.input_channel = input_channel
        self.hidden_channel1 = hidden_channel1
        self.hidden_channel2 = hidden_channel2
        self.output_channel = output_channel
        self.bias = bias
        self.reduction1 = Linear(self.input_channel, self.hidden_channel1, bias=self.bias)
        self.norm1 = BatchNorm1d(self.hidden_channel1)
        self.reduction2 = Linear(self.hidden_channel1, self.hidden_channel2, bias=self.bias)
        self.norm2 = BatchNorm1d(self.hidden_channel2)
        self.proj = Linear(self.hidden_channel1*self.hidden_channel2, self.output_channel, bias=self.bias)

    def reset_parameters(self):
        self.reduction1.reset_parameters()
        self.norm1.reset_parameters()
        self.reduction2.reset_parameters()
        self.norm2.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x:Tensor, edge_index:Tensor, edge_attr=None):
        # Mapping Layer
        x_des = self.norm1(F.leaky_relu(self.reduction1(x)))
        x_src = self.norm2(F.leaky_relu(self.reduction2(x_des)))
        feat = self.propagate(edge_index=edge_index, x=(x_src, x_des))

        return self.proj(feat)

    def message(self, x_j:Tensor, x_i:Tensor, edge_attr:Tensor = None) -> Tensor:
        # x_j is source, x_i is target
        E, D1 = x_i.shape
        E, D2 = x_j.shape
        # source(x_i) to target(x_j)
        KP = torch.einsum("ab,ac->abc", x_i, x_j).reshape(E, D1*D2)
        # How to fuse edge_attr
        return KP

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        x_src, x_des = x
        N = x_src.size(0)
        x_ = matmul(adj_t, x_src, reduce=self.aggr)
        return torch.einsum('ik,kj -> kij', x_des.transpose(0, 1), x_).reshape(N, -1)


class _FOGConv(MessagePassing):
    def __init__(self, input_channel, hidden_channel, output_channel, is_triu=True, **kwargs):
        super().__init__(aggr='add')
        kwargs.setdefault('aggr', 'add')
        super(_FOGConv, self).__init__(**kwargs)
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel
        self.is_triu = is_triu
        self.reduction1 = Linear(self.input_channel, self.hidden_channel, bias=self.bias)
        self.norm1 = BatchNorm1d(self.hidden_channel1)
        self.reduction2 = Linear(self.input_channel, self.hidden_channel, bias=self.bias)
        self.norm2 = BatchNorm1d(self.hidden_channel2)
        if self.is_triu:
            self.proj = Linear(int(self.hidden_channel1*(self.hidden_channel2+1)/2), self.output_channel, bias=self.bias)
        else:
            self.proj = Linear(self.hidden_channel*self.hidden_channel, self.output_channel, bias=self.bias)


    def reset_parameters(self):
        self.reduction1.reset_parameters()
        self.norm1.reset_parameters()
        self.reduction2.reset_parameters()
        self.norm2.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x:Tensor, edge_index:Tensor, edge_attr=None):
        # Mapping Layer
        x_des = self.norm1(F.relu(self.reduction1(x)))
        x_src = self.norm2(F.relu(self.reduction2(x)))
        feat = self.propagate(edge_index=edge_index, x=(x_src, x_des))
        return self.proj(feat)

    def message(self, x_j:Tensor, x_i:Tensor, edge_attr:Tensor = None) -> Tensor:
        # x_j is source, x_i is target
        E, D1 = x_i.shape
        E, D2 = x_j.shape
        # source(x_i) to target(x_j)
        KP = torch.einsum("ab,ac->abc", x_i, x_j)
        if self.is_triu:
            KP = Triuvec.apply(KP).reshape(E, -1)
        else:
            KP = KP.reshape(E, -1)
        # How to fuse edge_attr
        return KP

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        x_src, x_des = x
        N = x_src.size(0)
        x_ = matmul(adj_t, x_src, reduce=self.aggr)
        return torch.einsum('ik,kj -> kij', x_des.transpose(0, 1), x_).reshape(N, -1)



class FOGWithJK(torch.nn.Module):
    def __init__(self, args):
        super(FOGWithJK, self).__init__()
        self.num_layers = args.num_layers
        self.input_channel = args.input_channel
        self.embed_dim = args.embed_dim
        self.hidden_channel1 = args.hidden_channel1
        self.hidden_channel2 = args.hidden_channel2
        self.output_channel = args.output_channel 
        self.bias = args.bias
        self.dropout = args.dropout
        self.jk = args.jk
        self.use_residual = args.use_residual
        self.convs = torch.nn.ModuleList()
        self.convs.append(FOGConv(self.input_channel, self.hidden_channel1, self.hidden_channel2, self.embed_dim, bias=self.bias))
        self.bns = torch.nn.ModuleList()
        self.bns.append(BatchNorm1d(args.embed_dim))
        self.mlp = args.mlp
        # We can simple think output_channel = input_channel
        # Set FogConv
        for _ in range(self.num_layers - 2):
            self.convs.append(
                FOGConv(self.embed_dim, self.hidden_channel1, self.hidden_channel2, self.embed_dim, bias=self.bias)
            )
            self.bns.append(BatchNorm1d(self.embed_dim))
        self.convs.append(
            FOGConv(self.embed_dim, self.hidden_channel1, self.hidden_channel2, self.embed_dim, bias=self.bias)
        )
        # Set JumpKnowledge
        if self.jk != "last" and self.jk != "sum" and self.jk != "sump":
            self.jump = JumpingKnowledge(mode=self.jk, channels=self.embed_dim, num_layers=self.num_layers)
        elif self.jk == "sum":
            self.jump = SumJK(num_layers=self.num_layers, embed_dim=self.embed_dim, output_channel=self.mlp[-1],
                              mode="nop")
        elif self.jk == "sump":
            self.jump = SumJK(num_layers=self.num_layers, embed_dim=self.embed_dim, output_channel=self.mlp[-1],
                              mode="p")
        # Set predication layer
        if self.jk == "sump":
            self.mlp = None

        elif self.jk == 'cat':
            self.mlp = MLP(self.embed_dim*self.num_layers, self.mlp, batch_norm=False)
        else:
            self.mlp = MLP(self.embed_dim, self.mlp, batch_norm=False)
        self.dropout = args.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.mlp is not None:
            self.mlp.reset_parameters()
        if self.jk != "last":
            self.jump.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        _x = 0
        # forward
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + _x         # residual
            xs += [x]
            _x = x
        x = self.convs[-1](x, edge_index)
        if self.use_residual:
            x = x + _x  # residual
        xs += [x]
        # jk
        if self.jk != "last":
            x = self.jump(xs)
        # prediction layer
        if self.mlp is not None:
            x = self.mlp(x)
        return x.log_softmax(dim=-1)

class FOGGNN(torch.nn.Module):
    def __init__(self, args):
        super(FOGGNN, self).__init__()
        self.num_layers = args.num_layers
        self.input_channel = args.input_channel
        self.embed_dim = args.embed_dim
        self.hidden_channel1 = args.hidden_channel1
        self.hidden_channel2 = args.hidden_channel2
        self.output_channel = args.output_channel
        self.bias = args.bias
        self.dropout = args.dropout
        self.jk = args.jk
        self.use_residual = args.use_residual
        self.mlp = args.mlp

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            FOGConv(self.input_channel, self.hidden_channel1, self.hidden_channel2, int(self.embed_dim / 2),
                    bias=self.bias))
        self.bns = torch.nn.ModuleList()
        self.bns.append(BatchNorm1d(args.embed_dim))
        self.gnns = torch.nn.ModuleList()
        self.gnns.append(
            get_layer(args.attach_gnn, input_channels=self.input_channel, output_channels=int(self.embed_dim / 2),
                      args=args, layer_id=0))
        for i in range(self.num_layers - 2):
            self.convs.append(
                FOGConv(self.embed_dim, self.hidden_channel1, self.hidden_channel2, int(self.embed_dim/2), bias=self.bias)
            )
            self.gnns.append(
                get_layer(args.attach_gnn, input_channels=self.embed_dim, output_channels=int(self.embed_dim / 2),
                          args=args, layer_id=i + 1))
            self.bns.append(BatchNorm1d(self.embed_dim))
        self.convs.append(
            FOGConv(self.embed_dim, self.hidden_channel1, self.hidden_channel2, int(self.embed_dim/2), bias=self.bias)
        )
        self.gnns.append(
            get_layer(args.attach_gnn, input_channels=self.embed_dim, output_channels=int(self.embed_dim / 2),
                      args=args, layer_id=self.num_layers - 1))


        # Set JumpKnowledge
        if self.jk != "last" and self.jk != "sum" and self.jk != "sump":
            self.jump = JumpingKnowledge(mode=self.jk, channels=self.embed_dim, num_layers=self.num_layers)
        elif self.jk == "sum":
            self.jump = SumJK(num_layers=self.num_layers, embed_dim=self.embed_dim, output_channel=self.mlp[-1],
                              mode="nop")
        elif self.jk == "sump":
            self.jump = SumJK(num_layers=self.num_layers, embed_dim=self.embed_dim, output_channel=self.mlp[-1],
                              mode="p")
        # Set predication layer
        if self.jk == "sump":
            self.mlp = None

        elif self.jk == 'cat':
            self.mlp = MLP(self.embed_dim*self.num_layers, self.mlp, batch_norm=False)
        else:
            self.mlp = MLP(self.embed_dim, self.mlp, batch_norm=False)
        self.dropout = args.dropout



    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for gnn in self.gnns:
            gnn.reset_parameters()
        if self.mlp is not None:
            self.mlp.reset_parameters()
        if self.jk != "last":
            self.jump.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        _x = 0
        # forward
        for i in range(self.num_layers-1):
            x = torch.cat([self.convs[i](x, edge_index), self.gnns[i](x, edge_index)], dim=1)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + _x         # residual
            xs += [x]
            _x = x
        x = torch.cat([self.convs[-1](x, edge_index), self.gnns[-1](x, edge_index)], dim=1)
        if self.use_residual:
            x = x + _x  # residual
        xs += [x]
        # jk
        if self.jk != "last":
            x = self.jump(xs)
        # prediction layer
        if self.mlp is not None:
            x = self.mlp(x)
        return x.log_softmax(dim=-1)

class OGNN(torch.nn.Module):
    def __init__(self, args):
        super(OGNN, self).__init__()
        self.num_layers = args.num_layers
        self.input_channel = args.input_channel
        self.embed_dim = args.embed_dim
        self.output_channel = args.output_channel
        self.dropout = args.dropout
        self.jk = args.jk
        self.use_residual = args.use_residual
        self.mlp = args.mlp

        self.bns = torch.nn.ModuleList()
        self.bns.append(BatchNorm1d(args.embed_dim))
        self.gnns = torch.nn.ModuleList()
        self.gnns.append(
            get_layer(args.attach_gnn, input_channels=self.input_channel, output_channels=self.embed_dim,
                      args=args, layer_id=0))
        for i in range(self.num_layers - 2):
            self.gnns.append(
                get_layer(args.attach_gnn, input_channels=self.embed_dim, output_channels=self.embed_dim ,
                          args=args, layer_id=i + 1))
            self.bns.append(BatchNorm1d(self.embed_dim))

        self.gnns.append(
            get_layer(args.attach_gnn, input_channels=self.embed_dim, output_channels=self.embed_dim,
                      args=args, layer_id=self.num_layers - 1))


        # Set JumpKnowledge
        if self.jk != "last" and self.jk != "sum" and self.jk != "sump":
            self.jump = JumpingKnowledge(mode=self.jk, channels=self.embed_dim, num_layers=self.num_layers)
        elif self.jk == "sum":
            self.jump = SumJK(num_layers=self.num_layers, embed_dim=self.embed_dim, output_channel=self.mlp[-1],
                              mode="nop")
        elif self.jk == "sump":
            self.jump = SumJK(num_layers=self.num_layers, embed_dim=self.embed_dim, output_channel=self.mlp[-1],
                              mode="p")
        # Set predication layer
        if self.jk == "sump":
            self.mlp = None

        elif self.jk == 'cat':
            self.mlp = MLP(self.embed_dim*self.num_layers, self.mlp, batch_norm=False)
        else:
            self.mlp = MLP(self.embed_dim, self.mlp, batch_norm=False)
        self.dropout = args.dropout

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()
        for gnn in self.gnns:
            gnn.reset_parameters()
        if self.mlp is not None:
            self.mlp.reset_parameters()
        if self.jk != "last":
            self.jump.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        _x = 0
        # forward
        for i in range(self.num_layers-1):
            x = self.gnns[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + _x         # residual
            xs += [x]
            _x = x
        x = self.gnns[-1](x, edge_index)
        if self.use_residual:
            x = x + _x  # residual
        xs += [x]
        # jk
        if self.jk != "last":
            x = self.jump(xs)
        # prediction layer
        if self.mlp is not None:
            x = self.mlp(x)
        return x.log_softmax(dim=-1)

class SumJK(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, output_channel, mode="nop"):
        # mode \in nop, p
        super(SumJK, self).__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.output_channel = output_channel
        if self.mode == "p":
            self.linears = torch.nn.ModuleList()
            for i in range(self.num_layers):
                self.linears.append(Linear(self.embed_dim, self.output_channel))

    def reset_parameters(self):
        if self.mode == "p":
            for layer in self.linears:
                layer.reset_parameters()

    def forward(self, xs):
        res = 0
        for i, x in enumerate(xs):
            if self.mode == "p":
                res += self.linears[i](x)
            else:
                res += x
        return res

def get_layer(attach_gnn, input_channels, output_channels, args, layer_id) -> torch.nn.Module:
    layer = None
    # In general, input_channels == output_channels
    if attach_gnn == "gcn":
        layer = GCNConv(input_channels, output_channels, cached=True)
    elif attach_gnn == "sage":
        layer = SAGEConv(input_channels, output_channels)
    elif attach_gnn == "gin":
        layer = GINConv(
            Sequential(Linear(input_channels, args.embed_dim), BatchNorm1d(args.embed_dim), ReLU(),
                       Linear(args.embed_dim, output_channels), ReLU()))
    elif attach_gnn == "gat":
        layer = GATConv(input_channels, int(output_channels/args.heads[layer_id]), args.heads[layer_id], dropout=0.6)
    elif attach_gnn == "gated":
        # input channels should smaller than output_channels
        layer = GatedGraphConv(out_channels=output_channels, num_layers=args.num_layers)
    elif attach_gnn == "deepergcn":
        conv = GENConv(input_channels, output_channels, aggr='softmax',
                       t=1.0, learn_t=True, num_layers=2, norm='layer')
        norm = LayerNorm(output_channels, elementwise_affine=True)
        act = ReLU(inplace=True)
        layer = DeepGCNLayer(conv, norm, act, block='plain', dropout=0.1,
                             ckpt_grad=(layer_id+1) % 3)
    return layer




