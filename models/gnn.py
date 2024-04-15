# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2021/10/23 1:53
# @Author  : dongZheX
# @Version : python3.7
# @Desc    : $END$
import os

import torch
from torch_geometric.nn import MessagePassing

import torch.nn.functional as F
import argparse
from models.conv import GNN_node, GNN_node_Virtualnode
import sop_tools
from torch_scatter import scatter_mean
from models.sopool import Readout
from models.mlp import MLP


class GNN(torch.nn.Module):
    def __init__(self, args, dataset):
        super(GNN, self).__init__()
        self.args = args
        self.num_layer = args.num_layer
        self.dropout = args.drop_ratio
        self.JK = args.jk
        self.emb_dim = args.emb_dim
        self.num_tasks = dataset.num_tasks
        self.residual = args.use_residual
        self.tmp_dim = self.emb_dim
        self.graph_pred_list = self.args.graph_pred_list
        self.reduction_dim = self.args.reduction_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1")
        if self.args.virtual_node:
            self.gnn_node = GNN_node_Virtualnode(self.args, self.num_layer, self.emb_dim, JK=self.JK,
                                                 drop_ratio=self.dropout,
                                                 residual=self.residual, gnn_type=self.args.gnn, vp=self.args.vp)
        else:
            self.gnn_node = GNN_node(self.args, self.num_layer, self.emb_dim, JK=self.JK, drop_ratio=self.dropout,
                                     residual=self.residual, gnn_type=self.args.gnn)
        if self.graph_pred_list is None:
            self.graph_pred_list = []
        self.graph_pred_list.append(self.num_tasks)
        self.reduction_layer = None
        if self.JK == "cat":
            self.emb_dim = self.emb_dim * 2
        
        if self.args.multi_readout is not None:
            self.readouts = torch.nn.ModuleList()
            this_channel_list = []
            # multi_readout: num_layer+1
            for i in range(self.num_layer+1):
                tmp = None
                if self.reduction_dim is not None:
                    tmp = MLP(self.emb_dim, self.reduction_dim, batch_norm=self.args.norm)
                    self.tmp_dim = self.reduction_dim[-1]
                self.readouts.append(Readout(self.tmp_dim, tmp, self.args.multi_readout[i],
                                             num_iter=self.args.num_iter, is_triu=self.args.is_triu,
                                             dropout_ratio=self.args.final_dropout_ratio, k=self.args.k,
                                             learn_t=self.args.learn_t,
                                             isqrt=self.args.isqrt, args=self.args))

                this_channel_list.append(self.get_this_channels(self.args.multi_readout[i]))
        else:
            if self.reduction_dim is not None:
                self.reduction_layer = MLP(self.emb_dim, self.reduction_dim, batch_norm=self.args.norm)
                self.tmp_dim = self.reduction_dim[-1]
            # Readout
            self.readout = Readout(self.tmp_dim, self.reduction_layer, self.args.pool_method,
                                   num_iter=self.args.num_iter, is_triu=self.args.is_triu,
                                   dropout_ratio=self.args.final_dropout_ratio, k=self.args.k,
                                   learn_t=self.args.learn_t,
                                   isqrt=self.args.isqrt, args=self.args)
        this_channels = 0

        # graph_pred_layer
        if self.args.multi_pred:
            self.graph_pred_layer_list = torch.nn.ModuleList()
            for i in this_channel_list:
                self.graph_pred_layer_list.append(MLP(i, self.graph_pred_list, batch_norm=False))
        else:
            if self.args.multi_readout is not None:
                this_channels = self.get_this_channels(self.args.multi_readout[0])
            else:
                this_channels = self.get_this_channels(self.args.pool_method)
            self.graph_pred_layer = MLP(this_channels, self.graph_pred_list, batch_norm=False)

        # pretrain
        if self.args.pretrain is True:
            pretrain_dict = torch.load(os.getcwd() + "/" + self.args.model_path,map_location=torch.device("cpu"))['net']

            model_dict = self.state_dict()
            pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            self.load_state_dict(model_dict)
            for _, param in self.gnn_node.named_parameters():
                param.requires_grad = False
        if self.args.moredata is True:
            pretrain_dict = torch.load(os.getcwd() + "/" + self.args.model_path,map_location=torch.device("cpu"))['model_state_dict']

            model_dict = self.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            self.load_state_dict(model_dict)
            for _, param in self.gnn_node.named_parameters():
                param.requires_grad = False

    def open_grad(self):
        for _, param in self.gnn_node.named_parameters():
            param.requires_grad = True

    def reset_parameters(self):

        if self.args.multi_readout:
            for readout in self.readouts:
                readout.reset_parameters()
            if self.args.multi_pred:
                for layer in self.graph_pred_layer_list:
                    layer.reset_parameters()
        else:
            self.readout.reset_parameters()
            self.graph_pred_layer.reset_parameters()

    def get_this_channels(self, pool_method):
        this_channels = 0
        if pool_method == "set2set":
            this_channels = 2 * self.tmp_dim
        elif pool_method in sop_tools.seq:
            # G2DeNet not considered
            if self.args.is_triu is True:
                this_channels = int((self.tmp_dim + 1) * self.tmp_dim / 2)
            
            else:
                this_channels = self.tmp_dim * self.tmp_dim
        elif pool_method == "sop_att_m":
            d_k = self.tmp_dim/self.args.num_heads;
            if self.args.is_triu is True:
                this_channels = int(d_k * (d_k+1)/2) * self.args.num_heads
            else:
                this_channels = int(d_k * (d_k+1)) * self.args.num_heads
        else:
            this_channels = self.tmp_dim
        return this_channels

    def forward(self, batched_data, perturb=None):
        h_node, h_list = self.gnn_node(batched_data, perturb)
        
        output = 0
        outputs = []
        if self.args.multi_readout:
            for i, readout in enumerate(self.readouts):

                outputs.append(readout(h_list[i], batched_data.batch, batched_data.edge_index))
        else:
            output = self.readout(h_node, batched_data.batch, batched_data.edge_index)
        if self.args.multi_pred:
            for i in range(self.num_layer+1):
                output += F.dropout(self.graph_pred_layer_list[i](outputs[i]), p=self.args.fdr, training=self.training)
        else:
            if self.args.multi_readout:
                for o in outputs:
                    output += o
            output = F.dropout(self.graph_pred_layer(output), p=self.args.fdr, training=self.training)
        if self.args.fp>0:
            fp_feat = batched_data.y[:, 2]
            fp_feat = torch.sigmoid(fp_feat)
            output = torch.sigmoid(output)
            return torch.clamp((1-self.args.fp)*output + self.args.fp* fp_feat.reshape(-1,1),min=0, max=1)
        else:
            return output
