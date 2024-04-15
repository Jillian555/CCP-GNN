# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2021/10/22 12:50
# @Author  : dongZheX
# @Version : python3.7
# @Desc    : $END$
import sys
import torch_geometric
from models.MPNCOV import Sqrtm, Triuvec
import torch
from models.mlp import MLP
import fitlog
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean, scatter_max, scatter_sum, scatter_softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from torch.autograd import Variable
from collections import Counter
import torch_geometric.utils as ut
from torch_geometric.utils import softmax


def process_padding_zero(x, batch, edge):
    # input M x dim
    # output N x max(nodenum) * fim
    batchsize = batch[-1] + 1
    dim = x.size(1)
    count = np.asarray(list(Counter(batch.cpu().numpy()).values()))

    N = max(count)
    output = torch.zeros(N * batchsize.item(), dim, device=x.device)
    mask = torch.zeros(N * batchsize.item(), 1, device=x.device)
    e = edge.cpu().numpy()
    degree = torch.from_numpy(np.asarray([np.sum(e == i)/2 for i in range(x.shape[0])])).reshape(-1,1).to(x.device)
    _degree = torch.zeros(N * batchsize.item(), 1, device=x.device)
    idx = 0
    jdx = 0
    for i in range(batchsize):
        output[idx:idx+count[i]] = x[jdx:jdx+count[i]]
        mask[idx:idx + count[i]] = 1
        _degree[idx:idx + count[i]] = degree[jdx:jdx+count[i]]
        idx = idx + N
        jdx = jdx + count[i]
    output = output.view(batchsize, N, dim)
    mask = mask.view(batchsize, N, 1)
    return output, count, mask, _degree


class Readout(nn.Module):
    def __init__(self, hidden_channels, reduction_layer, operator='sop', num_iter=3, is_triu=False, dropout_ratio=0,
                 k=16, learn_t=False,  isqrt=False,  args=None):
        super(Readout, self).__init__()
        self.hidden_channels = hidden_channels
        self.reduction_layer = reduction_layer
        self.operator = operator
        self.num_iter = num_iter
        self.is_triu = is_triu
        self.dropout_ratio = dropout_ratio
        self.learn_t = learn_t
        self.k = k
        self.args = args
        self.isqrt = isqrt

        if self.operator == "sop_att":
            self.sop_att = _global_attention_sop(self.hidden_channels, is_triu=self.is_triu,
                                                 use_softmax=not self.args.sopattsigmoid, isqrt=self.isqrt,
                                                 num_iter=self.num_iter,  degree=args.degree,
                                                 sr=args.sr, fix_triu=args.fix_triu, args=args)
        if self.operator == "sop_att_m":
            self.sop_att_m = _global_attention_sop_m(self.hidden_channels, is_triu=self.is_triu,
                                                 use_softmax=not self.args.sopattsigmoid, isqrt=self.isqrt,
                                                 num_iter=self.num_iter,  degree=args.degree,
                                                 sr=args.sr, fix_triu=args.fix_triu, args=args)
        if self.operator == "sop_att_triu":
            self.sop_att_triu = _global_attention_sop_triu(self.hidden_channels, is_triu=self.is_triu,
                                                           use_softmax=not self.args.sopattsigmoid, isqrt=self.isqrt,
                                                           num_iter=self.num_iter, sr=args.sr,
                                                           fix_triu=args.fix_triu)
        elif self.operator == "sop_att_d":
            self.sop_att_d = _global_attention_sop_d(self.hidden_channels, is_triu=self.is_triu,
                                                     use_softmax=not self.args.sopattsigmoid, isqrt=self.isqrt,
                                                     num_iter=self.num_iter, degree=args.degree)
        elif self.operator == "sop_att_n":
            self.sop_att_n = _global_attention_sop_n(self.hidden_channels, is_triu=self.is_triu,
                                                     use_softmax=not self.args.sopattsigmoid, isqrt=self.isqrt,
                                                     num_iter=self.num_iter, degree=False)
        elif self.operator == "sop_att_f":
            self.sop_att_f = _global_attention_sop_f(self.hidden_channels, is_triu=self.is_triu,
                                                     use_softmax=not self.args.sopattsigmoid, isqrt=self.isqrt,
                                                     num_iter=self.num_iter, degree=args.degree)
        elif self.operator == "sop_att_u":
            self.sop_att_u = _global_attention_sop_u(self.hidden_channels)
        elif self.operator == "sop_mean_max":
            self.sop_mean_max = _mean_max_sop(self.hidden_channels, is_triu=self.is_triu, learn_t=self.learn_t,
                                              isqrt=self.isqrt, num_iter=self.num_iter)
        elif self.operator == "g2denet":
            self.G2DeNet = G2DeNet()

    def reset_parameters(self):
        if self.reduction_layer is not None:
            self.reduction_layer.reset_parameters()
        if self.operator == "sop_att":
            self.sop_att.reset_parameters()
        if self.operator == "sop_att_triu":
            self.sop_att_triu.reset_parameters()
        elif self.operator == "sop_att_d":
            self.sop_att_d.reset_parameters()
        elif self.operator == "sop_att_n":
            self.sop_att_n.reset_parameters()
        elif self.operator == "sop_att_f":
            self.sop_att_f.reset_parameters()
        elif self.operator == "sop_att_u":
            self.sop_att_u.reset_parameters()
        elif self.operator == "sop_mean_max":
            self.sop_mean_max.reset_parameters()

    def forward(self, x, batch, edge):
        if self.reduction_layer is not None:
            x = self.reduction_layer(x)
        # x, count, mask, degree = process_padding_zero(x, batch, edge)
        M,dim = x.size()
        batch_size = batch[-1].item() + 1
        if self.operator == "sop":
            output = torch.einsum("ik,kj->kij", x.transpose(1,0), x)
            output = scatter_sum(output, batch, dim=0)
            if self.isqrt:
                output = Sqrtm.apply(output, iterN=self.num_iter)
            if self.is_triu:
                output = Triuvec.apply(output).view(batch_size, -1)
            else:
                output = output.view(batch_size, -1)
        elif self.operator == "sop_max":
            output = _global_max_sop(x, batch=batch, is_triu=self.is_triu, isqrt=self.isqrt, num_iter=self.num_iter)
        elif self.operator == "sop_topk":
            x, count, mask, degree = process_padding_zero(x, batch, edge)
            output = _global_topk_sop(x, count=count, k=self.k, is_triu=self.is_triu, isqrt=self.isqrt,
                                      num_iter=self.num_iter)
        elif self.operator == "sop_att":
            output = self.sop_att(x, batch, edge)
        elif self.operator == "sop_att_m":
            output = self.sop_att_m(x, batch, edge)
        elif self.operator == "sop_att_triu":
            output = self.sop_att_triu(x, batch, edge)
        elif self.operator == "sop_att_d":
            output = self.sop_att_d(x, batch)
        elif self.operator == "sop_att_n":
            output = self.sop_att_n(x, batch, edge)
        elif self.operator == "sop_att_f":
            output = self.sop_att_f(x, batch, edge)
        elif self.operator == "sop_att_u":
            output = self.sop_att_u(x)
        elif self.operator == "sop_mean_max":
            x, count, mask, degree = process_padding_zero(x, batch, edge)
            output = self.sop_mean_max(input, count, mask)
        elif self.operator == "g2denet":
            x, count, mask, degree = process_padding_zero(x, batch, edge)
            output = self.G2DeNet(x)
            if self.isqrt:
                output = Sqrtm(output, 3)
            if self.is_triu:
                output = Triuvec.apply(output).view(batch_size, -1)
            else:
                output = output.view(batch_size, -1)
        elif self.operator == "mean":
            output = global_mean_pool(x, batch)
        elif self.operator == "max":
            output = global_max_pool(x, batch)
        elif self.operator == "sum":
            output = global_add_pool(x, batch)
        elif self.operator == "set2set":
            output = Set2Set(x, batch)
        elif self.operator == "attention":
            output = GlobalAttention(x, batch)
        if self.dropout_ratio != 0:
            output = F.dropout(output, p=self.dropout_ratio, training=self.training)
        return output


# max pooling with padding
def _global_max_sop(x, batch, is_triu=False, isqrt=False, num_iter=3):
    N = x.size(0)
    embed_dim = x.size(1)
    wj = torch.einsum('ik,kj -> kij', x.transpose(1, 0), x).reshape(N, -1)
    max_wj, _ = scatter_max(wj, batch, dim=0)
    batch_size = max_wj.size(0)
    if isqrt:
        max_wj = Sqrtm.apply(max_wj, num_iter)
    if is_triu:
        max_wj = Triuvec.apply(max_wj)
    output = max_wj.view(batch_size, -1)
    return output


# topk sop Need Padding
def _global_topk_sop(x, count, k, is_triu=False, isqrt=False, num_iter=3):
    # k should smaller that min(count)
    # k = k if k < min(count) else min(count)
    batch_size = x.size(0)
    N = x.size(1)
    embed_dim = x.size(2)
    wj = torch.einsum('pik,pkj -> pkij', x.transpose(2, 1), x)
    k = k if k <= wj.size(0) else wj.size(0)
    sorted_wj = torch.topk(wj, dim=1, k=k)[0]
    # # sorted_wj = wj
    # sorted_wj = sorted_wj[:, 0:k, :, :]
    sorted_wj = torch.mean(sorted_wj, dim=1)
    if isqrt:
        sorted_wj = Sqrtm.apply(sorted_wj, num_iter)
    if is_triu:
        sorted_wj = Triuvec.apply(sorted_wj)
    output = sorted_wj.view(batch_size, -1)
    return output


# sop_attention
class _global_attention_sop(nn.Module):
    def __init__(self, hidden_channels, use_softmax=True, is_triu=False, isqrt=False, num_iter=3, degree=False, sr=1, fix_triu=False, args=None):
        super(_global_attention_sop, self).__init__()
        self.hidden_channels = hidden_channels
        self.is_triu = is_triu
        self.isqrt = isqrt
        self.num_iter = num_iter
        self.printimp = "printprint"
        self.printimp1 = 0
        self.attention = nn.Linear(hidden_channels * hidden_channels, 1)
        self.use_softmax = use_softmax
        self.sr = sr
        self.fix_triu = fix_triu
        self.degree = degree
        self.degree_1 = args.degree_1
        self.degree_2 = args.degree_2
        self.mask = args.mask
        if self.degree_1 is True:
            self.degree_embedding = torch.nn.Embedding(512,1,padding_idx=0)
        if self.degree_2 is True:
            self.degree_linear = torch.nn.Linear(1,1)
    def reset_parameters(self):
        self.attention.reset_parameters()
        if self.degree_1:
            self.degree_embedding.reset_parameters()
        if self.degree_2:
            self.degree_linear.reset_parameters()
    def forward(self, x, batch, edge):
        batch_size = batch[-1].item() + 1
        M = x.size(0)
        embed_dim = x.size(1)
        wj = torch.einsum('ik,kj -> kij', x.transpose(1, 0), x)  # (M,D,D)
        flatten_wj = wj.view(M, -1)  # (B,N,D^2)
        imp = self.attention(flatten_wj)  # Calculate importance(M,1)
        degree = torch_geometric.utils.degree(edge[0], num_nodes=M).reshape(-1,1)
        if self.degree_1 is True:
            degree = self.degree_embedding(degree.long())
        if self.degree_2 is True:
            degree = self.degree_linear(degree)
        if self.sr != 1:
            imp = imp * self.sr
        if self.use_softmax is True:
            imp = scatter_softmax(imp, batch, dim=0)
            degree = scatter_softmax(degree, batch, dim=0)
        else:
            imp = torch.sigmoid(imp)
            degree = torch.sigmoid(degree)
        if self.degree:
            imp = imp + degree
            self.printimp1 = imp - degree
            
        self.printimp = imp
        
        if self.mask < 1:
            max_imp, _ = scatter_max(imp, batch, dim=0)
            threshold = self.mask * max_imp[batch]
            imp = torch.where(imp>=threshold, max_imp[batch], imp)
        output = scatter_mean(imp * flatten_wj, batch, dim=0) 
        output = output.view(batch_size, embed_dim, embed_dim)
        if self.isqrt is True:
            if self.fix_triu:
                phi = torch.eye(embed_dim, embed_dim).repeat(batch_size, 1, 1).to(x.device) * 1e-3
                output = output + phi
            output = Sqrtm.apply(output, self.num_iter)
        if self.is_triu:
            output = Triuvec.apply(output).squeeze()
            #layer norm
            #output = torch.nn.functional.layer_norm(output, normalized_shape=[output.shape[1]])
            #sqrt
            #processed_z = torch.sign(output) * torch.sqrt(torch.abs(output)+1e-5)
            #output = processed_z / (torch.norm(processed_z, p=2, dim=1, keepdim=True)+1e-5)
            
        output = output.view(batch_size, -1)
        return output

# multi-head attention sop_attention
class _global_attention_sop_m(nn.Module):
    def __init__(self, hidden_channels, use_softmax=True, is_triu=False, isqrt=False, num_iter=3, degree=False, sr=1, fix_triu=False, args=None):
        super(_global_attention_sop_m, self).__init__()
        self.hidden_channels = hidden_channels
        self.is_triu = is_triu
        self.isqrt = isqrt
        self.num_iter = num_iter
        self.num_heads = args.num_heads;
        self.d_k = int(self.hidden_channels / self.num_heads)
        self.printimp = "printprint"
        self.printimp1 = 0
        
        self.attentions = torch.nn.ModuleList()
        for i in range(self.num_heads):
            self.attentions.append(nn.Linear(self.d_k*self.d_k, 1))
        self.use_softmax = use_softmax
        self.sr = sr
        self.fix_triu = fix_triu
        self.degree = degree
        self.mask = args.mask
        self.degree_1 = args.degree_1
        self.degree_2 = args.degree_2
        self.mask = args.mask
        if self.degree_1 is True:
            self.degree_embedding = torch.nn.Embedding(512,self.num_heads)
        if self.degree_2 is True:
            self.degree_linear = torch.nn.Linear(1,self.num_heads) 
            
    def reset_parameters(self):
        self.attention.reset_parameters()
        if self.degree_1:
            self.degree_embedding.reset_parameters()
        if self.degree_2:
            self.degree_linear.reset_parameters()
            
    def forward(self, x, batch, edge):
        batch_size = batch[-1].item() + 1
        M = x.size(0)
        embed_dim = x.size(1)
        x = x.view(M, self.num_heads, self.d_k)
        
        #---------------------------------------------------------------------------------
        
        wj = torch.einsum('hik,hkj -> hkij', x.permute(1,2,0), x.permute(1,0,2))  # (H,M,D,D)
        flatten_wj = wj.view(self.num_heads, M, -1)  # (H,M,D^2)
        imp = torch.cat([self.attentions[i](flatten_wj[i,:,:]) for i in range(self.num_heads)]).view(self.num_heads,M,1)
        # Calculate importance(M,1)
        degree = torch_geometric.utils.degree(edge[0], num_nodes=M).reshape(-1,1)
        if self.degree_1:
            degree = self.degree_embedding(degree.long()).view(M,self.num_heads,1).permute(1,0,2)
        elif self.degree_2:
            degree = self.degree_linear(degree).view(M,self.num_heads,1).permute(1,0,2)
        else:
            degree = degree.unsqueeze(1).repeat(1,self.num_heads,1).permute(1,0,2)
        
        if self.sr != 1:
            imp = imp * self.sr
        if self.use_softmax is True:
            imp = scatter_softmax(imp, batch, dim=1)
            degree = scatter_softmax(degree, batch, dim=1)
        else:
            imp = torch.sigmoid(imp)
            degree = torch.sigmoid(degree)
        if self.degree_1 or self.degree_2:
            imp = imp + degree
            self.printimp1 = imp - degree
            
        self.printimp = imp
        
    
        output = scatter_sum(imp * flatten_wj, batch, dim=1)
        # output = scatter_sum(imp * flatten_wj, batch, dim=0)  
        output = output.view(batch_size*self.num_heads, self.d_k, self.d_k)
        if self.isqrt is True:
            if self.fix_triu:
                phi = torch.eye(self.d_k, self.d_k).repeat(batch_size*self.num_heads, 1, 1).to(x.device) * 1e-3
                output = output + phi
            output = Sqrtm.apply(output, self.num_iter)
        if self.is_triu:
            output = Triuvec.apply(output)
        output = output.view(batch_size, -1)
        return output

# triangle attention
class _global_attention_sop_triu(nn.Module):
    def __init__(self, hidden_channels, use_softmax=True, is_triu=False, isqrt=False, num_iter=3, fix_triu=False, degree=False, sr=1):
        super(_global_attention_sop_triu, self).__init__()
        self.hidden_channels = hidden_channels
        self.sr = sr
        self.is_triu = is_triu
        self.isqrt = isqrt
        self.num_iter = num_iter
        self.printimp = None
        self.printimp1 = None
        self.attention = nn.Linear(int(hidden_channels * (hidden_channels + 1) / 2), 1)
        self.use_softmax = use_softmax
        self.fix_triu = fix_triu
        self.degree = degree

    def reset_parameters(self):
        self.attention.reset_parameters()

    def forward(self, x, batch, edge):
        batch_size = batch[-1].item() + 1
        M = x.size(0)
        embed_dim = x.size(1)
        wj = torch.einsum('ik,kj -> kij', x.transpose(1, 0), x)  # (M,D,D)
        _wj = wj
        # wj = wj.view(M, embed_dim, embed_dim)
        wj = Triuvec.apply(wj)
        flatten_wj = wj.view(M, -1)  # (M,D^2)
        imp = self.attention(flatten_wj)  # Calculate importance(M,1)
        degree = torch_geometric.utils.degree(edge[0]).reshape(-1,1)
        
        if self.sr != 1:
            imp = imp * self.sr
        if self.use_softmax:
            imp = scatter_softmax(imp, batch, dim=0)
            degree = scatter_softmax(degree, batch, dim=0)
        else:
            imp = torch.sigmoid(imp)
            degree = torch.sigmoid(degree)
            # imp = F.sigmoid(imp)
            self.printimp1 = self.printimp1 - degree
        if degree:
            imp = imp + degree
            self.printimp1  = imp - degree
        self.printimp = imp
        output = scatter_sum(imp * _wj, batch, dim=0)  
        output = output.view(batch_size, embed_dim, embed_dim)
        if self.isqrt:
            if self.fix_triu:
                phi = torch.eye(embed_dim, embed_dim).repeat(batch_size, 1, 1).to(x.device) * 1e-3
                output = output + phi
            output = Sqrtm.apply(output, self.num_iter)
        if self.is_triu:
            output = Triuvec.apply(output)
        output = output.view(batch_size, -1)
        return output


# channel attention + sop
class _global_attention_sop_d(nn.Module):
    def __init__(self, hidden_channels, is_triu=True, use_softmax=True, isqrt=False, num_iter=3, degree=False):
        super(_global_attention_sop_d, self).__init__()
        self.hidden_channels = hidden_channels
        self.is_triu = is_triu
        self.isqrt = isqrt
        self.num_iter = num_iter
        self.use_softmax = False

        self.attention = channel_attention(hidden_channels=hidden_channels, use_softmax=False, mode=1)

    def reset_parameters(self):
        self.attention.reset_parameters()

    def forward(self, x, batch):
        M, embed_dim = x.size()
        batch_size = batch[-1].item() + 1
        # x=>(B,N,D)
        x = self.attention(x, batch)
        output = torch.einsum("ik,kj->kij", x.transpose(1,0), x)  # (B, D, D)
        if self.isqrt:
            output = Sqrtm.apply(output, self.num_iter)
        if self.is_triu:
            output = Triuvec.apply(output)
        output = output.view(batch_size, -1)
        return output


# spatial_attention + sop
class _global_attention_sop_n(nn.Module):
    def __init__(self, hidden_channels, is_triu=True, use_softmax=True, isqrt=False, num_iter=3, degree=False):
        super(_global_attention_sop_n, self).__init__()
        self.hidden_channels = hidden_channels
        self.is_triu = is_triu
        self.isqrt = isqrt
        self.num_iter = num_iter
        self.use_softmax = use_softmax
        self.attention = spatial_gate_attention(hidden_channels=hidden_channels, use_softmax=use_softmax, degree=degree)
        self.printimp = 0
        self.printimp1 = None

    def reset_parameters(self):
        self.attention.reset_parameters()

    def forward(self, x, batch, edge):
        batch_size = batch[-1].item() + 1
        M, embed_dim = x.size()
        # x=>(B,N,D)
        x = self.attention(x, batch, edge)
        self.printimp = self.attention.printimp
        self.printimp1 = self.attention.printimp1
        output = torch.einsum("ik,kj->kij", x.transpose(1,0), x)
        if self.isqrt:
            output = Sqrtm.apply(output, self.num_iter)
        if self.is_triu:
            output = Triuvec.apply(output)
        output = output.view(batch_size, -1)
        return output


# first-order importance, second-order representation
class _global_attention_sop_f(nn.Module):
    def __init__(self, hidden_channels, is_triu=True, use_softmax=True, isqrt=False, num_iter=3, degree=False):
        super(_global_attention_sop_f, self).__init__()
        self.hidden_channels = hidden_channels
        self.is_triu = is_triu
        self.isqrt = isqrt
        self.num_iter = num_iter
        self.printimp = None
        self.printimp1 = None
        self.use_softmax = use_softmax
        self.module_list = torch.nn.ModuleList()
        self.attention = nn.Linear(hidden_channels, 1)
        self.degree = degree

    def reset_parameters(self):
        self.attention.reset_parameters()

    def forward(self, x, batch, edge):
        M, embed_dim = x.size()
        batch_size = batch[-1].item() + 1
        wj = torch.einsum('ik,kj -> kij', x.transpose(1, 0), x)  # (M,D,D)
        flatten_wj = wj.view(M, -1)  # (M,D^2)
        imp = self.attention(x)   # Calculate importance(M,N)
        degree = torch_geometric.utils.degree(edge[0]).reshape(-1,1)
    
        if self.use_softmax:
            imp = scatter_softmax(imp, batch, dim=0)
            degree = scatter_softmax(degree, batch, dim=0)
        else:
            imp = torch.sigmoid(imp)
            degree = torch.sigmoid(degree)
            # imp = F.sigmoid(imp)
        if self.degree:
            imp = im + degree
            self.printimp1 = imp - degree
        self.printimp = imp
        output = scatter_sum(imp * flatten_wj, batch, dim=0) 
        output = output.view(batch_size, embed_dim, embed_dim)
        if self.isqrt:
            output = Sqrtm.apply(output, self.num_iter)
        if self.is_triu:
            output = Triuvec.apply(output)
        output = output.view(batch_size, -1)
        return output


# first-order representation, second-order statistics
class _global_attention_sop_fs(nn.Module):
    def __init__(self, hidden_channels, use_softmax=True,  degree=False):
        super(_global_attention_sop_fs, self).__init__()
        self.hidden_channels = hidden_channels
        self.printimp = None
        self.printimp1 = None
        self.use_softmax = use_softmax
        self.module_list = torch.nn.ModuleList()
        self.attention = nn.Linear(hidden_channels * hidden_channels, 1)
        self.degree = degree

    def reset_parameters(self):
        self.attention.reset_parameters()

    def forward(self, x, batch, edge):
        M, embed_dim = x.size()
        batch_size = batch[-1].item() + 1
        wj = torch.einsum('ik,kj -> kij', x.transpose(1, 0), x)  # (M,D,D)
        flatten_wj = wj.view(M, -1)  # (M,D^2)
        imp = self.attention(flatten_wj)   # Calculate importance(M,N)
        degree = torch_geometric.utils.degree(edge[0]).reshape(-1,1)
        if self.use_softmax:
            imp = scatter_softmax(imp, batch, dim=0)
            degree = scatter_softmax(degree, batch, dim=0)
        else:
            imp = torch.sigmoid(imp)
            degree = torch.sigmoid(degree)
            # imp = F.sigmoid(imp)
        if self.degree:
            imp = imp + degree
            self.printimp1 = imp + degree
        self.printimp = imp
        output = scatter_sum(imp * x, batch, dim=0) 
        output = output.view(batch_size, embed_dim)
        return output

# attn_pool
class _global_attention_sop_u(nn.Module):
    def __init__(self, hidden_channels):
        super(_global_attention_sop_u, self).__init__()
        self.hidden_channels = hidden_channels
        self.attention = nn.Linear(hidden_channels, 1)

    def reset_parameters(self):
        self.attention.reset_parameters()

    def forward(self, x, batch):
        M, embed_dim = x.size()
        batch_size = batch[-1].item() + 1
        output = torch.einsum("ik,kj->kij",x.transpose(0,1),x)
        output = scatter_sum(output, batch, dim=0)
        output = self.attention(output)
        return output.reshape(batch_size, -1)


class channel_attention(nn.Module):
    def __init__(self, hidden_channels, mode=1, use_softmax=False):
        super(channel_attention, self).__init__()
        self.hidden_channels = hidden_channels
        self.use_softmax = use_softmax
        self.mode = mode
        if self.mode == 1:
            self.r = 4
            self.fc1 = nn.Linear(hidden_channels, int(hidden_channels / self.r))
            self.fc2 = nn.Linear(int(hidden_channels / self.r), hidden_channels)
        elif self.mode == 2:
            self.fc = nn.Linear(hidden_channels, hidden_channels)
        elif self.mode == 3:
            pass
    def reset_parameters(self):
        if self.mode == 1:
            self.fc1.reset_parameters()
            self.fc2.reset_parameters()
        elif self.mode ==2:
            self.fc.reset_parameters()

    def forward(self, x, batch):
        M, embed_dim = x.size()
        batch_size = batch[-1].item() + 1
        # x=>(M,D)
        gap = scatter_mean(x, batch, dim=0)  # (B,D)
        if self.mode == 1:
            imp = self.fc1(gap)  # (B,D/r)
            imp = F.leaky_relu(imp)
            imp = self.fc2(imp).reshape(batch_size, embed_dim)
        elif self.mode == 2:
            imp = self.fc(gap).reshape(batch_size, embed_dim)
        elif self.mode == 3:
            imp = gap.reshape(batch_size,  embed_dim)
        if self.use_softmax:
            imp = F.softmax(imp, dim=2)
        else:
            imp = torch.sigmoid(imp)
        imp = torch.repeat_interleave(imp, (scatter_sum(batch+1, batch)/scatter_mean(batch+1, batch)).type_as(batch), dim=0)
        return x * imp


class spatial_gate_attention(nn.Module):
    def __init__(self, hidden_channels, use_softmax=True, degree=False):
        super(spatial_gate_attention, self).__init__()
        self.hidden_channels = hidden_channels
        self.use_softmax = use_softmax
        self.printimp = 0
        self.printimp1 = 0
        self.attention = nn.Linear(hidden_channels, 1)
        self.degree = degree

    def reset_parameters(self):
        self.attention.reset_parameters()

    def forward(self, x, batch, edge):
        M, embed_dim = x.size()
        imp = self.attention(x)  # Calculate importance(M*1)
        degree = torch_geometric.utils.degree(edge[0]).reshape(-1,1)
        if self.use_softmax:
            imp = scatter_softmax(imp, batch, dim=0)
            degree = scatter_softmax(degree, batch, dim=0)
        else:
            imp = torch.sigmoid(imp)
            degree = torch.sigmoid(degree)
        if self.degree:
            imp = imp + degree
            self.printimp1 = imp - degree
        x = imp * x  # MxD
        self.printimp = imp
        return x


class _mean_max_sop(nn.Module):
    def __init__(self, hidden_dim, learn_t=False, learn_p=False, t=1.0, p=1.0, ttype="softmax", is_triu=False,
                 isqrt=False, num_iter=3):
        super(_mean_max_sop, self).__init__()
        self.hidden_dim = hidden_dim
        self.is_triu = is_triu
        self.learn_t = learn_t
        self.learn_p = learn_p
        self.isqrt = isqrt
        self.t = t
        self.tmp = t
        self.p = p
        self.ttype = ttype
        self.num_iter = num_iter
        if learn_t and ttype == 'softmax':
            if t < 1.0:
                c = torch.nn.Parameter(torch.Tensor([1 / t]), requires_grad=True)
                self.t = 1 / c
            else:
                self.t = torch.nn.Parameter(torch.Tensor([t]), requires_grad=True)
        else:
            self.t = t
        if learn_p and ttype == 'power':
            self.p = torch.nn.Parameter(torch.Tensor([p]), requires_grad=True)
        else:
            self.p = p

    def reset_parameters(self):
        if self.learn_t and self.ttype == 'softmax':
            if self.t < 1.0:
                c = torch.nn.Parameter(torch.Tensor([1 / self.tmp]), requires_grad=True)
                self.t = 1 / c
            else:
                self.t = torch.nn.Parameter(torch.Tensor([self.tmp]), requires_grad=True)

    def forward(self, x, count, mask):
        batch_size = x.size(0)
        N = x.size(1)
        embed_dim = x.size(2)
        wj = torch.einsum('pik,pkj -> pkij', x.transpose(2, 1), x)
        flatten_wj = wj.view(batch_size, N, -1)
        if self.ttype == "softmax":
            sf = torch.sum(F.softmax(flatten_wj * self.t, dim=1) * flatten_wj, dim=1)
        elif self.ttype == "power":
            min_value, max_value = 1e-7, 1e1
            sf = torch.clamp_(flatten_wj, min_value, max_value)
            sf = torch.pow(sf, self.p)
            sf = torch.sum(sf, dim=1) / sf.size(1)
            torch.clamp_(sf, min_value, max_value)
            sf = torch.pow(sf, 1 / self.p)
        output = sf
        output = output.view(batch_size, embed_dim, embed_dim)
        if self.isqrt:
            output = Sqrtm.apply(output, self.num_iter)
        if self.is_triu:
            output = Triuvec.apply(output)
        output = output.view(batch_size, -1)
        return output


class G2DeNet(nn.Module):
    def __init__(self):
        super(G2DeNet, self).__init__()

    def forward(self, x):
        batchSize, N, dim = x.size()
        A = torch.cat((torch.eye(dim, device=x.device), torch.zeros((1, dim), device=x.device)), dim=0).repeat(
            batchSize, 1, 1)
        # print(A)
        B = torch.zeros((dim + 1, dim + 1), device=x.device)
        B[-1, -1] = 1
        B = B.repeat(batchSize, 1, 1)
        b = torch.zeros((dim + 1, 1), device=x.device)
        b[-1, 0] = 1
        b = b.repeat(batchSize, 1, 1)
        one = torch.ones((N, 1), device=x.device).repeat(batchSize, 1, 1)
        Y = 1 / N * A.bmm(x.transpose(2, 1)).bmm(x).bmm(A.transpose(2, 1))
        former = A.bmm(x.transpose(2, 1)).bmm(one).bmm(b.transpose(2, 1))
        Y = Y + 1 / N * (former.transpose(2, 1) + former) + B
        return Y



