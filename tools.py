# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2021/10/11 12:57
# @Author  : dongZheX
# @Version : python3.7
# @Desc    : $END$

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

def init_args(parser):

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=250)
    parser.add_argument('--hidden_channel1', type=int, default=16)
    parser.add_argument('--hidden_channel2', type=int, default=16)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--fitlog', type=str, default='logs/')
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--jk', type=str, default="last", help="jump knowledge")
    parser.add_argument('--add_self_loop', action="store_true")
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--heads', type=int, nargs='+')
    parser.add_argument('--mlp', type=int, nargs='+')
    parser.add_argument('--output_channel', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--use_residual', action="store_true")
    parser.add_argument('--gru_layers', type=int, default=3)
    parser.add_argument('--attach_gnn', type=str, default='gcn')
    parser.add_argument('--seed', type=int, default=0)
    ## above is basis settings

def get_optimizer(model, args):
    optimizer = None
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer