# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2021/10/31 13:41
# @Author  : dongZheX
# @Version : python3.7
# @Desc    : $END$
import random
import os
import numpy as np
import torch
import argparse
import sys
import torch
import traceback
import shutil
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from models.gnn import GNN
import os
from tqdm import tqdm
import argparse
import time
from argparse import Namespace
import random
import sop_tools
import numpy as np
import fitlog
from matplotlib import pyplot as plt
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import json


def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


args = json.load(open("configs/ogbg-molhiv-ccp-sr1-128-2-32-gin.json"), object_hook=lambda d: Namespace(**d))
args.pool_method = "sop_att_m"
args.degree_3 = False
args.num_heads = 4
args.reduction_dim = [128]
args.mask = False
args.degree_1 = False
args.degree_2 = False
args.moredata = False
args.isqrt = True
set_seed(1)
dataset = PygGraphPropPredDataset(name=args.dataset)

data = dataset
model = GNN(args, dataset).to(torch.device("cuda:0"))
loader = DataLoader(data,batch_size=2,shuffle=False,num_workers=0)
for step, batch in enumerate(loader):
    batch = batch.to(torch.device("cuda:0"))
    print(model(batch))
