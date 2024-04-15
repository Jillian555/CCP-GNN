# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2021/10/23 15:15
# @Author  : dongZheX
# @Version : python3.7
# @Desc    : $END$
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import sys
import torch
import traceback
import shutil
import torch_geometric
from torch_scatter import scatter_add
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
import json
from torch_sparse import spmm
cls_criterion = torch.nn.BCEWithLogitsLoss()
bce = torch.nn.BCELoss()
reg_criterion = torch.nn.MSELoss()
torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=10000)



def train(model, device, loader, optimizer, task_type, args):
    model.train()
    avg_loss = 0
    num = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        num = num + 1
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            if args.dataset=='ogbg-molpcba':
                is_labeled = batch.y == batch.y
            else:
                is_labeled = (batch.y[:,0] == batch.y[:,0]).reshape(-1,1)


            if "classification" in task_type:
                if args.fp > 0:
                    loss = bce(pred.to(torch.float32)[is_labeled], batch.y[:,0].unsqueeze(-1).to(torch.float32)[is_labeled])
                else:

                    loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            if args.pool_method == "sop_att" and args.smooth > 0:
                fx = model.readout.sop_att.printimp 
                edge_index = batch.edge_index
                row, col = edge_index[0], edge_index[1]
                deg = torch_geometric.utils.degree(row, num_nodes=fx.shape[0]).reshape(-1,1)
                # deg_inv_sqrt = deg.pow_(-0.5)
                deg.masked_fill_(deg == float('inf'), 0)
                loss_smooth = fx*deg - spmm(edge_index, torch.ones(row.shape[0]).to(device), fx.shape[0], fx.shape[0], fx)
                loss_smooth =  args.smooth * torch.norm(loss_smooth)
                # matmul()
                avg_loss += loss_smooth.item()
                loss += loss_smooth
            avg_loss += loss.item()
            
            loss.backward()
            optimizer.step()
    return avg_loss / num


def eval(model, device, loader, evaluator, args):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            if args.dataset=='ogbg-molpcba':
                y_true.append(batch.y.detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                batch.y = batch.y[:,0].unsqueeze(-1)
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def main():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--config', type=str,help='which configs to use', default="")
    parser.add_argument('--device', type=str, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--moredata", action="store_true")
    parser.add_argument('--gnn', type=str,help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float,help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int,help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int,help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, help='dataset name (default: ogbg-ppa)')
    parser.add_argument('--filename', type=str, help='filename to output result (default: )')
    parser.add_argument('--fitlog', type=str)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--choice', type=int, default=0)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_scheduler', type=float, nargs='+')
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--use_residual', action='store_true')
    parser.add_argument('--jk', type=str)
    parser.add_argument('--fdr', type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--pool_method', type=str)
    parser.add_argument('--reduction_dim', type=int, nargs='+')
    parser.add_argument('--final_dropout_ratio', type=float)
    parser.add_argument('--isqrt', action='store_true')
    parser.add_argument('--k', type=int)
    parser.add_argument('--learn_t', action='store_true')
    parser.add_argument('--learn_p', action='store_true')
    parser.add_argument('--mean_max_type', type=str)
    parser.add_argument('--num_iter', type=int)
    parser.add_argument('--is_triu', action='store_true')
    parser.add_argument('--sopattsigmoid', action='store_true')
    parser.add_argument('--vp', type=float)
    parser.add_argument('--fix_triu', action='store_true')
    parser.add_argument('--fix_time', type=int, default=1)
    parser.add_argument('--degree', action='store_true')
    parser.add_argument('--pred_norm', action='store_true')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--sr', type=float)
    parser.add_argument('--graph_pred_list', type=int, nargs='+')
    parser.add_argument('--degree_1', action="store_true")
    parser.add_argument('--degree_2', action="store_true")
    parser.add_argument('--degree_3', action="store_true")
    parser.add_argument('--mask', type=float, default=1)
    parser.add_argument('--smooth', type=float, default=0)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--fp', type=float, default=0)
    _args = parser.parse_args()
    device = torch.device("cuda:" + str(_args.device)) if torch.cuda.is_available() else torch.device("cpu")  
    resume = _args.resume
    if resume != "":
        print("Resuming...........")
        args = json.load(open(resume+"/hyper.log"))['hyper']
        args = argparse.Namespace(**args)
        args.dataset = args.dataset.replace("_","-")
        if args.reduction_dim is not None:
            args.reduction_dim = [int(v) for v in args.reduction_dim[1:-1].split(", ")]
        if args.lr_scheduler is not None:
            args.lr_scheduler = [float(v) for v in args.lr_scheduler[1:-1].split(", ")]
        fitlog.set_log_dir(resume)
        args.pretrain = False
    else:
        args = json.load(open(_args.config), object_hook=lambda d: Namespace(**d))
        for arg in vars(_args):
            if getattr(_args, arg) is True:
                setattr(args, arg, True)
                continue
            if getattr(args, arg, "No this arg") == "No this arg":
                setattr(args, arg, getattr(_args, arg))
            if getattr(_args, arg) is not None and getattr(_args, arg) is not False:
                setattr(args, arg,  getattr(_args, arg))
        args.fitlog = os.getcwd() + "/" + args.fitlog
        if not os.path.exists(args.fitlog):
            os.makedirs(args.fitlog)
        fitlog.set_log_dir(args.fitlog)
        if args.epochs == 1 and resume != "":
            fitlog.create_log_folder()
    set_seed(args.seed)  

    dataset = PygGraphPropPredDataset(name=args.dataset)
    if args.fp > 0:
        fp_feat = np.load('rf_preds_hiv/rf_final_pred.npy')
        dataset.data.y = torch.cat((dataset.data.y, torch.from_numpy(fp_feat)), 1)
    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
    split_idx = dataset.get_idx_split()
    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    model = GNN(args, dataset).to(device)
    optimizer = sop_tools.get_optimizer(model, args)
    scheduler = None
    if args.lr_scheduler is not None:
        if len(args.lr_scheduler) == 2:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_scheduler[0]),
                                                  gamma=args.lr_scheduler[1])
    start_epoch = 0
    if resume != "":
        pth = torch.load(resume + "/checkpoint.pth")
        model.load_state_dict(pth['net'])
        optimizer.load_state_dict(pth['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(pth['scheduler'])
        start_epoch = pth['epoch']
    valid_curve = []
    test_curve = []
    train_curve = []
    loss_curve = []
    max_val = 0
    have_save = False
    fitlog.add_hyper(args)
    for epoch in range(start_epoch+1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        if epoch == args.fix_time:
            model.open_grad()
        if epoch == args.epochs:
            args.redis = 1
       
        loss = train(model, device, train_loader, optimizer, dataset.task_type, args)
    
        fitlog.add_loss(loss, name="Loss", step=epoch)
        if args.lr_scheduler is not None:
            if len(args.lr_scheduler) == 2:
                scheduler.step()
            elif len(args.lr_scheduler) == 3:
                if epoch % int(args.lr_scheduler[0]) == 0:
                    for p in optimizer.param_groups:
                        if p['lr'] > args.lr_scheduler[1]:
                            p['lr'] -= args.lr_scheduler[1]

        train_perf = eval(model, device, train_loader, evaluator, args)
        valid_perf = eval(model, device, valid_loader, evaluator, args)
        test_perf = eval(model, device, test_loader, evaluator, args)
        if max_val < valid_perf[dataset.eval_metric]:
            # have_save = True
            save_pth = dict()
            save_pth['net'] = model.state_dict()
            save_pth['optimizer'] = optimizer.state_dict()
            save_pth['scheduler'] = scheduler.state_dict() if scheduler is not None else None
            save_pth['epoch'] = epoch
            torch.save(save_pth, args.fitlog + fitlog.get_log_folder() + "/pretrain.pth")
            max_val = valid_perf[dataset.eval_metric]
        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        loss_curve.append(loss)
        fitlog.add_metric({"TRAIN_" + dataset.eval_metric.upper(): train_perf[dataset.eval_metric]}, step=epoch)
        fitlog.add_metric({"VAL_" + dataset.eval_metric.upper(): valid_perf[dataset.eval_metric]}, step=epoch)
        fitlog.add_metric({"TEST_" + dataset.eval_metric.upper(): test_perf[dataset.eval_metric]}, step=epoch)
    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    fitlog.add_best_metric({"BEST_TRAIN_" + dataset.eval_metric.upper(): best_train})
    fitlog.add_best_metric({"BEST_VAL_" + dataset.eval_metric.upper(): valid_curve[best_val_epoch]})
    fitlog.add_best_metric({"BEST_TEST_" + dataset.eval_metric.upper(): test_curve[best_val_epoch]})
    fitlog.finish()


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


if __name__ == "__main__":
    main()
