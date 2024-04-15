# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2021/10/11 12:46
# @Author  : dongZheX
# @Version : python3.7
# @Desc    : $END$
import argparse
import tools
import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from models.gcn import GCN
from models.graphsage import SAGE
from models.fog import FOGConv, FOGWithJK, FOGGNN, OGNN
import fitlog
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import to_undirected, add_self_loops
from logger import Logger
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument("--selected_id", type=int, default=0)
    tools.init_args(parser)
    args = parser.parse_args()
    args.fitlog = os.getcwd() + "/" + args.fitlog

    if not os.path.exists(args.fitlog):
        os.makedirs(args.fitlog)
    fitlog.set_log_dir(args.fitlog)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # print(args.gnn)
    if args.seed > 0:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    if args.add_self_loop:
        data.edge_index = add_self_loops(edge_index=data.edge_index)
    # if args.gnn != "fog":
    #     transform = T.ToSparseTensor()
    #     data = transform(data)
    args.input_channel = data.x.size(-1)
    out_channel = args.output_channel
    if args.mlp is None:
        args.mlp = [dataset.num_classes]
    else:
        args.mlp.append(dataset.num_classes)
    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    # print(data)
    model = None

    # if args.gnn == "fog" and args.jk == "None":
    #     model = FOG(num_layers=args.num_layers, input_channel=data.num_features, embed_dim=args.embed_dim,
    #                 hidden_channel1=args.hidden_channel1,hidden_channel2=args.hidden_channel2,
    #                 num_tasks=dataset.num_tasks, bias=args.bias, dropout=args.dropout).to(device)
    if args.gnn == "fog":
        model = FOGWithJK(args).to(device)
    if args.gnn == "foggnn":
        model = FOGGNN(args).to(device)
    if args.gnn == "ognn":
        model = OGNN(args).to(device)
    elif args.gnn == "gcn":
        model = GCN(data.num_features, args.embed_dim,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
    elif args.gnn == "sage":
        model = SAGE(data.num_features, args.embed_dim,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)


    for run in range(args.runs):
        model.reset_parameters()
        optimizer = tools.get_optimizer(model, args)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size,
                                                    gamma=args.lr_decay_factor)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            scheduler.step()
            result = test(model, data, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            logger.add_result(run, result)
            if args.selected_id == run:
                fitlog.add_loss(loss, name="Loss", step=epoch)
                fitlog.add_metric({"TRAIN_" + dataset.eval_metric.upper(): train_acc}, step=epoch)
                fitlog.add_metric({"VAL_" + dataset.eval_metric.upper(): valid_acc}, step=epoch)
                fitlog.add_metric({"TEST_" + dataset.eval_metric.upper(): test_acc}, step=epoch)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
            logger.add_result(run, result)
        logger.print_statistics(run=run, args=args)
        # save_model
    logger.print_statistics(args=args)
if __name__ == "__main__":
    main()