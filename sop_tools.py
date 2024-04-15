'''
@author: dongZheX
@description: these tools help your to build second order pooling
'''
import torch
import torch.optim as optim
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, global_sort_pool
from models.sopool import Readout


from models.mlp import MLP
# the args sop need

seq = {"_isqrt", "sop", "sop_max", "sop_mean", "sop_att", "sop_topk", "sop_att_triu", "sop_mean_max", "g2denet", "isqrt", "sop_att_d", "sop_att_n" ,"n_sop_att","sop_att_f","sop_att_u","sop_att_f"}
def sop_args(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--pool_method', type=str, default="sop", help='the type of global pooling method')
    parser.add_argument('--reduction_dim', type=int, nargs='+')
    parser.add_argument('--weight_decay', type=int, default=0,
                        help="the weight_decay parameters in torch.optim.optimizer")
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate during the training proccess")
    parser.add_argument('--final_dropout_ratio', type=float, default=0,
                        help="the dropout_ratio after graph pooling method")
    parser.add_argument('--k', type=int, default=16, help="the k of sop_topk pooling method")
    parser.add_argument('--learn_t', action="store_true")
    parser.add_argument('--learn_p', action="store_true")
    parser.add_argument('--mean_max_type', type=str, default="softmax", help="softmax or power")
    parser.add_argument('--padding', action="store_true",
                        help="if use padding, the training will be speed up, but loss acc")
    parser.add_argument('--optimizer', type=str, default="adam", help="the type of optimizer")
    parser.add_argument('--use_residual', action="store_true", help="whether use residual structure")
    parser.add_argument('--jk', type=str, default="last", help="whether use jump knowledge")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="if you use momentum, you can set the value of momentum")
    parser.add_argument('--lr_scheduler', type=float, nargs='+')
    parser.add_argument('--is_triu', action="store_true", help="whether use triu dimension reduction")
    parser.add_argument('--num_iter', type=int, default=3,
                        help="if you use isqrt operator, you can set the num of iter")
    parser.add_argument('--fs', type=int, default=0, help="if add a first order information into second order information")
    # 1 : concat
    # 2 : after graph_pred concat
    # 3: after graph_pred concat and attention/ two a
    parser.add_argument('--fp', type=int, default=0, help="if fs =2 / fs=1")
    # 0 : after graph_pred
    # 1 : after log_softmax 
    parser.add_argument('--fs_imp', nargs="+", type=float)
    parser.add_argument('--fs_learn_att', action="store_true")
    parser.add_argument('--fs_dim', type=int, default=128, help="if add a first order information into second order information")
    parser.add_argument('--rdt', type=int, default=0,
                        help="the position of reduction_dim_layer, and whether use batchnorm")
    # 0:no batchnorm, after padding;
    # 1:no batchnorm, before padding;
    # 2:with batchnorm, after padding;
    # 3:witch batchnorm, before padding
    parser.add_argument('--isqrt', action="store_true",
                        help="whether use isqrt to sop++")
    parser.add_argument('--maskc', action="store_true")
    parser.add_argument('--sopattsigmoid', action="store_true")
    # about flag
    parser.add_argument('--attack', type=str, default="None",
                        help="whether use flag to data argument")
    parser.add_argument('--step_size', type=float,
                        help="about flag")
    parser.add_argument('-m', type=int, default=3) 
    parser.add_argument('--fdr', type=float, default=0) 
    parser.add_argument('--fuse_list', type=int, nargs="+")  
    parser.add_argument('--fuse_sop_list', nargs="+")
    parser.add_argument('--fuse_fs_list', type=int, nargs="+")
    parser.add_argument('--fuse_imp_list', type=int, nargs="+")
    parser.add_argument('--tell_me', type=str, default="None")
    parser.add_argument('--select', type=str, nargs='+')
    parser.add_argument('--DaP', type=float, nargs='+')
    parser.add_argument('--DaP_mode', type=str, default="avg")
    parser.add_argument('--vp', type=float, default=1)
    parser.add_argument('--phi', type=float, default=0)
    parser.add_argument('--gpmlp', type=int, nargs='+')
    parser.add_argument('--thr', type=float, default=-1)
    parser.add_argument('--degree', action="store_true")
    parser.add_argument('--loss', type=str, default="normal")
    parser.add_argument('--cb', type=str, nargs="+")
    parser.add_argument('--focal', type=float, nargs="+")
    parser.add_argument('--gpi', action="store_true")
    parser.add_argument('--sr', type=float, default=1)
    parser.add_argument('--fix_triu', action="store_true")
    parser.add_argument('--one_init', action="store_true")
    parser.add_argument('--pretrain', action="store_true")
    parser.add_argument('--fixed_time', type=int, default=10)
    parser.add_argument('--model_path', type=str, default="pretrain/pretrain.pth")
# optimizer
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

# get pool layer
def get_global_pool(args, emb_dim):
    pool = None
    if args.pool_method == "sum":
        # self.pool = global_add_pool
        pool = global_add_pool
    elif args.pool_method == "mean":
        pool = global_mean_pool
    elif args.pool_method == "sort":
        pool = global_sort_pool
        emb_dim = emb_dim * args.k
    elif args.pool_method == "max":
        pool = global_max_pool
    elif args.pool_method == "attention":
        pool = GlobalAttention(
            gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                        torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
    elif args.pool_method == "set2set":
        pool = Set2Set(emb_dim, processing_steps=2)
    elif args.pool_method in sep or args.pool_method == "sop_att_m":
        if args.padding:
            pool = _SecondOrderPooling(operator=args.pool_method, hidden_dim=emb_dim, new_dim = args.reduction_dim,
                                            is_triu=args.is_triu, dropout_ratio=args.final_dropout_ratio, k=args.k,
                                            learn_t=args.learn_t, num_iter=args.num_iter, rdt=args.rdt,
                                            isqrt=args.isqrt, args=args)
        else:
            pool = SecondOrderPooling(operator=args.pool_method, hidden_dim=emb_dim, new_dim=args.reduction_dim,
                                           is_triu=args.is_triu, dropout_ratio=args.final_dropout_ratio, k=args.k,
                                           learn_t=args.learn_t, num_iter=args.num_iter)
        if args.reduction_dim is not None:
            emb_dim = args.reduction_dim[-1]
    return emb_dim, pool


def get_pred_layer(args, tmp_dim, emb_dim, num_tasks):
    graph_pred_linear = None
    hidden = []
    if args.pool_method == "g2denet":
        emb_dim = emb_dim + 1

    if args.pool_method == "set2set":
        if args.gpmlp is not None:
            hidden.extend(args.gpmlp)
            hidden.append(num_tasks)
            graph_pred_linear = MLP(2 * emb_dim, hidden)
        else:
            graph_pred_linear = torch.nn.Linear(2 * emb_dim, num_tasks)
    elif args.pool_method == "sop_att_u":
        if args.gpmlp is not None:
            hidden.extend(args.gpmlp)
            hidden.append(num_tasks)
            graph_pred_linear = MLP(emb_dim, hidden)
        else:
            graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
    elif args.pool_method in sep and args.is_triu:
        if args.fs == 1:
            if args.gpmlp is not None:
                hidden.extend(args.gpmlp)
                hidden.append(num_tasks)
                graph_pred_linear = MLP(int(emb_dim * (emb_dim + 1) / 2 + tmp_dim), hidden)

            else:
                graph_pred_linear = torch.nn.Linear(int(emb_dim * (emb_dim + 1) / 2 + tmp_dim), num_tasks)
        # elif args.add_first == 2:
        #     self.graph_pred_linear = torch.nn.Linear(args.fs_dim, self.num_tasks)
        else:
            if args.gpmlp is not None:
                hidden.extend(args.gpmlp)
                hidden.append(num_tasks)
                graph_pred_linear = MLP(int(emb_dim * (emb_dim + 1) / 2), hidden)
            else:
                graph_pred_linear = torch.nn.Linear(int(emb_dim * (emb_dim + 1) / 2), num_tasks)
    elif args.pool_method in sep and not args.is_triu:
        if args.fs == 1:
            if args.gpmlp is not None:
                hidden.extend(args.gpmlp)
                hidden.append(num_tasks)
                graph_pred_linear = MLP(emb_dim * emb_dim + tmp_dim, hidden)
            else:
                graph_pred_linear = torch.nn.Linear(emb_dim * emb_dim + tmp_dim, num_tasks)
        # elif args.add_first == 2:
        #     self.graph_pred_linear = torch.nn.Linear(args.fs_dim, self.num_tasks)
        else:
            if args.gpmlp is not None:
                hidden.extend(args.gpmlp)
                hidden.append(num_tasks)
                graph_pred_linear = MLP(emb_dim * emb_dim, hidden)
            else:
                graph_pred_linear = torch.nn.Linear(emb_dim * emb_dim, num_tasks)
    # elif graph_pooling == "g2denet":
    #     self.graph_pred_linear = torch.nn.Linear((self.emb_dim+1) * (self.emb_dim+1), self.num_tasks)
    else:
        if args.gpmlp is not None:
            hidden.extend(args.gpmlp)
            hidden.append(num_tasks)
            graph_pred_linear = MLP(emb_dim, hidden)
        else:
            graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)

    return graph_pred_linear
