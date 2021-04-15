import os
import os.path as osp
import time
import math
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.sparse as ts

import torch_geometric.utils.num_nodes as geo_num_nodes
import torch_geometric.transforms as T
from torch_geometric.utils import dense_to_sparse, degree
from torch_geometric.datasets import Planetoid, TUDataset, Flickr, Coauthor, CitationFull
from torch_geometric.data import Data, ClusterData, ClusterLoader
from torch_geometric.utils import to_dense_adj

from utils import *
from scipy import sparse, io
from get_partition import my_partition_graph
from get_boundary import my_get_boundary
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix, tril
# from network import GCN, GAT
from models import SAGE
from sampler import NeighborSampler
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--iteration', type=int, default=1)
parser.add_argument('--times', type=int, default=4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--draw', type=int, default=100)
parser.add_argument('--ratio_graph', type=int, default=10)
parser.add_argument('--model', type=str, default='GCN', choices=['GraphSAGE'])
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--dataset', type=str, default="CiteSeer", choices=['Cora', 'CiteSeer', 'Pubmed'])
parser.add_argument('--num_groups', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--total_subgraphs', type=int, default=12)
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:8'])
parser.add_argument('--save_prefix', type=str, default='./graph_tune')
parser.add_argument('--hard', action='store_true', default=False)
parser.add_argument('--rho', type=float, default=1e-3, help='coefficience of sparse regularization')
parser.add_argument('--phi', type=float, default=10, help='coefficience of diagonalization regularization')
parser.add_argument('--repeat', type=int, default=5, help='repeat run 5 times for default')
parser.add_argument('--quant', action='store_true', default=False)
parser.add_argument('--num_act_bits', type=int, default=32, help='will quantize node features if enable')
parser.add_argument('--num_wei_bits', type=int, default=32, help='will quantize weights if enable')
parser.add_argument('--num_agg_bits', type=int, default=32, help='will quantize aggregation if enable')
parser.add_argument('--num_att_bits', type=int, default=32, help='will quantize attention module if enable')
parser.add_argument('--enable_chunk_q', action='store_true', default=False, help='enable chunk based quantization')
parser.add_argument('--enable_chunk_q_mix', action='store_true', default=False, help='enable mixed precision chunk based quantization')
parser.add_argument('--q_max', type=int, default=4)
parser.add_argument('--q_min', type=int, default=2)
parser.add_argument('--chunk_min', type=int, default=40)
parser.add_argument('--lookback', type=int, default=3)
args = parser.parse_args()

device = torch.device(args.device)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)

chunk_min = args.chunk_min

# create logging file
global save_path
if args.quant is False:
    save_path = os.path.join(args.save_prefix, '{}_{}'.format(args.model, args.dataset))
else:
    save_path = os.path.join(args.save_prefix, '{}_{}_ACT_{}-bit_WEI_{}-bit_AGG_{}-bit'.format(args.model, args.dataset, args.num_act_bits, args.num_wei_bits, args.num_agg_bits))
if not os.path.exists(save_path):
    os.makedirs(save_path)
args.logger_file = os.path.join(save_path, 'log_tune_{}_{}_{}.txt'.format(args.num_groups, args.num_classes, args.total_subgraphs))
if os.path.exists(args.logger_file):
    os.remove(args.logger_file)
handlers = [logging.FileHandler(args.logger_file, mode='w'),
            logging.StreamHandler()]
logging.basicConfig(level=logging.INFO,
                    datefmt='%m-%d-%y %H:%M',
                    format='%(asctime)s:%(message)s',
                    handlers=handlers)

logging.info('device: {}'.format(args.device))
logging.info('start tuning {}'.format(args.model))

# load dataset
if args.dataset in ['Cora', 'CiteSeer', 'Pubmed']:
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
else:
    print('please check the dataset info!')
    exit()
data = dataset[0]
print('len of dataset: {}'.format(len(dataset)))

# get parition info
# if args.dataset == 'CiteSeer':
#     degree_split = [3, 6] # for num_class = 3
# elif args.dataset == 'Cora':
#     degree_split = [4]
# elif args.dataset == 'Pubmed':
#     degree_split = [7, 12]
# _, n_subgraphs, class_graphs = my_partition_graph(data, degree_split, args.total_subgraphs, args.num_groups, dataset=args.dataset)
# n_subgraphs, n_classes, n_groups = my_get_boundary(n_subgraphs, class_graphs, args.num_groups)

# load dataset and model from pretrain checkpoints
if args.quant is False:
    load_path = os.path.join('./pretrain_partition', '{}_{}'.format(args.model, args.dataset))
else:
    load_path = os.path.join('./pretrain_partition', '{}_{}_ACT_{}-bit_WEI_{}-bit_AGG_{}-bit'.format(args.model, args.dataset, args.num_act_bits, args.num_wei_bits, args.num_agg_bits))
checkpoint  = torch.load(load_path + '/ckpt.pth.tar', map_location='cpu')
state_dict  = checkpoint['state_dict']
n_subgraphs = checkpoint['n_subgraphs']
n_classes   = checkpoint['n_classes']
n_groups    = checkpoint['n_groups']
data = checkpoint['data'].to(device)
adj  = checkpoint['adj'].to(device)

model = SAGE(dataset.num_features, 32, dataset.num_classes, data=data, device=device, quant=args.quant,
                num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits,
                chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min)

model.load_state_dict(state_dict)
model.to(device)
data.to(device)
logging.info('successfully load both data and model !')

adj_size = data.x.size(0)
ori_nnz = np.count_nonzero(adj.cpu().to_dense().numpy())


# hepler functions
def update_gradients_adj(grads_vars, adj_mask):
    temp_grad_adj1 = 0
    var1 = None
    var2 = None
    temp_grad_adj2 = 0
    for key,var in grads_vars.items():
        grad = var.grad
        if key == "support1":
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = temp_grad_adj.t_()
            temp_grad_adj1 = temp_grad_adj + transposed_temp_grad_adj
            var1 = var
        if key == "support2":
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = temp_grad_adj.t_()
            temp_grad_adj2 = temp_grad_adj + transposed_temp_grad_adj
            var2 = var
    grad_adj = (temp_grad_adj1 + temp_grad_adj2) / 4 # Why are we doing this?
    var1.grad = grad_adj
    var2.grad = grad_adj
    return [var1,var2]

def prune_adj(oriadj, non_zero_idx:int, percent:int):
    original_prune_num = int(((non_zero_idx - oriadj.size()[0]) / 2) * (percent / 100)) # how many to be pruned
    adj = SparseTensor.from_torch_sparse_coo_tensor(oriadj).to_scipy()

    # find the lower half of the matrix
    low_adj = tril(adj, -1)
    non_zero_low_adj = low_adj.data[low_adj.data != 0]

    low_pcen = np.percentile(abs(non_zero_low_adj), percent) # threshold
    under_threshold = abs(low_adj.data) < low_pcen
    before = len(non_zero_low_adj)
    low_adj.data[under_threshold] = 0
    non_zero_low_adj = low_adj.data[low_adj.data != 0]
    after = len(non_zero_low_adj)

    rest_pruned = original_prune_num - (before - after)
    if rest_pruned > 0:
        mask_low_adj = (low_adj.data != 0)
        low_adj.data[low_adj.data == 0] = 2000000
        flat_indices = np.argpartition(low_adj.data, rest_pruned - 1)[:rest_pruned]
        low_adj.data = np.multiply(low_adj.data, mask_low_adj)
        low_adj.data[flat_indices] = 0
    low_adj.eliminate_zeros()
    new_adj = low_adj + low_adj.transpose()
    new_adj = new_adj + sparse.eye(new_adj.shape[0])
    return SparseTensor.from_scipy(new_adj).to_torch_sparse_coo_tensor().to(device)

def get_mask(oriadj, non_zero_idx:int, percent:int):
    original_prune_num = int(((non_zero_idx - oriadj.size()[0]) / 2) * (percent / 100))
    adj = SparseTensor.from_torch_sparse_coo_tensor(oriadj).to_scipy()

    # find the lower half of the matrix
    low_adj = tril(adj, -1)
    non_zero_low_adj = low_adj.data[low_adj.data != 0]

    low_pcen = np.percentile(abs(non_zero_low_adj), percent)
    under_threshold = abs(low_adj.data) < low_pcen
    before = len(non_zero_low_adj)
    low_adj.data[under_threshold] = 0
    non_zero_low_adj = low_adj.data[low_adj.data != 0]
    after = len(non_zero_low_adj)

    rest_pruned = original_prune_num - (before - after)
    if rest_pruned > 0:
        mask_low_adj = (low_adj.data != 0)
        low_adj.data[low_adj.data == 0] = 2000000
        flat_indices = np.argpartition(low_adj.data, rest_pruned - 1)[:rest_pruned]
        low_adj.data = np.multiply(low_adj.data, mask_low_adj)
        low_adj.data[flat_indices] = 0
    low_adj.eliminate_zeros()
    new_adj = low_adj + low_adj.transpose()
    new_adj = new_adj + sparse.eye(new_adj.shape[0])
    return SparseTensor.from_scipy((new_adj != adj)).to_torch_sparse_coo_tensor().int()

def calc_dist(m1,m2):
    diff = m1 - m2
    neg = diff < 0
    diff[neg] = -diff[neg]
    return torch.sum(diff.coalesce().values())

def count_group_nnz(adj, adj_size, group):
    _adj = SparseTensor.from_torch_sparse_coo_tensor(adj)
    row, col, value = _adj.coo()
    group_nnz = np.zeros((group, group))
    for i in range(len(value)):
        x = math.floor(row[i] / math.ceil(adj_size / group))
        y = math.floor(col[i] / math.ceil(adj_size / group))
        # print(row[i], col[i])
        # print(x, y)
        group_nnz[x][y] += 1
    return group_nnz

def identify_group(row, col, idx_list):
    x = 0
    y = 0
    for i in range(len(idx_list)):
        if i == 0:
            if row < idx_list[i]:
                x = i
            if col < idx_list[i]:
                y = i
        else:
            if row >= idx_list[i-1] and row < idx_list[i]:
                x = i
            if col >= idx_list[i-1] and col < idx_list[i]:
                y = i
    return x, y

def remove_boundary_nodes_among_groups(edge_index, adj_size, n_groups, adj=None):
    row = edge_index[0]
    col = edge_index[1]
    # boundary = n_groups[1] - n_groups[0] + 1
    boundary = n_groups[0]

    remove_list = []
    reserver_list = []
    cnt = 0
    for i in range(len(row)):
        if row[i] < boundary:
            x = 0
        else:
            x = 1
        if col[i] < boundary:
            y = 0
        else:
            y = 1
        if x + y == 1:
            remove_list.append(i)
        else:
            reserver_list.append(i)
            if x == 0 and y == 0:
                cnt += 1

    # counter = 0
    # for i in range(len(remove_list)):
    #     item = remove_list[i] - counter
    #     row = torch.cat([row[:item], row[item+1:]])
    #     col = torch.cat([col[:item], col[item+1:]])
    #     counter += 1
    row = row[reserver_list]
    col = col[reserver_list]
    edge_index = torch.stack([row, col])
    if adj is not None:
        adj = SparseTensor.from_torch_sparse_coo_tensor(adj).to_dense()
        edge_weight = adj[row,col].cpu().float()
        # print(row.shape)
        # print(edge_weight.shape)
        new_adj = SparseTensor(row=row, col=col, value=torch.clone(edge_weight)).to_torch_sparse_coo_tensor().to(args.device)
        return edge_index, new_adj, cnt
    else:
        return edge_index, cnt

def count_subgraph_nnz(edge_index, n_subgraphs):
    idx_subgraphs = []
    for i in range(len(n_subgraphs)):
        idx_subgraphs.append(sum(n_subgraphs[:i+1]))

    row = edge_index[0]
    col = edge_index[1]

    subgraph_nnz = np.zeros((len(n_subgraphs), len(n_subgraphs)))
    for i in range(len(row)):
        x, y = identify_group(row[i], col[i], idx_subgraphs)
        subgraph_nnz[x][y] += 1
    return subgraph_nnz

def remove_chunks(edge_index, adj_size, n_subgraphs, pre_subgraph_nnz):
    row = edge_index[0]
    col = edge_index[1]

    idx_subgraphs = []
    for i in range(len(n_subgraphs)):
        idx_subgraphs.append(sum(n_subgraphs[:i+1]))

    remove_list = []
    reserver_list = []
    for i in range(len(row)):
        x, y = identify_group(row[i], col[i], idx_subgraphs)
        if pre_subgraph_nnz[x][y] < chunk_min:
            remove_list.append(i)
        else:
            reserver_list.append(i)

    row = row[reserver_list]
    col = col[reserver_list]
    edge_index = torch.stack([row, col])

    return edge_index

def main_tune(data, model, device, iteration=0):

    _adj = SparseTensor.from_torch_sparse_coo_tensor(model.adj1.cpu())
    row, col, value = _adj.coo()
    edge_index = torch.stack([row, col])

    subgraph_nnz = count_subgraph_nnz(edge_index, n_subgraphs)
    logging.info('subgraph_nnz before tuning: \n {}'.format(subgraph_nnz.astype(int)))

    logging.info('dataset information: {}'.format(data))
    logging.info('n_subgraphs: {}'.format(n_subgraphs))
    logging.info('n_classes  : {}'.format(n_classes))
    logging.info('n_groups   : {}'.format(n_groups))

    # Optimize on the graph
    model.adj1.requires_grad = True
    model.adj2.requires_grad = True

    adj_optimizer = torch.optim.SGD([model.adj1, model.adj2], lr=0.01)
    weight_optimizer = torch.optim.Adam(model.parameters(), lr= 0.01, weight_decay=5e-4) # 0.01+citeseer-norm+hidden256=0.62

    support1, support2 = model.adj1, model.adj2
    partial_adj_mask = support1.clone().detach()
    rho = 0
    non_zero_idx = SparseTensor.from_torch_sparse_coo_tensor(support1.detach()).nnz()
    print("non-zero indices", non_zero_idx)
    Z1 = U1 = Z2 = U2 = partial_adj_mask.clone().detach()
    adj_mask = partial_adj_mask.clone()
    # id1 = id2 = model.id
    id1 = model.id.to(device)
    id2 = model.id.to(device)

    x = data.x.to(device)
    y = data.y.squeeze().to(device)

    non_zero_idx = SparseTensor.from_torch_sparse_coo_tensor(model.adj1).nnz()
    group_nnz_bef = count_group_nnz(support1, adj_size, group=args.num_groups)

    _adj = SparseTensor.from_torch_sparse_coo_tensor(support1)
    row, col, value = _adj.coo()
    diagonalization = torch.sum(abs(row - col) * value) / len(row)

    print('U1 : ', U1)
    print('Z1 : ', Z1)
    print('id1: ', id1)

    # d1 = support1 + U1 - (Z1 + id1)
    # d2 = support2 + U2 - (Z2 + id2)
    # admm_loss = lambda m: loss(m) + \
    #             args.rho * (torch.sum(d1.coalesce().values() * d1.coalesce().values()) + \
    #             torch.sum(d2.coalesce().values()*d2.coalesce().values())) + \
    #             args.phi * diagonalization

    logging.info('finish initialize the loss and optimizer for graph.')


    # graph optimization

    ## --------------------------------------
    ## Graph Sparsification
    ## --------------------------------------
    logging.info('***'*20)
    logging.info('optimization at iteration {}'.format(iteration))
    logging.info('***'*20)

    best_prune_acc = 0
    for j in range(args.times):
        for epoch in range(args.epochs):

            # update graph optimizer
            loss, acc = train(model, epoch, adj_optimizer, x, y, support1, U1, Z1, support2, U2, Z2, adjOpt=True, adj_mask=adj_mask)
            train_acc, val_acc, test_acc = test(model, data, x, y, batch_size=64)
            if test_acc > best_prune_acc:
                best_prune_acc = test_acc

            logging.info("Pruning Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(j*args.epochs + epoch, train_acc, val_acc, test_acc))

        # update U,Z
        adj1,adj2 = model.adj1, model.adj2
        Z1 = adj1 - id1 + U1
        Z1 = prune_adj(Z1,non_zero_idx,args.ratio_graph) - id1
        U1 = U1 + (adj1 - id1 - Z1)
        Z2 = adj2 - id2 + U2
        Z2 = prune_adj(Z2,non_zero_idx,args.ratio_graph) - id2
        U2 = U2 + (adj2 - id2 - Z2)

    # prune graph
    adj1,adj2 = model.adj1, model.adj2
    adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
    adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
    model.adj1 = adj1
    model.adj2 = adj2

    group_nnz_aft = count_group_nnz(model.adj1, adj_size, group=args.num_groups)
    logging.info('group_nnz before pruning: \n {}'.format(group_nnz_bef))
    logging.info('group_nnz after  pruning: \n {}'.format(group_nnz_aft))

    scipy_adj = SparseTensor.from_torch_sparse_coo_tensor(model.adj1).to_scipy()
    io.mmwrite(save_path + '/{}_newadj.mtx'.format(args.dataset), scipy_adj)
    logging.info('saving new adj. matrix ...')

    train_acc, val_acc, test_acc = test(model, data, x, y)
    logging.info('After tuning results:')
    logging.info('Pruning ratio: {}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
                  args.ratio_graph, train_acc, val_acc, test_acc))

    cur_adj1 = model.adj1.cpu().to_dense().numpy()
    cur_adj2 = model.adj2.cpu().to_dense().numpy()
    logging.info("original number of non-zeros: {}".format(ori_nnz))
    logging.info("finish L1 training, num of edges * 2 + diag in adj1: {}".format(np.count_nonzero(cur_adj1)))


    ## --------------------------------------
    ## Update adjacency matrix
    ## --------------------------------------

    _adj = SparseTensor.from_torch_sparse_coo_tensor(model.adj1.cpu())
    row, col, value = _adj.coo()
    edge_index = torch.stack([row, col])
    print(edge_index.shape)
    data.to(torch.device("cpu"))
    data = Data(edge_index=edge_index, x=data.x, y=data.y, train_mask=data.train_mask, test_mask=data.test_mask, val_mask=data.val_mask)
    data = data.to(device)

    ## --------------------------------------
    ## Retraining
    ## --------------------------------------

    logging.info('***'*20)
    logging.info('retraining at iteration {}'.format(iteration))
    logging.info('***'*20)

    state_dict = model.state_dict()

    model = SAGE(dataset.num_features, 32, dataset.num_classes, data=data, device=device, quant=args.quant,
                num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits,
                chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x = data.x.to(device)
    y = data.y.squeeze().to(device)

    # model.load_state_dict(state_dict)

    train_acc, val_acc, test_acc = test(model, data.to(device), x, y, batch_size=64)
    logging.info('Before retraining: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(train_acc, val_acc, test_acc))

    best_model = None
    best_val_acc = best_test_acc = 0
    for retrain_epoch in range(100):
        loss, acc = retrain(model, data, epoch, optimizer, x, y)
        train_acc, val_acc, test_acc = test(model, data, x, y, batch_size=64)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = model
        logging.info('Retrain Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
                      retrain_epoch, train_acc, val_acc, test_acc))

    logging.info('Best val. acc : {}'.format(best_val_acc))
    logging.info('Best test acc : {}'.format(best_test_acc))

    return best_model, data


def main_hardgroup(data, model, device):

    ## --------------------------------------
    ## Graph Partition
    ## --------------------------------------

    subgraph_nnz = count_subgraph_nnz(data.edge_index, n_subgraphs)
    # logging.info('subgraphs within each classes for one group: {}'.format(class_graphs))
    logging.info('subgraph_nnz before hard tuning: \n {}'.format(subgraph_nnz.astype(int)))

    logging.info('***'*20)
    logging.info('hardcore group by removing boundary nodes')
    logging.info('***'*20)

    data = data.to(torch.device("cpu"))
    ### previous hard pruning ###
    # edge_index, boundary = remove_boundary_nodes_among_groups(data.edge_index, adj_size, n_groups)
    # -------- add edge weights -----------
    # edge_index, new_adj, boundary = remove_boundary_nodes_among_groups(data.edge_index, adj_size, n_groups, adj=model.adj1)
    # -------------------------------------

    ### now remove chunks
    edge_index = remove_chunks(data.edge_index, adj_size, n_subgraphs, pre_subgraph_nnz=subgraph_nnz)
    data = Data(edge_index=edge_index, x=data.x, y=data.y, train_mask=data.train_mask, test_mask=data.test_mask, val_mask=data.val_mask).to(device)

    subgraph_nnz = count_subgraph_nnz(data.edge_index, n_subgraphs)
    # logging.info('subgraphs within each classes for one group: {}'.format(class_graphs))
    logging.info('subgraph_nnz after hard tuning: \n {}'.format(subgraph_nnz.astype(int)))
    logging.info('number of nodes within each subgraph: {}'.format(n_subgraphs))
    logging.info('number of edges within each subgraph: {}'.format(subgraph_nnz.diagonal()))


    ## --------------------------------------
    ## Retraining
    ## --------------------------------------

    logging.info('***'*20)
    logging.info('retraining after hardcore grouping')
    logging.info('***'*20)

    state_dict = model.state_dict()

    model = SAGE(dataset.num_features, 32, dataset.num_classes, data=data, device=device, quant=args.quant,
                num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits,
                chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x = data.x.to(device)
    y = data.y.squeeze().to(device)

    # model.load_state_dict(state_dict)

    train_acc, val_acc, test_acc = test(model, data.to(device), x, y)
    logging.info('Before retraining: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(train_acc, val_acc, test_acc))

    best_model = None
    best_val_acc = best_test_acc = 0
    for retrain_epoch in range(100):
        loss, acc = retrain(model, data, retrain_epoch, optimizer, x, y)
        train_acc, val_acc, test_acc = test(model, data, x, y, batch_size=64)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = model
        logging.info('Retrain Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
                      retrain_epoch, train_acc, val_acc, test_acc))

    logging.info('Best val. acc : {}'.format(best_val_acc))
    logging.info('Best test acc : {}'.format(best_test_acc))

    logging.info('\n')
    logging.info('original number of non-zeros: {}'.format(ori_nnz))
    logging.info('finish tuning, num of edges * 2 + diag in adj1: {}'.format(
                  np.count_nonzero(model.adj1.cpu().to_dense().numpy())))

    scipy_adj = SparseTensor.from_torch_sparse_coo_tensor(model.adj1).to_scipy()
    io.mmwrite(save_path + '/{}_newadj_hard.mtx'.format(args.dataset), scipy_adj)
    logging.info('saving new hard grouped adj. matrix ...')

    return best_model, data


def retrain(model, data, epoch, optimizer, x, y):
    model.train()

    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                            sizes=[25, 10], batch_size=64, shuffle=True,
                            num_workers=12)

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description('Epoch {:03d}'.format(epoch))

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        # print(batch_size)
        # print(n_id.shape)
        # print(data.edge_index[:,adjs[0][1][0]],adjs[0][1])

        optimizer.zero_grad()
        # out = model(x[n_id], adjs)
        out = model(x, adjs, n_id)

        # loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss = F.nll_loss(out[n_id[:batch_size]], y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        # total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        total_correct += int(out[n_id[:batch_size]].argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc

@torch.no_grad()
def test(model, data, x, y, batch_size=64):
    model.eval()

    out = model.inference(x, batch_size, device=device)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


# Train graph with SAGE
def train(model, epoch, optimizer, x, y, support1, U1, Z1, support2, U2, Z2, batch_size=64, adjOpt=False, retrain=False, adj_mask=None):
    model.train()
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                sizes=[25, 10], batch_size=batch_size, shuffle=True,
                                num_workers=12)


    id1 = model.id
    id2 = model.id
    d1 = (support1 + U1) + (-Z1 + (-id1))
    d2 = (support2 + U2) + (-Z2 + (-id2))

    _adj = SparseTensor.from_torch_sparse_coo_tensor(support1)
    row, col, value = _adj.coo()
    diagonalization = torch.sum(abs(row - col) * value) / len(row)

    # admm_loss = lambda m: loss(m) + \
    #             args.rho * (torch.sum(d1.coalesce().values() * d1.coalesce().values()) + \
    #             torch.sum(d2.coalesce().values()*d2.coalesce().values())) + \
    #             args.phi * diagonalization

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples, one for each layer.
        adjs = [adj.to(device) for adj in adjs]

        # Optimization
        optimizer.zero_grad()
        out = model(x, adjs, n_id)  # modified to pass in the whole x
        loss = F.nll_loss(out[n_id[:batch_size]], y[n_id[:batch_size]])
        if not retrain:
            loss += args.rho * (torch.sum(d1.coalesce().values() * d1.coalesce().values()) + \
                    torch.sum(d2.coalesce().values()*d2.coalesce().values())) + \
                    args.phi * diagonalization
        loss.backward(retain_graph=True)
        if adjOpt:
            update_gradients_adj({"support1": support1, "support2": support2}, adj_mask)
        optimizer.step()

        if retrain:
            for k, m in enumerate(model.modules()):
                if isinstance(m, SAGEConvWeighted):
                    for ind,lin in enumerate([m.lin_l, m.lin_r]):
                        weight_copy = lin.weight.data.abs().clone()
                        mask = weight_copy.gt(0).float().to(device)
                        lin.weight.grad.data.mul_(mask)

        total_loss += float(loss)
        total_correct += int(out[n_id[:batch_size]].argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


# main calling function

for iteration in range(args.iteration):
    model, data = main_tune(data, model, device, iteration)

torch.save({"state_dict":model.state_dict(),"adj":model.adj1, "data":data}, save_path + '/ckpt.pth.tar')

if args.hard:
    model, data = main_hardgroup(data, model, device)

torch.save({"state_dict":model.state_dict(),"adj":model.adj1, "data":data}, save_path + '/ckpt_hard.pth.tar')

