import os.path as osp
import argparse

import os
import math
import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Planetoid, TUDataset, Flickr
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv  # noga
from utils import *
from sparse_pytorch_train import *
import numpy as np
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix, tril
from scipy import sparse, io
import torch.sparse as ts
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int, default=4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--ratio_graph', type=int, default=40)
parser.add_argument("--draw", type=int, default=100)
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--lookback', type=int, default=3)
parser.add_argument("--thres", type=float, default=0.0)
parser.add_argument("--dataset", type=str, default="CiteSeer")
parser.add_argument('--save_dir', type=str, default='./graph_train')
parser.add_argument("--log", type=str, default="{:05d}")
parser.add_argument('--group', type=int, default=3)
parser.add_argument('--class', type=int, default=3)
args = parser.parse_args()

dataset = args.dataset
logging.basicConfig(filename=f"test_{dataset}_mask_change.txt",level=logging.DEBUG)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
if dataset == "F":
    dataset = Flickr(path, transform=T.NormalizeFeatures())
    print(len(dataset))
else:
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# print(f"Number of graphs in {dataset} dataset:", len(dataset))
data = dataset[0] # fetch the first graph
adj_size = data.x.size(0)
checkpoint = torch.load(f"./pretrain/{args.dataset}_model.pth.tar")
state_dict = checkpoint["state_dict"]
adj = checkpoint["adj"]
ori_nnz = np.count_nonzero(adj.cpu().to_dense().numpy())
data = cluster_graph(data, num_parts=args.group, save=False)
model, data = Net(dataset, data, args, adj=adj).to(device), data.to(device)
model.load_state_dict(state_dict)

loss = lambda m: F.nll_loss(m()[data.train_mask], data.y[data.train_mask])
# print("construct admm training")
support1 = model.adj1 # sparse
support2 = model.adj2 # sparse
partial_adj_mask = support1.clone()
adj_variables = [support1,support2]
rho = 1e-3
non_zero_idx = SparseTensor.from_torch_sparse_coo_tensor(model.adj1).nnz() # number of non zeros
print(non_zero_idx)
Z1 = U1 = Z2 = U2 = partial_adj_mask.clone()
model.adj1.requires_grad = True
model.adj2.requires_grad = True
adj_mask = partial_adj_mask.clone()

# create save dir
if not osp.exists(args.save_dir):
    os.mkdir(args.save_dir)

# Update the gradient of the adjacency matrices
# grads_vars: {name: torch.Tensor}
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

# torch.sparse
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

def post_processing(): # unused
    adj1,adj2 = model.adj1, model.adj2
    adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
    adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
    model.adj1 = adj1
    model.adj2 = adj2

    # print("Optimization Finished!")
    train_acc, val_acc, tmp_test_acc = test(model, data)
    log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(args.ratio, train_acc, val_acc, tmp_test_acc))
    log_4_test = 'Tune Ratio: {:d}'
    # print(log_4_test.format(args.ratio))
    cur_adj1 = model.adj1.cpu().numpy()
    cur_adj2 = model.adj2.cpu().numpy()

    torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, "{}/{}_prune_{}.pth.tar".format(args.save_dir, args.dataset, str(args.ratio_graph)))


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

group_nnz_bef = count_group_nnz(support1, adj_size, group=args.group)

id1 = model.id
id2 = model.id
# e = torch.ones(id1.shape[0]).to_sparse()
# eT = torch.ones(id1.shape[0]).t.to_sparse()
# Define new loss function
_adj = SparseTensor.from_torch_sparse_coo_tensor(support1)
row, col, value = _adj.coo()
# print(row)
# print(col)
# print(value)
diagonalization = torch.sum(abs(row - col) * value)

d1 = support1 + U1 - (Z1 + id1)
d2 = support2 + U2 - (Z2 + id2)
admm_loss = lambda m: loss(m) + \
            rho * (torch.sum(d1.coalesce().values() * d1.coalesce().values()) + \
            torch.sum(d2.coalesce().values()*d2.coalesce().values())) + \
            1e-2 * diagonalization

adj_optimizer = torch.optim.SGD(adj_variables,lr=0.001)
adj_map = {"support1": support1, "support2": support2}

# jteller@fiverings.com
best_prune_acc = 0
lookbacks = []
counter = 0
for j in range(args.times):
    for epoch in range(args.epochs):
        if counter == args.draw:
            break
        model.train()
        adj_optimizer.zero_grad()
        # Calculate gradient
        admm_loss(model).backward(retain_graph=True)
        # Update to correct gradient
        update_gradients_adj(adj_map, adj_mask)
        # Use the optimizer to update adjacency matrix
        # print("Support 1 grad:", support1.grad, support1.grad.shape)
        adj_optimizer.step()
        # print(adj_variables[0][adj_variables[0] != 1])
        # print("Support values:", support1.coalesce().values()[support1.coalesce().values() < 0.9999])

        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_prune_acc:
            best_prune_acc = val_acc
            test_acc = tmp_test_acc
        log = "Pruning Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(j*args.epochs+epoch, train_acc, val_acc, tmp_test_acc)
        counter += 1
        print(log.format(epoch, train_acc, best_prune_acc, tmp_test_acc))


    # Use learnt U1, Z1 and so on to prune
    adj1,adj2 = model.adj1, model.adj2
    Z1 = adj1 - id1 + U1
    Z1 = prune_adj(Z1,non_zero_idx,args.ratio_graph) - id1
    U1 = U1 + (adj1 - id1 - Z1)

    Z2 = adj2 - id2 + U2
    Z2 = prune_adj(Z2,non_zero_idx,args.ratio_graph) - id2
    U2 = U2 + (adj2 - id2 - Z2)

    if counter == args.draw:
        break

adj1,adj2 = model.adj1, model.adj2
adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
model.adj1 = adj1.float()
model.adj2 = adj2.float()

group_nnz_aft = count_group_nnz(model.adj1, adj_size, group=args.group)

scipy_adj = SparseTensor.from_torch_sparse_coo_tensor(model.adj1).to_scipy()
print(scipy_adj)
io.mmwrite(f"./pretrain/{args.dataset}_newadj.mtx", scipy_adj)

# print("Optimization Finished!")
train_acc, val_acc, tmp_test_acc = test(model, data)
log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
print(log.format(args.ratio_graph, train_acc, val_acc, tmp_test_acc))
log_4_test = 'Tune Ratio: {:d}'
print(log_4_test.format(args.ratio_graph))
cur_adj1 = model.adj1.cpu().to_dense().numpy()
cur_adj2 = model.adj2.cpu().to_dense().numpy()
print("original number of non-zeros: ", ori_nnz)
print("finish L1 training, num of edges * 2 + diag in adj1:", np.count_nonzero(cur_adj1))

print('group_nnz before pruning: \n', group_nnz_bef)
print('group_nnz after pruning: \n', group_nnz_aft)

torch.save({"state_dict":model.state_dict(),"adj":model.adj1}, "{}/{}_prune_{}.pth.tar".format(args.save_dir, args.dataset, str(args.ratio_graph)))





