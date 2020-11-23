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
parser.add_argument('--iteration', type=int, default=4)
parser.add_argument('--times', type=int, default=4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--ratio_graph', type=int, default=10)
parser.add_argument("--draw", type=int, default=100)
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--lookback', type=int, default=3)
parser.add_argument("--thres", type=float, default=0.0)
parser.add_argument("--dataset", type=str, default="CiteSeer")
parser.add_argument('--save_dir', type=str, default='./graph_train')
parser.add_argument("--log", type=str, default="{:05d}")
parser.add_argument('--num_groups', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--total_subgraphs', type=int, default=10)
args = parser.parse_args()

dataset = args.dataset
logging.basicConfig(filename=f"test_{dataset}_mask_change.txt",level=logging.DEBUG)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
if dataset == "F":
    dataset = Flickr(path, transform=T.NormalizeFeatures())
    print(len(dataset))
elif dataset in ['Caltech36']:
    dataset = Facebook100(path, dataset, 'housing', transform=T.NormalizeFeatures())
else:
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
degree_split = [3, 5] # for num_class = 3
data, n_subgraphs, class_graphs = my_partition_graph(data, degree_split, args.total_subgraphs, args.num_groups, dataset=args.dataset)
n_subgraphs, n_classes, n_groups = my_get_boundary(n_subgraphs, class_graphs, args.num_groups)
# # print(f"Number of graphs in {dataset} dataset:", len(dataset))
# data = dataset[0] # fetch the first graph
# if args.dataset in ['Caltech36']:
#     print('set mask')
#     data = set_random_mask(data)
#     data.y = data.y + 1

checkpoint = torch.load(f"./pretrain/{args.dataset}_model.pth.tar")
state_dict = checkpoint["state_dict"]
data = checkpoint["data"]
adj = checkpoint["adj"]
adj_size = data.x.size(0)

ori_nnz = np.count_nonzero(adj.cpu().to_dense().numpy())
# data = cluster_graph(data, num_parts=args.group, save=False)
model, data = Net(dataset, data, args, adj=adj).to(device), data.to(device)
model.load_state_dict(state_dict)

scipy_adj = SparseTensor.from_torch_sparse_coo_tensor(model.adj1).to_scipy()
print(scipy_adj)
io.mmwrite(f"./pretrain/{args.dataset}_adj.mtx", scipy_adj)

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
            if row > idx_list[i-1] and row < idx_list[i]:
                x = i
            if col > idx_list[i-1] and col < idx_list[i]:
                y = i
    return x, y

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

group_nnz_bef = count_group_nnz(support1, adj_size, group=args.num_groups)

def remove_boundary_nodes_among_groups(edge_index, adj_size, group):
    row = edge_index[0]
    col = edge_index[1]

    remove_list = []
    for i in range(len(row)):
        x = math.floor(row[i] / math.ceil(adj_size / group))
        y = math.floor(col[i] / math.ceil(adj_size / group))
        if x + y == 1:
            remove_list.append(i)

    counter = 0
    for i in range(len(remove_list)):
        item = remove_list[i] - counter
        row = torch.cat([row[:item], row[item+1:]])
        col = torch.cat([col[:item], col[item+1:]])
        counter += 1
    edge_index = torch.stack([row, col])
    return edge_index


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
diagonalization = torch.sum(abs(row - col) * value) / len(row)

d1 = support1 + U1 - (Z1 + id1)
d2 = support2 + U2 - (Z2 + id2)
admm_loss = lambda m: loss(m) + \
            rho * (torch.sum(d1.coalesce().values() * d1.coalesce().values()) + \
            torch.sum(d2.coalesce().values()*d2.coalesce().values())) + \
            10 * diagonalization

adj_optimizer = torch.optim.SGD(adj_variables,lr=0.001)
adj_map = {"support1": support1, "support2": support2}

# retrain util
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.

def retrain(model, data): # keep nonzeros in weights
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()

    # zero_num = get_conv_zero_param(model)
    # print(f"Number of zero parameters: {zero_num}")
    # ------ don't update those pruned grads! -
    # for k, m in enumerate(model.modules()):
    #     # print(k, m)
    #     if isinstance(m, GCNConv):
    #         weight_copy = m.weight.data.abs().clone()
    #         mask = weight_copy.gt(0).float().to(device)
    #         m.weight.grad.data.mul_(mask)
    # -----------------------------------------

    optimizer.step()

for iteration in range(args.iteration):

    ## --------------------------------------
    ## Graph Sparsification
    ## --------------------------------------
    print('***'*20)
    print('pruning at iteration {}'.format(iteration))
    print('***'*20)

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

    group_nnz_aft = count_group_nnz(model.adj1, adj_size, group=args.num_groups)

    # eye = torch.eye(data.x.size(0)).to_sparse().to(device)
    # print(model.adj1)
    # _adj = SparseTensor.from_torch_sparse_coo_tensor(model.adj1)
    # row, col, value = _adj.coo()
    # edge_attr = torch.ones(row.size(0)).to(device)
    # newadj = SparseTensor(row=row, col=col, value=torch.clone(edge_attr)).to_torch_sparse_coo_tensor().to(device)
    # scipy_adj = SparseTensor.from_torch_sparse_coo_tensor(newadj).to_scipy()
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


    ## --------------------------------------
    ## Update adjacency matrix
    ## --------------------------------------

    # if iteration > 0:
    #     subgraph_nnz = count_subgraph_nnz(data.edge_index, n_subgraphs)
    #     print('subgraph_nnz before re-partition: \n', subgraph_nnz)

    # update edge_index in the data
    _adj = SparseTensor.from_torch_sparse_coo_tensor(model.adj1.cpu())
    row, col, value = _adj.coo()
    edge_index = torch.stack([row, col])
    print(edge_index.shape)
    data.to(torch.device("cpu"))
    data = Data(edge_index=edge_index, x=data.x, y=data.y, train_mask=data.train_mask, test_mask=data.test_mask, val_mask=data.val_mask)

    ## --------------------------------------
    ## Retraining
    ## --------------------------------------

    print('***'*20)
    print('retraining at iteration {}'.format(iteration))
    print('***'*20)

    state_dict = model.state_dict()

    model = Net(dataset, data, args, adj=model.adj1).to(device)
    model.load_state_dict(state_dict)
    train_acc, val_acc, tmp_test_acc = test(model, data.to(device))
    print('Loaded pruned model with accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(train_acc, val_acc, tmp_test_acc))

    for retrain_epoch in range(100):
        retrain(model, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        test_acc = 0
        if tmp_test_acc > test_acc:
            test_acc = tmp_test_acc
        log = 'Retrain Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(retrain_epoch, train_acc, val_acc, test_acc))

        if retrain_epoch == 99:
            print('best retraining accuracy: ', test_acc)

    ## --------------------------------------
    ## Graph Partition
    ## --------------------------------------
    edge_index = remove_boundary_nodes_among_groups(data.edge_index, adj_size, args.num_groups)
    data = Data(edge_index=edge_index, x=data.x, y=data.y, train_mask=data.train_mask, test_mask=data.test_mask, val_mask=data.val_mask)
    # print('***'*20)
    # print('re-partition at iteration {}'.format(iteration))
    # print('***'*20)

    # if iteration == 0:
    #     # my partition
    #     degree_split = [3, 5] # for num_class = 3
    #     data.to(torch.device("cpu"))
    #     data, n_subgraphs, class_graphs = my_partition_graph(data, degree_split, args.total_subgraphs, args.num_groups, dataset=args.dataset)
    #     n_subgraphs, n_classes, n_groups = my_get_boundary(n_subgraphs, class_graphs, args.num_groups)
    #     print(data)
    #     # print(data.train_mask, data.test_mask)
    #     print(class_graphs)
    #     print('n_subgraphs: ', n_subgraphs)
    #     print('n_classes: ', n_classes)
    #     print('n_groups: ', n_groups)

    group_nnz_aft = count_group_nnz(model.adj1, adj_size, group=args.num_groups)
    print('group_nnz before pruning: \n', group_nnz_bef)
    print('group_nnz after pruning: \n', group_nnz_aft)

    print('subgraphs within each classes for one group: ', class_graphs)

    subgraph_nnz = count_subgraph_nnz(data.edge_index, n_subgraphs)
    print('subgraph_nnz after re-partition: \n', subgraph_nnz)
    print('number of nodes within each subgraph: ', n_subgraphs)
    print('number of edges within each subgraph: ', subgraph_nnz.diagonal())

    print("original number of non-zeros: ", ori_nnz)
    print("finish L1 training, num of edges * 2 + diag in adj1:", np.count_nonzero(cur_adj1))

    #     state_dict = model.state_dict()
    #     model, data = Net(dataset, data, args).to(device), data.to(device)
    #     model.load_state_dict(state_dict)




torch.save({"state_dict":model.state_dict(),"adj":model.adj1,"data":data}, "{}/{}_prune_{}.pth.tar".format(args.save_dir, args.dataset, str(args.ratio_graph)))





