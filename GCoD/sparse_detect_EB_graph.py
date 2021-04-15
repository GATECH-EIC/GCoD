import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv  # noga
from utils import *
from sparse_pytorch_train import *
import numpy as np
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix, tril
from scipy import sparse
import torch.sparse as ts
import logging

parser = argparse.ArgumentParser() # EB
parser.add_argument('--times', type=int, default=4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--ratio_graph', type=int, default=20)
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--lookback', type=int, default=3)
parser.add_argument("--thres", type=float, default=0.1)
parser.add_argument("--dataset", type=str, default="CiteSeer")
parser.add_argument('--save_dir', type=str, default='./graph_train_eb')
parser.add_argument("--log", type=str, default="{:00d}: {:00d}")
args = parser.parse_args()

dataset = args.dataset
logging.basicConfig(filename=f"test_{dataset}_eb_emerge_new.txt",level=logging.DEBUG)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# print(f"Number of graphs in {dataset} dataset:", len(dataset))
data = dataset[0]
model, data = Net(dataset, data, args).to(device), data.to(device)
checkpoint = torch.load(f"./pretrain/{args.dataset}_model.pth.tar")
model.load_state_dict(checkpoint)

loss = lambda m: F.nll_loss(m()[data.train_mask], data.y[data.train_mask])
# print("construct admm training")
support1 = model.adj1 # sparse
support2 = model.adj2 # sparse
s1_values = support1.coalesce().values()
s2_values = support2.coalesce().values()
partial_adj_mask = support1.clone()
adj_variables = [support1,support2]
rho = 1e-3
non_zero_idx = SparseTensor.from_torch_sparse_coo_tensor(model.adj1).nnz()
Z1 = U1 = Z2 = U2 = partial_adj_mask.clone()
model.adj1.requires_grad = True
model.adj2.requires_grad = True
adj_mask = partial_adj_mask.clone()

# create save dir
if not osp.exists(args.save_dir):
    os.mkdir(args.save_dir)

if not osp.exists('./masks'):
    os.mkdir('./masks')

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
    return SparseTensor.from_scipy(new_adj).to_torch_sparse_coo_tensor().to(device)

def scipy_to_pytorch_sparse(coo):
    values = coo.data
    indices = np.stack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return ts.FloatTensor(i,v,torch.Size(shape))

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
    return SparseTensor.from_scipy((new_adj != adj))

def calc_dist(m1,m2):
    diff = m1 - m2
    neg = diff < 0
    diff[neg] = -diff[neg]
    return torch.sum(diff).item()

def post_processing():
    adj1,adj2 = model.adj1, model.adj2
    adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
    adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
    model.adj1 = adj1.float()
    model.adj2 = adj2.float()

    # print("Optimization Finished!")
    # train_acc, val_acc, tmp_test_acc = test(model, data)
    log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(args.ratio, train_acc, val_acc, tmp_test_acc))
    log_4_test = 'Tune Ratio: {:d}'
    # print(log_4_test.format(args.ratio))
    # cur_adj1 = model.adj1.cpu().numpy()
    # cur_adj2 = model.adj2.cpu().numpy()

    print('save back!')

    torch.save({"state_dict":model.state_dict(),"adj":model.adj1}, "{}/{}_prune_{}.pth.tar".format(args.save_dir, args.dataset, str(args.ratio_graph)))

id1 = model.id
id2 = model.id
# e = torch.ones(id1.shape[0]).to_sparse()
# eT = torch.ones(id1.shape[0]).t.to_sparse()
# Define new loss function
d1 = support1 + U1 - (Z1 + id1)
d2 = support2 + U2 - (Z2 + id2)
admm_loss = lambda m: loss(m) + \
            rho * (torch.sum(d1.coalesce().values() * d1.coalesce().values()) +
            torch.sum(d2.coalesce().values()*d2.coalesce().values()))

adj_optimizer = torch.optim.SGD(adj_variables,lr=0.001)
adj_map = {"support1": support1, "support2": support2}

# jteller@fiverings.com
best_prune_acc = 0
lookbacks = []
counter = 0
first_d = -1
for j in range(args.times):
    for epoch in range(args.epochs):
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
        cur_mask = get_mask(model.adj1 - id1, non_zero_idx, args.ratio_graph)
        torch.save(cur_mask, f"./masks/{args.dataset}_{args.ratio_graph}_{counter}_mask")
        cur_mask = cur_mask.to_dense().int().cpu()
        if len(lookbacks) < args.lookback:
            lookbacks.append(cur_mask)
        else:
            can_return = True
            total = 0
            for mask in lookbacks:

                dist = calc_dist(mask, cur_mask)
                if first_d == -1:
                    first_d = dist
                dist /= first_d
                total = max(dist,total)

                if total > args.thres:
                    can_return = False
                    break
            # print(counter, total)
            if can_return:
                logging.info(args.log.format(args.ratio_graph, j * args.epochs + epoch))
                print(f"Found EB! At {j * args.epochs + epoch}")
                post_processing()
                exit()
            lookbacks = lookbacks[1:]
            lookbacks.append(cur_mask)
        torch.save(cur_mask, f"./masks/{args.dataset}_{args.ratio_graph}_{counter}_mask")
        counter += 1

    # Use learnt U1, Z1 and so on to prune
    adj1,adj2 = model.adj1, model.adj2
    Z1 = adj1 - id1 + U1
    Z1 = prune_adj(Z1,non_zero_idx,args.ratio_graph) - id1
    U1 = U1 + (adj1 - id1 - Z1)

    Z2 = adj2 - id2 + U2
    Z2 = prune_adj(Z2,non_zero_idx,args.ratio_graph) - id2
    U2 = U2 + (adj2 - id2 - Z2)

adj1,adj2 = model.adj1, model.adj2
adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
model.adj1 = adj1.float()
model.adj2 = adj2.float()

# print("Optimization Finished!")
train_acc, val_acc, tmp_test_acc = test(model, data)
log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
print(log.format(args.ratio_graph, train_acc, val_acc, tmp_test_acc))
log_4_test = 'Tune Ratio: {:d}'
print(log_4_test.format(args.ratio_graph))
cur_adj1 = model.adj1.cpu().to_dense().numpy()
cur_adj2 = model.adj2.cpu().to_dense().numpy()
print(np.count_nonzero(cur_adj1))

torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, "{}/{}_prune_{}.pth.tar".format(args.save_dir, args.dataset, str(args.ratio_graph)))





