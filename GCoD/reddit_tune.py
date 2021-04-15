import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Reddit
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv  # noga
from utils import *
from pytorch_train import *
import numpy as np
from sampler import NeighborSampler
from reddit import *

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int, default=4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument("--draw", type=int, default=0)
parser.add_argument('--ratio_graph', type=int, default=90)
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--save_file', type=str, default="model.pth.tar")
parser.add_argument("--dataset", type=str, default="Reddit")
args = parser.parse_args()

dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Reddit(path, dataset, transform=T.NormalizeFeatures())
# print(f"Number of graphs in {dataset} dataset:", len(dataset))
data = dataset[0]
model= SAGE(dataset.num_features, 256, dataset.num_classes).to(device)

global_edge_index = data.edge_index
global_values = torch.ones(global_edge_index[0].shape)
train_loader = NeighborSampler(global_edge_index, node_idx=data.train_mask,
                                sizes=[25, 10], batch_size=1024, shuffle=True,
                                num_workers=12)
subgraph_loader = NeighborSampler(global_edge_index, node_idx=None, sizes=[-1],
                                    batch_size=1024, shuffle=False,
                                    num_workers=12)

x = data.x.to(device)
y = data.y.squeeze().to(device)

checkpoint = torch.load(f"./pretrain_pytorch/{args.dataset}_model.pth.tar")
model.load_state_dict(checkpoint)

# Update the gradient of the adjacency matrices
# grads_vars: {name: torch.Tensor}
def update_gradients_adj(grads_vars ,adj_p_mask:np.ndarray):
    temp_grad_adj1 = 0
    var1 = None
    var2 = None
    temp_grad_adj2 = 0
    for key,var in grads_vars.items():
        grad = var.grad
        if key == "support1":
            adj_mask = torch.from_numpy(adj_p_mask).to(device)
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = torch.transpose(temp_grad_adj,1,0)
            temp_grad_adj1 = temp_grad_adj + transposed_temp_grad_adj
            var1 = var
        if key == "support2":
            adj_mask = torch.from_numpy(adj_p_mask).to(device)
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = torch.transpose(temp_grad_adj,1,0)
            temp_grad_adj2 = temp_grad_adj + transposed_temp_grad_adj
            var2 = var
    grad_adj = (temp_grad_adj1 + temp_grad_adj2) / 4 # Why are we doing this?
    var1.grad = grad_adj
    var2.grad = grad_adj
    return [var1,var2]

def prune_adj(oriadj:torch.Tensor, non_zero_idx:int, percent:int) -> torch.Tensor:
    original_prune_num = int(((non_zero_idx - oriadj.shape[0]) / 2) * (percent / 100))
    adj = np.copy(oriadj.detach().cpu().numpy())
    # print(f"Pruning {percent}%")
    low_adj = np.tril(adj, -1)
    non_zero_low_adj = low_adj[low_adj != 0]

    low_pcen = np.percentile(abs(non_zero_low_adj), percent)
    under_threshold = abs(low_adj) < low_pcen
    before = len(non_zero_low_adj)
    low_adj[under_threshold] = 0
    non_zero_low_adj = low_adj[low_adj != 0]
    after = len(non_zero_low_adj)

    rest_pruned = original_prune_num - (before - after)
    # print(adj.shape[0],original_prune_num,before,after, before-after)
    if rest_pruned > 0:
        mask_low_adj = (low_adj != 0)
        low_adj[low_adj == 0] = 2000000
        flat_indices = np.argpartition(low_adj.ravel(), rest_pruned - 1)[:rest_pruned]
        row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
        low_adj = np.multiply(low_adj, mask_low_adj)
        low_adj[row_indices, col_indices] = 0
    adj = low_adj + np.transpose(low_adj)
    adj = np.add(adj, np.identity(adj.shape[0]))
    return torch.from_numpy(adj).to(device)



best_prune_acc = 0

for batch_size, n_id, adjs in train_loader:
    # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
    adjs = [adj.to(device) for adj in adjs]
    subgraphs = [(edge_to_adj(adj[0]),adj[1],adj[2]) for adj in adjs]
    to_forward = [(adj_to_edge(sub[0]), sub[1], sub[2]) for sub in subgraphs]
    adjs_to_diff = [sub[0] for sub in subgraphs]
    partial_masks = [adj.cpu().numpy() for adj in adjs_to_diff]
    for adj in adjs_to_diff:
        adj.requires_grad = True
    Z1 = U1 = torch.from_numpy(np.zeros_like(adjs_to_diff[0])).to(device)
    Z2 = U2 = torch.from_numpy(np.zeros_like(adjs_to_diff[1])).to(device)
    non_zero_idx = np.count_nonzero(adjs_to_diff[0].cpu().numpy())
    proxy = [(U1, Z1), (U2, Z2)]
    id1 = torch.eye(adjs_to_diff[0].shape[0]).to(device)
    id2 = torch.eye(adjs_to_diff[1].shape[0]).to(device)
    optimizer = torch.optim.Adam(adjs_to_diff, lr=0.01)
    cnt = 0
    for j in range(args.times):
        for epoch in range(args.epochs):
            if cnt == args.draw:
                break
            model.train()
            optimizer.zero_grad()
            out = model(x[n_id], to_forward)
            # update the subnetwork
            admm_loss = F.nll_loss(out, y[n_id[:batch_size]])
            for ind, (U1, Z1) in enumerate(proxy):
                support = adjs_to_diff[ind]
                admm_loss += rho *F.mse_loss(support + U1, Z1 + torch.eye(support.shape[0]).to(device))
            admm_loss.backward(retain_graph=True)
            adj1 = adjs_to_diff[0]
            adj2 = adjs_to_diff[1]
            update_gradients_adj({"support1": adj1, "support2": adj2},partial_masks[0])
            optimizer.step()

        # Use learnt U1, Z1 and so on to prune
        adj1,adj2 = adjs_to_diff[0],adjs_to_diff[1]
        Z1 = adj1 - id1 + U1
        Z1 = prune_adj(Z1,non_zero_idx,args.ratio_graph) - id1
        U1 = U1 + (adj1 - id1 - Z1)

        Z2 = adj2 - id2 + U2
        Z2 = prune_adj(Z2,non_zero_idx,args.ratio_graph) - id2
        U2 = U2 + (adj2 - id2 - Z2)

        if cnt == args.draw:
            break

    adj1,adj2 = adjs_to_diff[0], adjs_to_diff[1]
    adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
    adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
    edge_index1, e_id1, _ = adjs[0]
    edge_index2, e_id2, _ = adjs[1]
    global_values[e_id1] = adj1[torch.transpose(edge_index1,0,1)]
    global_values[e_id2] = adj2[torch.transpose(edge_index2,0,1)]

# print("Optimization Finished!")
train_acc, val_acc, tmp_test_acc = test(model, data)
log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
# print(log.format(args.ratio, train_acc, val_acc, tmp_test_acc))
log_4_test = 'Tune Ratio: {:d}'
# print(log_4_test.format(args.ratio))
cur_adj1 = model.adj1.cpu().numpy()
cur_adj2 = model.adj2.cpu().numpy()
print("finish L1 training, num of edges * 2 + diag in adj1:", np.count_nonzero(cur_adj1))
# print("finish L1 training, num of edges * 2 + diag in adj2:", np.count_nonzero(cur_adj2))
# print("symmetry result adj1: ", testsymmetry(cur_adj1))
# print("symmetry result adj2: ", testsymmetry(cur_adj2))
# print("is equal of two adj", isequal(cur_adj1, cur_adj2))

torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./graph_pruned_pytorch/{args.save_file}")





