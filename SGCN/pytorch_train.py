import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv  # noga
from utils import *
import numpy as np
from Facebook100_dataset import Facebook100

import networkx as nx

device = torch.device('cpu')

#
# Differentiable conversion from edge_index/edge_attr to adj
def edge_to_adj(edge_index, edge_attr=None,num_nodes=None):
    row, col = edge_index

    if edge_attr is None:
        edge_attr = torch.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1)
        assert edge_attr.size(0) == row.size(0)
    n_nodes = geo_num_nodes.maybe_num_nodes(edge_index, num_nodes)
    diff_adj = torch.zeros([n_nodes,n_nodes])
    diff_adj += torch.eye(diff_adj.shape[0])
    diff_adj[row,col] = edge_attr
    return diff_adj

def adj_to_edge(adj:torch.Tensor):
    new_adj = adj - torch.eye(adj.shape[0])
    edge_index = (new_adj > 0).nonzero(as_tuple=False).t()
    row,col = edge_index
    edge_weight = new_adj[row,col].float()
    return (edge_index.to(device),edge_weight.to(device))

class Net(torch.nn.Module):
    def __init__(self, dataset, data, args, adj=()):
        super(Net, self).__init__()
        self.data = data
        self.conv1 = GCNConv(dataset.num_features, 16,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes,
                             normalize=not args.use_gdc)
        # print(adj)
        if len(adj) == 0:
            self.adj1 = edge_to_adj(data.edge_index,data.edge_attr,data.num_nodes)
        else:
            self.adj1 = torch.from_numpy(adj)
        # self.adj1 = torch.eye(data.num_nodes)
        self.adj2 = self.adj1.clone()
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = self.data.x, self.data.edge_index, self.data.edge_attr
        self.ei1, self.ew1 = adj_to_edge(self.adj1)
        self.ei2, self.ew2 = adj_to_edge(self.adj2)
        x = F.relu(self.conv1(x, self.ei1, self.ew1))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, self.ei2, self.ew2)
        return F.log_softmax(x, dim=1)

def train(model,data):
    model.train()
    optimizer.zero_grad()

    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(), []
    masks = ['train_mask', 'val_mask', 'test_mask']
    for i,(_, mask) in enumerate(data('train_mask', 'val_mask', 'test_mask')):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--use_gdc', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='CiteSeer', choices=['CiteSeer', 'Reddit', 'Cora', 'Caltech36'])
    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset in ['CiteSeer']:
        dataset = args.dataset
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        # print(dataset, data)
        # exit()
    elif args.dataset in ['Reddit']:
        dataset = args.dataset
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = pyg.datasets.Reddit(path, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif args.dataset in ['Caltech36']:
        dataset = args.dataset
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = Facebook100(path, dataset, 'housing', transform=T.NormalizeFeatures())
        data = dataset[0]

    model, data = Net(dataset, data, args).to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.
    best_val_acc = test_acc = 0
    for epoch in range(1, args.epochs):
        train(model,data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Pretrain Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print(model.state_dict().keys())
    torch.save(model.state_dict(), "./pretrain_pytorch/model.pth.tar")