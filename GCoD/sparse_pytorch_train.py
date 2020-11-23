import os.path as osp
import argparse

import os
import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Planetoid, TUDataset, Flickr, Coauthor, CitationFull
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, DataParallel
from utils import *
import numpy as np
from torch_geometric.utils import dense_to_sparse, degree
from torch_sparse import SparseTensor
# metis
from torch_geometric.data import Data, ClusterData, ClusterLoader
from torch_geometric.utils import to_dense_adj
try:
    torch.ops.torch_sparse.partition
    with_metis = True
except RuntimeError:
    with_metis = False
import pytest

from scipy import sparse, io
from Facebook100_dataset import Facebook100
from get_partition import my_partition_graph
from get_boundary import my_get_boundary

# device = torch.device("cpu")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def to_scipy(index, value, m, n):
    assert not index.is_cuda and not value.is_cuda
    (row, col), data = index.detach(), value.detach()
    return scipy.sparse.coo_matrix((data, (row, col)), (m, n))

@pytest.mark.skipif(not with_metis, reason='Not compiled with METIS support')
def cluster_graph(data, num_parts, save=True):
    cluster_data = ClusterData(data, num_parts=num_parts, recursive=True, log=True)

    # row, col, value = cluster_data.data.adj.coo()

    edge_attr = torch.ones(cluster_data.data.adj.storage._row.size(0))
    cluster_data.data.adj.set_value_(edge_attr, layout='coo')

    if save is True:
        eye = torch.eye(cluster_data.data.adj.size(0)).to_sparse().to(device)
        adj = cluster_data.data.adj.to_torch_sparse_coo_tensor().to(device)
        scipy_adj = SparseTensor.from_torch_sparse_coo_tensor(adj + eye).to_scipy()
        io.mmwrite(f"./pretrain/{args.dataset}_adj.mtx", scipy_adj)

    return cluster_data.data

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
    new_adj = adj - torch.eye(adj.shape[0]).to(device)
    edge_index = (new_adj > 0).nonzero(as_tuple=False).t()
    row,col = edge_index
    edge_weight = new_adj[row,col].float()
    return (edge_index.to(device),edge_weight.to(device))

class Net(torch.nn.Module):
    def __init__(self, dataset, data, args, adj=()):
        super(Net, self).__init__()
        self.data = data
        if args.dataset == "F":
            hidden = 128
        else:
            hidden = 16

        num_classes = dataset.num_classes
        if args.dataset in ['Caltech36']:
            num_classes += 1
        self.conv1 = GCNConv(dataset.num_features, hidden,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden, num_classes,
                             normalize=not args.use_gdc)
        # zhihan write before
        if data.edge_attr == None:
            print('add self loop!')
            data.edge_attr = torch.ones(data.edge_index[0].size(0))
        if len(adj) == 0:
            self.adj1 = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.clone(data.edge_attr)).to_torch_sparse_coo_tensor().to(device)
            self.adj1 = self.adj1 + torch.eye(self.adj1.shape[0]).to_sparse().to(device)
            # self.adj1 = (torch.clone(data.edge_index).to(device), torch.clone(data.edge_attr).to(device))
        else:
            self.adj1 = adj
        self.id = torch.eye(self.adj1.shape[0]).to_sparse().to(device)
        self.adj2 = self.adj1.clone()

        # haoran write w.r.t. PyG METIS
        # if len(adj) == 0:
        #     self.adj1 = self.data.adj.to_torch_sparse_coo_tensor().to(device)
        #     self.adj1 = self.adj1 + torch.eye(self.adj1.shape[0]).to_sparse().to(device)
        # else:
        #     self.adj1 = adj

        # self.id = torch.eye(self.adj1.shape[0]).to_sparse().to(device)
        # self.adj2 = self.adj1.clone()


    def forward(self):
        x = self.data.x
        # self.ei1, self.ew1 = self.adj1
        # self.ei2, self.ew2 = self.adj2
        x = F.relu(self.conv1(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj1 -self.id)))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj2 - self.id))
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


def set_random_mask(data):
    num_nodes = len(data.y)
    perm = torch.randperm(num_nodes)
    train_size = int(num_nodes * 0.7)
    val_size = int(num_nodes * 0.8)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[perm[:train_size]] = 1
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask[perm[train_size:val_size]] = 1
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask[perm[val_size:]] = 1
    new_data = Data(x=data.x, edge_index=data.edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, y=data.y)
    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--use_gdc', type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="CiteSeer") # choice: F: Flicker, C: DBLP, CiteSeer, Cora, Pumbed
    parser.add_argument('--num_groups', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--total_subgraphs', type=int, default=10)
    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = args.dataset

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    lrate = 0.01
    if dataset == "F": # cpu
        dataset = Flickr(path, transform=T.NormalizeFeatures())
        print(len(dataset))
        lrate = 0.1
    elif dataset == "C": # has problem: miss masks
        dataset = CitationFull(path, "DBLP", transform=T.NormalizeFeatures())
        print(len(dataset))
    elif args.dataset in ['Caltech36']:
        dataset = Facebook100(path, dataset, 'housing', transform=T.NormalizeFeatures())
    else: # normal
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    print(len(dataset))
    # print(dataset.num_classes)
    data = dataset[0]
    if args.dataset in ['Caltech36']:
        data = set_random_mask(data)
        data.y = data.y + 1
    print(data)
    # print(data.y)

    edge_attr = torch.ones(data.edge_index[0].size(0))
    eye = torch.eye(data.x.size(0)).to_sparse().to(device)
    oriadj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.clone(edge_attr)).to_torch_sparse_coo_tensor().to(device)
    scipy_oriadj = SparseTensor.from_torch_sparse_coo_tensor(oriadj + eye).to_scipy()
    io.mmwrite(f"./pretrain/{args.dataset}_oriadj.mtx", scipy_oriadj)

    ### My Partition
    # deg = degree(data.edge_index[0], dtype=torch.float)
    # print(deg, len(deg))

    # divide degree among different classes
    degree_split = [3, 5] # for num_class = 3
    data, n_subgraphs, class_graphs = my_partition_graph(data, degree_split, args.total_subgraphs, args.num_groups, dataset=args.dataset)
    n_subgraphs, n_classes, n_groups = my_get_boundary(n_subgraphs, class_graphs, args.num_groups)
    print(data)
    # print(data.train_mask, data.test_mask)
    print(class_graphs)
    print('n_subgraphs: ', n_subgraphs)
    print('n_classes: ', n_classes)
    print('n_groups: ', n_groups)
    # exit()
    ###

    ### PyG METIS
    # print(data)
    # data = cluster_graph(data, num_parts=args.group)
    # print(data)
    # exit()
    ###

    model, data = Net(dataset, data, args), data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=lrate)  # Only perform weight-decay on first convolution.
    best_val_acc = test_acc = 0
    for epoch in range(1, args.epochs):
        train(model,data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Pretrain Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    # print(model.state_dict().keys())
    # print('pretrain Epochs: ',args.epochs)
    if not osp.exists('./pretrain'):
        os.mkdir('./pretrain')
    torch.save({"state_dict":model.state_dict(),"adj":model.adj1, "data":data}, f"./pretrain/{args.dataset}_model.pth.tar")