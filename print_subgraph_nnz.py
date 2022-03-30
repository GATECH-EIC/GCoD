import os
import os.path as osp
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric.utils.num_nodes as geo_num_nodes
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, DataParallel, GATConv
from torch_geometric.utils import dense_to_sparse, degree
from torch_geometric.datasets import Planetoid, TUDataset, Flickr, Coauthor, CitationFull
from torch_geometric.data import Data, ClusterData, ClusterLoader
from torch_geometric.utils import to_dense_adj

from utils import *
from scipy import sparse, io
from get_partition import my_partition_graph
from get_boundary import my_get_boundary
from torch_sparse import SparseTensor
# from network import GCN, GAT
from models import GCN, GAT

torch.set_printoptions(profile="full")


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT'])
parser.add_argument('--dataset', type=str, default="CiteSeer", choices=['Cora', 'CiteSeer', 'Pubmed'])
args = parser.parse_args()

device = torch.device('cuda:0')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)

# load dataset
if args.dataset in ['Cora', 'CiteSeer', 'Pubmed']:
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
else:
    print('please check the dataset info!')
    exit()
data = dataset[0]
print('len of dataset: {}'.format(len(dataset)))


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

if args.model == 'GCN':
    if args.dataset == 'Cora':
        bd = [361, 423, 124, 118, 125, 157, 420, 417, 131, 143, 146, 143]
    elif args.dataset == 'CiteSeer':
        bd = [556, 526, 192, 202, 75, 134, 538, 554, 237, 183, 53, 77]
    elif args.dataset == 'Pubmed':
        bd = [3768, 3867, 781, 279, 225, 444, 4292, 4155, 755, 251, 584, 316]
    else:
        print('No supports for {} dataset!'.format(dataset))
elif args.model == 'GAT':
    if args.dataset == 'Cora':
        bd = [361, 423, 124, 118, 125, 157, 420, 417, 131, 143, 146, 143]
    elif args.dataset == 'CiteSeer':
        bd = [565, 547, 192, 202, 49, 137, 533, 529, 237, 183, 54, 99]
    elif args.dataset == 'Pubmed':
        bd = [3543, 4059, 781, 279, 225, 444, 3700, 4780, 755, 251, 584, 316]
    else:
        print('No supports for {} dataset!'.format(dataset))
else:
    print('No supports for {} model!'.format(model))

print('add self loop!')
data = data.to(device)
data.edge_attr = torch.ones(data.edge_index[0].size(0)).to(device)
adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.clone(data.edge_attr)).to_torch_sparse_coo_tensor().to(device)
adj = adj + torch.eye(adj.shape[0]).to_sparse().to(device)
_adj = SparseTensor.from_torch_sparse_coo_tensor(adj.cpu())
row, col, value = _adj.coo()
edge_index = torch.stack([row, col])

subgraph_nnz = count_subgraph_nnz(edge_index, n_subgraphs=bd)
print('subgraph_nnz: \n {}'.format(subgraph_nnz.astype(int)))