import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DataParallel, GATConv, SAGEConv
from torch_geometric.utils import dense_to_sparse, degree
from torch_sparse import SparseTensor
from utils import *
from tqdm import tqdm

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


class GCN(torch.nn.Module):
    def __init__(self, dataset, data, args, adj=(), device='cpu'):
        super(GCN, self).__init__()
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
            data.edge_attr = torch.ones(data.edge_index[0].size(0)).to(device)
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


    def forward(self, return_time=False):
        if return_time is False:
            x = self.data.x
            # self.ei1, self.ew1 = self.adj1
            # self.ei2, self.ew2 = self.adj2
            x = F.relu(self.conv1(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj1 -self.id)))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj2 - self.id))
            return F.log_softmax(x, dim=1)
        else:
            x = self.data.x
            edge_1 = SparseTensor.from_torch_sparse_coo_tensor(self.adj1 - self.id)
            edge_2 = SparseTensor.from_torch_sparse_coo_tensor(self.adj2 - self.id)
            start_time = time.time()
            x = F.relu(self.conv1(x, edge_1))
            # x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_2)
            end_time = time.time()
            return end_time - start_time

class GAT(torch.nn.Module):
    def __init__(self, dataset, data, hidden_unit, heads, dropout=0.5, adj=(), device='cpu'):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.data = data
        self.adj = adj
        edge_index = self.data.edge_index
        if len(self.adj) == 0:
            if data.edge_attr == None:
                data.edge_attr = torch.ones(edge_index[0].size(0)).to(device)
            self.adj1 = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.clone(data.edge_attr)).to_torch_sparse_coo_tensor().to(device)
            self.adj1 = self.adj1 + torch.eye(self.adj1.shape[0]).to_sparse().to(device)
        else:
            self.adj1 = adj
        self.id = torch.eye(self.adj1.shape[0]).to_sparse().to(device)
        self.adj2 = self.adj1.clone()

        self.conv1 = GATConv(
            dataset.num_features, hidden_unit, heads=heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_unit * heads, dataset.num_classes, heads=1, concat=False, dropout=dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, return_time=False):
        if return_time is True:
            x, edge_index = self.data.x, self.data.edge_index
            x = F.dropout(x, p=self.dropout, training=self.training)
            start_time = time.time()
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            end_time = time.time()

            return end_time - start_time
        else:
            x, edge_index = self.data.x, self.data.edge_index
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, device='cpu', return_time=False):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.

        if return_time is True:
            start_time = time.time()
            for i in range(self.num_layers):
                xs = []
                for batch_size, n_id, adj in subgraph_loader:
                    edge_index, _, size = adj.to(device)
                    x = x_all[n_id].to(device)
                    x_target = x[:size[1]]
                    x = self.convs[i]((x, x_target), edge_index)
                    if i != self.num_layers - 1:
                        x = F.relu(x)
                    xs.append(x.cpu())
                    pbar.update(batch_size)
                x_all = torch.cat(xs, dim=0)
            pbar.close()
            end_time = time.time()
            return end_time - start_time
        else:
            for i in range(self.num_layers):
                xs = []
                for batch_size, n_id, adj in subgraph_loader:
                    edge_index, _, size = adj.to(device)
                    x = x_all[n_id].to(device)
                    x_target = x[:size[1]]
                    x = self.convs[i]((x, x_target), edge_index)
                    if i != self.num_layers - 1:
                        x = F.relu(x)
                    xs.append(x.cpu())

                    pbar.update(batch_size)

                x_all = torch.cat(xs, dim=0)

            pbar.close()

            return x_all