import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Planetoid, TUDataset, Flickr, Coauthor, CitationFull
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, DataParallel
from utils import *
import numpy as np
from torch_geometric.utils import dense_to_sparse
from torch_sparse import SparseTensor

# device = torch.device("cpu")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        self.conv1 = GCNConv(dataset.num_features, hidden,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden, dataset.num_classes,
                             normalize=not args.use_gdc)
        # print(adj)
        if data.edge_attr == None:
            data.edge_attr = torch.ones(data.edge_index[0].size(0))
        if len(adj) == 0:
            self.adj1 = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.clone(data.edge_attr)).to_torch_sparse_coo_tensor().to(device)
            self.adj1 = self.adj1 + torch.eye(self.adj1.shape[0]).to_sparse().to(device)
            # self.adj1 = (torch.clone(data.edge_index).to(device), torch.clone(data.edge_attr).to(device))
        else:
            self.adj1 = adj
        self.id = torch.eye(self.adj1.shape[0]).to_sparse().to(device)
        self.adj2 = self.adj1.clone()
        

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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--use_gdc', type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="CiteSeer") # choice: F: Flicker, C: DBLP, CiteSeer, Cora, Pumbed
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
    else: # normal
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    print(len(dataset))
    data = dataset[0]    
    print(data)
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
    torch.save(model.state_dict(), f"./pretrain_pytorch/{args.dataset}_model.pth.tar")