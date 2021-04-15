import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from reddit_dataset import Reddit
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from utils import *
import numpy as np
from sampler import NeighborSampler
from tqdm import tqdm
from get_reddit_partition import my_partition_graph
from get_boundary import my_get_boundary

device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
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

def train(model, epoch, optimizer,x,y):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        # print(batch_size)
        # print(n_id)
        # print(data.edge_index[:,adjs[0][1][0]],adjs[0][1])

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test(model,x,y):
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--use_gdc', type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="reddit")
    parser.add_argument('--num_groups', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--total_subgraphs', type=int, default=140)
    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = args.dataset

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    # dataset = Reddit(path)
    # print(dataset.num_features, dataset.num_classes)
    # exit()
    # data = dataset[0]

    ###
    degree_split = [100, 200, 300, 500, 800, 1000, 1200, 1400, 1600, 2000, 2500, 3000, 4000]
    data, n_subgraphs, class_graphs = my_partition_graph(degree_split, args.total_subgraphs, args.num_groups, dataset=args.dataset)
    n_subgraphs, n_classes, n_groups = my_get_boundary(n_subgraphs, class_graphs, args.num_groups)
    print(data)
    # print(data.train_mask, data.test_mask)
    print(class_graphs)
    print('n_subgraphs: ', n_subgraphs)
    print('n_classes: ', n_classes)
    print('n_groups: ', n_groups)
    # exit()
    ###

    ###
    # edge_attr = torch.ones(data.edge_index[0].size(0))
    # eye = torch.eye(data.x.size(0)).to_sparse().to(device)
    # oriadj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.clone(edge_attr)).to_torch_sparse_coo_tensor().to(device)
    # scipy_oriadj = SparseTensor.from_torch_sparse_coo_tensor(oriadj + eye).to_scipy()
    # io.mmwrite(f"./pretrain/{args.dataset}_oriadj.mtx", scipy_oriadj)
    ###

    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                sizes=[25, 10], batch_size=1024, shuffle=True,
                                num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                    batch_size=1024, shuffle=False,
                                    num_workers=12)

    # model = SAGE(dataset.num_features, 256, dataset.num_classes)
    model = SAGE(602, 256, 41)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    print(x)

    for epoch in range(1, 11):
        loss, acc = train(model, epoch, optimizer, x, y)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test(model, x, y)
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}')

    torch.save({"state_dict":model.state_dict()}, f"./pretrain/{args.dataset}_model.pth.tar")
    torch.save({"state_dict":model.state_dict(), "data":data}, f"./pretrain/{args.dataset}_model_data.pth.tar")