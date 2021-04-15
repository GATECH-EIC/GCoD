from torch_geometric.datasets import Planetoid, TUDataset, Flickr, Coauthor, CitationFull
import argparse
import os
import torch_geometric.transforms as T
import dgl
import torch
from dgl.distributed import partition_graph
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from scipy import sparse, io
from dgl.data import RedditDataset, CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

def my_partition_graph(degree_split, tot_subgraphs, tot_groups, dataset='CiteSeer'):

    if dataset == 'reddit':
        data = RedditDataset()
        g = data[0]
    elif dataset == 'cora':
        data = CoraGraphDataset()
        g = data[0]
    elif dataset == 'citeseer':
        data = CiteseerGraphDataset()
        g = data[0]
    elif dataset == 'pubmed':
        data = PubmedGraphDataset()
        g = data[0]

    assert(tot_subgraphs % tot_groups == 0, 'tot_subgraph should be devided by tot_groups')
    tot_subgraphs //= tot_groups

    # e = data.edge_index
    # u, v = e[0], e[1]
    # train_mask = data.train_mask
    # test_mask = data.test_mask
    # val_mask = data.val_mask
    # x = data.x
    # y = data.y

    # g = dgl.graph((u, v))
    g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)

    # g.ndata['train_mask'] = train_mask
    # g.ndata['test_mask'] = test_mask
    # g.ndata['val_mask'] = val_mask
    g.ndata['x'] = g.ndata['feat']
    g.ndata['y'] = g.ndata['label']
    g.ndata['in_deg'] = g.in_degrees()

    n_node = g.num_nodes()
    n_edge = g.num_edges()
    avg_edge = n_edge / tot_subgraphs
    print(n_node, n_edge)

    n_filter = len(degree_split) + 1
    degree_split.insert(0, 0)
    degree_split.append(n_node)
    print(degree_split)

    clusters = []
    new_nodes = []
    n_in_edges = []

    for i in range(n_filter):
        nodes = g.filter_nodes(lambda x: torch.logical_and(degree_split[i] <= x.data['in_deg'], x.data['in_deg'] < degree_split[i + 1]))
        print(nodes.shape)
        clusters.append(g.subgraph(nodes))
        new_nodes.extend(nodes)
        n_in_edges.append(g.ndata['in_deg'][nodes].sum())

    print(len(new_nodes))
    # g = g.subgraph(new_nodes)

    n_subgraph = [0] * n_filter
    reminder = [0] * n_filter
    tot = 0
    for i in range(n_filter):
        n_subgraph[i] = int(n_in_edges[i] / avg_edge)
        print(n_in_edges[i], avg_edge)
        reminder[i] = n_in_edges[i] - avg_edge * n_subgraph[i]
        tot += n_subgraph[i]

    idx = [i[0] for i in sorted(enumerate(reminder), key=lambda x: x[1], reverse=True)]
    for i in range(0, tot_subgraphs - tot):
        n_subgraph[idx[i % n_filter]] += 1

    # new_nodes = [[] for _ in range(tot_groups)]
    # new_train_mask = [[] for _ in range(tot_groups)]
    # new_test_mask = [[] for _ in range(tot_groups)]
    # new_val_mask = [[] for _ in range(tot_groups)]
    # new_x = [[] for _ in range(tot_groups)]
    # new_y = [[] for _ in range(tot_groups)]
    # new_graph_size = [[] for _ in range(tot_groups)]

    # for i in range(n_filter):
    #     partition_graph(clusters[i], 'metis', n_subgraph[i] * tot_groups, dataset,
    #                     reshuffle=True, balance_edges=True)
    #     print('class', i, 'has', clusters[i].num_nodes(), 'nodes')
    #     for j in range(n_subgraph[i] * tot_groups):
    #         subg, node_feat, _, _, _ = dgl.distributed.load_partition(dataset + '/metis.json', j)
    #         nodes = subg.filter_nodes(lambda x: x.data['inner_node'])
    #         new_graph_size[j % tot_groups].append(nodes.shape[0])
    #         new_nodes[j % tot_groups].extend(clusters[i].ndata[dgl.NID][subg.ndata['orig_id'][nodes]])
    #         new_train_mask[j % tot_groups].append(node_feat['train_mask'][nodes])
    #         new_test_mask[j % tot_groups].append(node_feat['test_mask'][nodes])
    #         new_val_mask[j % tot_groups].append(node_feat['val_mask'][nodes])
    #         new_x[j % tot_groups].append(node_feat['x'][nodes])
    #         new_y[j % tot_groups].append(node_feat['y'][nodes])

    # nodes = []
    # train_mask, test_mask, val_mask, x, y = [], [], [], [], []
    # graph_size = []
    # for i in range(tot_groups):
    #     nodes += new_nodes[i]
    #     train_mask += new_train_mask[i]
    #     test_mask += new_test_mask[i]
    #     val_mask += new_val_mask[i]
    #     x += new_x[i]
    #     y += new_y[i]
    #     graph_size += new_graph_size[i]

    # train_mask = torch.cat(train_mask, dim=0).astype(torch.bool)
    # test_mask = torch.cat(test_mask, dim=0).astype(torch.bool)
    # val_mask = torch.cat(val_mask, dim=0).astype(torch.bool)
    # x = torch.cat(x, dim=0)
    # y = torch.cat(y, dim=0)
    # # print(g.num_nodes(), g.num_edges())
    # # print(len(nodes))
    # # print(sorted(nodes))
    # g = g.subgraph(nodes)
    # print(g.num_nodes(), g.num_edges())
    # u, v = g.edges()
    # print(u.shape)
    # edge_index = torch.stack([u, v])
    # print(edge_index.shape)
    # data = Data(x=x, y=y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask, edge_index=edge_index)
    new_nodes = [[] for _ in range(tot_groups)]
    new_graph_size = [[] for _ in range(tot_groups)]

    for i in range(n_filter):
        partition_graph(clusters[i], 'metis', n_subgraph[i] * tot_groups, dataset,
                        reshuffle=True, balance_edges=True)
        print(f'class {i} has {clusters[i].num_nodes()} nodes')
        for j in range(n_subgraph[i] * tot_groups):
            subg, node_feat, _, _, _ = dgl.distributed.load_partition(dataset + '/metis.json', j)[:5]
            nodes = subg.filter_nodes(lambda x: x.data['inner_node'])
            new_graph_size[j % tot_groups].append(nodes.shape[0])
            new_nodes[j % tot_groups].extend(clusters[i].ndata[dgl.NID][subg.ndata['orig_id'][nodes]])

    nodes = []
    graph_size = []
    for i in range(tot_groups):
        nodes += new_nodes[i]
        graph_size += new_graph_size[i]

    nodes = torch.stack(nodes)

    g = g.subgraph(nodes)
    u, v = g.edges()
    edge_index = torch.stack([u, v])
    data = Data(x=g.ndata['x'], y=g.ndata['y'], train_mask=g.ndata['train_mask'], test_mask=g.ndata['test_mask'],
                val_mask=g.ndata['val_mask'], edge_index=edge_index)
    return data, graph_size, n_subgraph


def save_adj(data, save_name):
    edge_attr = torch.ones(data.edge_index[0].size(0))
    eye = torch.eye(data.x.size(0)).to_sparse().to(device)
    oriadj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.clone(edge_attr)).to_torch_sparse_coo_tensor().to(device)
    scipy_oriadj = SparseTensor.from_torch_sparse_coo_tensor(oriadj + eye).to_scipy()
    io.mmwrite(f"./partition/{save_name}.mtx", scipy_oriadj)

if __name__ == "__main__":
    degree_split = [100, 200, 300, 500, 800, 1000, 1200, 1400, 1600, 2000, 2500, 3000, 4000]
    tot_subgraphs = 140
    tot_groups = 2

    # create save dir
    if not os.path.exists('./partition/'):
        os.mkdir('./partition/')


    device = torch.device("cpu")

    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'reddit')
    # dataset = Planetoid(path, 'reddit', transform=T.NormalizeFeatures())
    # data = dataset[0]

    # print(data)
    # save_adj(data, 'before')

    data, n_subgraph, n_class = my_partition_graph(degree_split, tot_subgraphs, tot_groups, dataset='reddit')

    print(data)
    print(n_subgraph, sum(n_subgraph))
    print(n_class)
    # save_adj(data, 'after')

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--use_gdc', type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="CiteSeer") # choice: F: Flicker, C: DBLP, CiteSeer, Cora, Pumbed
    args = parser.parse_args()

    degree_split = [3, 5]
    tot_subgraph = 10
    tot_groups = 2

    assert(tot_subgraph % tot_groups == 0, 'tot_subgraph should be devided by tot_groups')

    tot_subgraph //= tot_groups

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = args.dataset

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
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

    e = dataset[0].edge_index
    u, v = e[0], e[1]
    feat = dataset[0].x

    g = dgl.graph((u, v))

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    g.ndata['feat'] = feat
    g.ndata['in_deg'] = g.in_degrees()

    n_node = g.num_nodes()
    n_edge = g.num_edges()

    avg_edge = n_edge / tot_subgraph

    print(n_node, n_edge)

    n_filter = len(degree_split) + 1
    degree_split.insert(0, 0)
    degree_split.append(n_node)
    print(degree_split)

    clusters = []
    new_nodes = []
    n_in_edges = []

    for i in range(n_filter):
        nodes = g.filter_nodes(lambda x: torch.logical_and(degree_split[i] <= x.data['in_deg'], x.data['in_deg'] < degree_split[i + 1]))
        print(nodes.shape)
        clusters.append(g.subgraph(nodes))
        new_nodes.extend(nodes)
        n_in_edges.append(g.ndata['in_deg'][nodes].sum())

    print(len(new_nodes))
    g = g.subgraph(new_nodes)

    n_subgraph = [0] * n_filter
    reminder = [0] * n_filter
    tot = 0
    for i in range(n_filter):
        n_subgraph[i] = int(n_in_edges[i] / avg_edge)
        print(n_in_edges[i], avg_edge)
        reminder[i] = n_in_edges[i] - avg_edge * n_subgraph[i]
        tot += n_subgraph[i]

    idx = [i[0] for i in sorted(enumerate(reminder), key=lambda x: x[1], reverse=True)]
    for i in range(0, tot_subgraph - tot):
        n_subgraph[idx[i % n_filter]] += 1

    new_nodes = [[] for _ in range(tot_groups)]
    new_feats = [[] for _ in range(tot_groups)]

    for i in range(n_filter):
        partition_graph(clusters[i], 'metis', n_subgraph[i] * tot_groups, args.dataset,
                        reshuffle=True, balance_edges=True)
        print('class', i, 'has', clusters[i].num_nodes(), 'nodes')
        for j in range(n_subgraph[i] * tot_groups):
            subg, node_feat, _, _, _ = dgl.distributed.load_partition(args.dataset + '/metis.json', j)
            nodes = subg.filter_nodes(lambda x: x.data['inner_node'])
            print(nodes.shape)
            new_nodes[j % tot_groups].extend(subg.ndata['orig_id'][nodes])
            new_feats[j % tot_groups].append(node_feat['feat'][nodes])

    nodes = []
    feats = []
    for i in range(tot_groups):
        nodes += new_nodes[i]
        feats += new_feats[i]

    feats = torch.cat(feats, dim=0)
    g = g.subgraph(nodes)
    u, v = g.edges()

    edge_index = torch.stack([u, v])
    data = Data(x=feats, edge_index=edge_index)
    print(data)
"""