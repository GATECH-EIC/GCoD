import os
import time
import torch
import numpy as np
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, DataParallel, GATConv, SAGEConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import DataParallel, global_mean_pool
from torch_geometric.utils import dense_to_sparse, degree
from torch_sparse import SparseTensor
from .utils import *
from tqdm import tqdm
from .gcn_conv import GCNConv
from .gat_conv import GATConv
from .gin_conv import GINConv
from .sage_conv import SAGEConv
from .sampler import NeighborSampler
from .quantize import *


__all__ = ['GCN', 'GAT', 'GIN', 'SAGE']


class my_QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, chunk_q=False):
        super(my_QLinear, self).__init__(in_features, out_features, bias)
        if chunk_q is True:
            for i in range(6):
                _q_act = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)
                setattr(self, 'quantize_chunk_act_{}'.format(i), _q_act)
        else:
            self.quantize_input = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

        self.chunk_q = chunk_q

    def forward(self, input, num_act_bits=None, num_wei_bits=None, act_quant_bits=None, n_classes=None):
        # self.quantize_input = QuantMeasure(num_bits)
        if self.chunk_q is True:
            # Chunk-based quantization
            qx_list = []
            pre_limit = 0
            for i, bit in enumerate(act_quant_bits):
                now_limit = n_classes[i]
                _qx = getattr(self, 'quantize_chunk_act_{}'.format(i))(input[pre_limit: now_limit, :], bit)
                pre_limit = now_limit
                qx_list.append(_qx)
            qinput = torch.cat(qx_list, 0)
        else:
            qinput = self.quantize_input(input, num_act_bits)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=num_wei_bits, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=num_act_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None
        output = F.linear(qinput, qweight, qbias)
        return output

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
    def __init__(self, dataset, data, args, adj=(), device='cpu', quant=False, num_act_bits=None, num_wei_bits=None, num_agg_bits=None,
                chunk_q=False, n_classes=None, n_subgraphs=None, chunk_q_mix=None, q_max=None, q_min=None):
        super(GCN, self).__init__()
        self.data = data
        self.quant = quant
        self.num_act_bits = num_act_bits
        self.num_wei_bits = num_wei_bits
        self.num_agg_bits = num_agg_bits
        self.chunk_q = chunk_q
        self.n_classes = n_classes
        self.n_subgraphs = n_subgraphs
        self.chunk_q_mix = chunk_q_mix
        self.q_max = q_max
        self.q_min = q_min
        if args.dataset == "NELL":
            hidden = 128
        else:
            hidden = 16 # 128

        num_classes = dataset.num_classes
        if args.dataset in ['Caltech36']:
            num_classes += 1
        self.conv1 = GCNConv(dataset.num_features, hidden,
                             normalize=not args.use_gdc, chunk_q=self.chunk_q)
        self.conv2 = GCNConv(hidden, num_classes,
                             normalize=not args.use_gdc, chunk_q=self.chunk_q)
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

        # chunk-based quantization bits
        if self.chunk_q is True:
            self.act_quant_bits, self.agg_quant_bits = self.get_chunk_quant_bits()
            print(self.act_quant_bits, self.agg_quant_bits)
            if self.chunk_q_mix:
                total_act_bits = 0
                total_agg_bits = 0
                for i in range(len(self.act_quant_bits)):
                    total_act_bits += self.act_quant_bits[i] * self.nodes_in_classes_list[i]
                    total_agg_bits += self.agg_quant_bits[i] * self.nodes_in_classes_list[i]
                print('mean bits for activation: {:.3f}'.format(total_act_bits / np.sum(self.nodes_in_classes_list)))
                print('mean bits for activation: {:.3f}'.format(total_act_bits / np.sum(self.nodes_in_classes_list)))
            else:
                print('mean bits for activation: {:.3f}'.format(np.mean(self.act_quant_bits)))
                print('mean bits for activation: {:.3f}'.format(np.mean(self.agg_quant_bits)))
        else:
            self.act_quant_bits, self.agg_quant_bits = None, None
            # exit()

        # haoran write w.r.t. PyG METIS
        # if len(adj) == 0:
        #     self.adj1 = self.data.adj.to_torch_sparse_coo_tensor().to(device)
        #     self.adj1 = self.adj1 + torch.eye(self.adj1.shape[0]).to_sparse().to(device)
        # else:
        #     self.adj1 = adj

        # self.id = torch.eye(self.adj1.shape[0]).to_sparse().to(device)
        # self.adj2 = self.adj1.clone()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def get_mean_act_bits(self):
        if self.chunk_q_mix:
            total_act_bits = 0
            for i in range(len(self.act_quant_bits)):
                total_act_bits += self.act_quant_bits[i] * self.nodes_in_classes_list[i]
            return total_act_bits / np.sum(self.nodes_in_classes_list)
        else:
            return np.mean(self.act_quant_bits)

    def get_mean_agg_bits(self):
        if self.chunk_q_mix:
            total_agg_bits = 0
            for i in range(len(self.agg_quant_bits)):
                total_agg_bits += self.agg_quant_bits[i] * self.nodes_in_classes_list[i]
            return total_agg_bits / np.sum(self.nodes_in_classes_list)
        else:
            return np.mean(self.agg_quant_bits)

    def get_chunk_quant_bits(self):
        # print(degree_list.shape)
        # print(torch.max(degree_list))
        # print(torch.mean(degree_list))
        # print(torch.min(degree_list))
        # print(degree_list[:self.n_classes[0]])
        # print(degree_list[self.n_classes[0]: self.n_classes[1]])
        # print(degree_list[self.n_classes[1]: self.n_classes[2]])
        # print(degree_list[self.n_classes[2]: self.n_classes[3]])

        if self.chunk_q_mix:
            adj = torch.clone(self.adj1).to_dense()
            degree_list = torch.sum(adj, dim=1)

            mean_in_degree_list = []
            self.nodes_in_classes_list = []
            pre_limit = 0
            for i, position in enumerate(self.n_classes):
                now_limit = position
                _degree = degree_list[pre_limit: now_limit]
                mean_in_degree_list.append(torch.mean(_degree))
                self.nodes_in_classes_list.append(now_limit - pre_limit)
                pre_limit = now_limit

            print(mean_in_degree_list)
            print(self.nodes_in_classes_list)

            # TODO:
            # map different bits w.r.t. the mean degrees
            # insights - high degree, high bits
            # act_q_max = 4
            # act_q_min = 2
            act_q_max = self.q_max
            act_q_min = self.q_min
            chunk_d_max = max(mean_in_degree_list)
            chunk_d_min = min(mean_in_degree_list)
            act_quant_bits = []
            for i in range(len(mean_in_degree_list)):
                _act_q = act_q_min + (mean_in_degree_list[i] - chunk_d_min) / (chunk_d_max - chunk_d_min) * (act_q_max - act_q_min)
                act_quant_bits.append(int(_act_q))

            # agg_q_max = 4
            # agg_q_min = 2
            agg_q_max = self.q_max
            agg_q_min = self.q_min
            agg_quant_bits = []
            for i in range(len(mean_in_degree_list)):
                _agg_q = agg_q_min + (mean_in_degree_list[i] - chunk_d_min) / (chunk_d_max - chunk_d_min) * (agg_q_max - agg_q_min)
                agg_quant_bits.append(int(_agg_q))

        else:
            act_quant_bits = []
            agg_quant_bits = []
            for i in range(len(self.n_classes)):
                act_quant_bits.append(self.num_act_bits)
                agg_quant_bits.append(self.num_agg_bits)

            assert len(act_quant_bits) == len(self.n_classes)
            assert len(agg_quant_bits) == len(self.n_classes)

        return act_quant_bits, agg_quant_bits

    def forward(self, return_time=False):

        if return_time is False:
            x = self.data.x
            # self.ei1, self.ew1 = self.adj1
            # self.ei2, self.ew2 = self.adj2
            x = F.relu(self.conv1(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj1 -self.id), quant=self.quant,
                                num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
                                chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj2 - self.id), quant=self.quant,
                                num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
                                chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)
            return F.log_softmax(x, dim=1)
        else:
            x = self.data.x
            edge_1 = SparseTensor.from_torch_sparse_coo_tensor(self.adj1 - self.id)
            edge_2 = SparseTensor.from_torch_sparse_coo_tensor(self.adj2 - self.id)
            start_time = time.time()
            x = F.relu(self.conv1(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj1 -self.id), quant=self.quant,
                                num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
                                chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits))
            # x = F.dropout(x, training=self.training)
            x = self.conv2(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj2 - self.id), quant=self.quant,
                                num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
                                chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)
            end_time = time.time()
            return end_time - start_time

class GAT(torch.nn.Module):
    def __init__(self, dataset, data, hidden_unit, heads, dropout=0.5, adj=(), device='cpu', quant=False,
                num_act_bits=None, num_wei_bits=None, num_agg_bits=None, num_att_bits=None,
                chunk_q=False, n_classes=None, n_subgraphs=None, chunk_q_mix=None, q_max=None, q_min=None):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.data = data
        self.adj = adj
        self.quant = quant
        self.num_act_bits = num_act_bits
        self.num_wei_bits = num_wei_bits
        self.num_agg_bits = num_agg_bits
        self.num_att_bits = num_att_bits
        self.chunk_q = chunk_q
        self.n_classes = n_classes
        self.n_subgraphs = n_subgraphs
        self.chunk_q_mix = chunk_q_mix
        self.q_max = q_max
        self.q_min = q_min
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

        # chunk-based quantization bits
        if self.chunk_q is True:
            self.act_quant_bits, self.agg_quant_bits = self.get_chunk_quant_bits()
            print(self.act_quant_bits, self.agg_quant_bits)
            if self.chunk_q_mix:
                total_act_bits = 0
                total_agg_bits = 0
                for i in range(len(self.act_quant_bits)):
                    total_act_bits += self.act_quant_bits[i] * self.nodes_in_classes_list[i]
                    total_agg_bits += self.agg_quant_bits[i] * self.nodes_in_classes_list[i]
                print('mean bits for activation: {:.3f}'.format(total_act_bits / np.sum(self.nodes_in_classes_list)))
                print('mean bits for activation: {:.3f}'.format(total_act_bits / np.sum(self.nodes_in_classes_list)))
            else:
                print('mean bits for activation: {:.3f}'.format(np.mean(self.act_quant_bits)))
                print('mean bits for activation: {:.3f}'.format(np.mean(self.agg_quant_bits)))
        else:
            self.act_quant_bits, self.agg_quant_bits = None, None
            # exit()

        self.conv1 = GATConv(
            dataset.num_features, hidden_unit, heads=heads, dropout=dropout, quant=quant, chunk_q=self.chunk_q)
        self.conv2 = GATConv(
            hidden_unit * heads, dataset.num_classes, heads=1, concat=False, dropout=dropout, quant=quant, chunk_q=self.chunk_q)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def get_mean_act_bits(self):
        if self.chunk_q_mix:
            total_act_bits = 0
            for i in range(len(self.act_quant_bits)):
                total_act_bits += self.act_quant_bits[i] * self.nodes_in_classes_list[i]
            return total_act_bits / np.sum(self.nodes_in_classes_list)
        else:
            return np.mean(self.act_quant_bits)

    def get_mean_agg_bits(self):
        if self.chunk_q_mix:
            total_agg_bits = 0
            for i in range(len(self.agg_quant_bits)):
                total_agg_bits += self.agg_quant_bits[i] * self.nodes_in_classes_list[i]
            return total_agg_bits / np.sum(self.nodes_in_classes_list)
        else:
            return np.mean(self.agg_quant_bits)

    def get_chunk_quant_bits(self):
        # print(degree_list.shape)
        # print(torch.max(degree_list))
        # print(torch.mean(degree_list))
        # print(torch.min(degree_list))
        # print(degree_list[:self.n_classes[0]])
        # print(degree_list[self.n_classes[0]: self.n_classes[1]])
        # print(degree_list[self.n_classes[1]: self.n_classes[2]])
        # print(degree_list[self.n_classes[2]: self.n_classes[3]])

        if self.chunk_q_mix:
            adj = torch.clone(self.adj1).to_dense()
            degree_list = torch.sum(adj, dim=1)

            mean_in_degree_list = []
            self.nodes_in_classes_list = []
            pre_limit = 0
            for i, position in enumerate(self.n_classes):
                now_limit = position
                _degree = degree_list[pre_limit: now_limit]
                mean_in_degree_list.append(torch.mean(_degree))
                self.nodes_in_classes_list.append(now_limit - pre_limit)
                pre_limit = now_limit

            print(mean_in_degree_list)
            print(self.nodes_in_classes_list)

            # TODO:
            # map different bits w.r.t. the mean degrees
            # insights - high degree, high bits
            # act_q_max = 4
            # act_q_min = 2
            act_q_max = self.q_max
            act_q_min = self.q_min
            chunk_d_max = max(mean_in_degree_list)
            chunk_d_min = min(mean_in_degree_list)
            act_quant_bits = []
            for i in range(len(mean_in_degree_list)):
                _act_q = act_q_min + (mean_in_degree_list[i] - chunk_d_min) / (chunk_d_max - chunk_d_min) * (act_q_max - act_q_min)
                act_quant_bits.append(int(_act_q))

            # agg_q_max = 4
            # agg_q_min = 2
            agg_q_max = self.q_max
            agg_q_min = self.q_min
            agg_quant_bits = []
            for i in range(len(mean_in_degree_list)):
                _agg_q = agg_q_min + (mean_in_degree_list[i] - chunk_d_min) / (chunk_d_max - chunk_d_min) * (agg_q_max - agg_q_min)
                agg_quant_bits.append(int(_agg_q))

        else:
            act_quant_bits = []
            agg_quant_bits = []
            for i in range(len(self.n_classes)):
                act_quant_bits.append(self.num_act_bits)
                agg_quant_bits.append(self.num_agg_bits)

            assert len(act_quant_bits) == len(self.n_classes)
            assert len(agg_quant_bits) == len(self.n_classes)

        return act_quant_bits, agg_quant_bits

    def forward(self, return_time=False):
        if return_time is True:
            x, edge_index = self.data.x, self.data.edge_index
            x = F.dropout(x, p=self.dropout, training=self.training)
            start_time = time.time()
            x = F.elu(self.conv1(x, edge_index, quant=self.quant,
                            num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits, num_att_bits=self.num_att_bits,
                            chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, quant=self.quant,
                            num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits, num_att_bits=self.num_att_bits,
                            chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)
            end_time = time.time()

            return end_time - start_time
        else:
            x, edge_index = self.data.x, self.data.edge_index
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv1(x, edge_index, quant=self.quant,
                            num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits, num_att_bits=self.num_att_bits,
                            chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, quant=self.quant,
                            num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits, num_att_bits=self.num_att_bits,
                            chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)

            return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, dataset, data, num_layers, hidden, adj=(), device='cpu', quant=False,
                num_act_bits=None, num_wei_bits=None, num_agg_bits=None, num_att_bits=None,
                chunk_q=False, n_classes=None, n_subgraphs=None, chunk_q_mix=None, q_max=None, q_min=None):
        super(GIN, self).__init__()

        self.data = data
        self.adj = adj
        self.quant = quant
        self.num_act_bits = num_act_bits
        self.num_wei_bits = num_wei_bits
        self.num_agg_bits = num_agg_bits
        self.num_att_bits = num_att_bits
        self.chunk_q = chunk_q
        self.n_classes = n_classes
        self.n_subgraphs = n_subgraphs
        self.chunk_q_mix = chunk_q_mix
        self.q_max = q_max
        self.q_min = q_min

        # self.edges = [torch.ones(data.edge_index.size()[1])]*2 # store the global edge weights
        # ref_adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=self.edges[0].to(device)).to_torch_sparse_coo_tensor()
        # self.id = SparseTensor.eye(ref_adj.shape[0]).to_torch_sparse_coo_tensor().to(device)
        # del ref_adj
        # self.adj1, self.adj2 = sparse_adj_from_weights(data, self, device)

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

        # self.conv1 = GINConv(
        #     Sequential(
        #         Linear(dataset.num_features, hidden),
        #         ReLU(),
        #         Linear(hidden, hidden),
        #         ReLU(),
        #         BN(hidden),
        #     ), train_eps=True, quant=quant, chunk_q=self.chunk_q)
        # self.convs = torch.nn.ModuleList()
        # for i in range(num_layers - 1):
        #     self.convs.append(
        #         GINConv(
        #             Sequential(
        #                 Linear(hidden, hidden),
        #                 ReLU(),
        #                 Linear(hidden, hidden),
        #                 ReLU(),
        #                 BN(hidden),
        #             ), train_eps=True, quant=quant, chunk_q=self.chunk_q))
        # self.lin1 = Linear(hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

        if not self.quant:
            self.conv1 = GINConv(Linear(dataset.num_features, hidden))
            self.conv2 = GINConv(Linear(hidden, dataset.num_classes))
            # self.lin = Linear(hidden, dataset.num_classes)
        else:
            self.conv1 = GINConv(my_QLinear(dataset.num_features, hidden, self.chunk_q), chunk_q=self.chunk_q)
            self.conv2 = GINConv(my_QLinear(hidden, dataset.num_classes, self.chunk_q), chunk_q=self.chunk_q)
            # self.lin = my_QLinear(hidden, dataset.num_classes, self.chunk_q)

        # chunk-based quantization bits
        if self.chunk_q is True:
            self.act_quant_bits, self.agg_quant_bits = self.get_chunk_quant_bits()
            print(self.act_quant_bits, self.agg_quant_bits)
            if self.chunk_q_mix:
                total_act_bits = 0
                total_agg_bits = 0
                for i in range(len(self.act_quant_bits)):
                    total_act_bits += self.act_quant_bits[i] * self.nodes_in_classes_list[i]
                    total_agg_bits += self.agg_quant_bits[i] * self.nodes_in_classes_list[i]
                print('mean bits for activation: {:.3f}'.format(total_act_bits / np.sum(self.nodes_in_classes_list)))
                print('mean bits for activation: {:.3f}'.format(total_act_bits / np.sum(self.nodes_in_classes_list)))
            else:
                print('mean bits for activation: {:.3f}'.format(np.mean(self.act_quant_bits)))
                print('mean bits for activation: {:.3f}'.format(np.mean(self.agg_quant_bits)))
        else:
            self.act_quant_bits, self.agg_quant_bits = None, None
            # exit()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        # self.lin.reset_parameters()


    def get_mean_act_bits(self):
        if self.chunk_q_mix:
            total_act_bits = 0
            for i in range(len(self.act_quant_bits)):
                total_act_bits += self.act_quant_bits[i] * self.nodes_in_classes_list[i]
            return total_act_bits / np.sum(self.nodes_in_classes_list)
        else:
            return np.mean(self.act_quant_bits)

    def get_mean_agg_bits(self):
        if self.chunk_q_mix:
            total_agg_bits = 0
            for i in range(len(self.agg_quant_bits)):
                total_agg_bits += self.agg_quant_bits[i] * self.nodes_in_classes_list[i]
            return total_agg_bits / np.sum(self.nodes_in_classes_list)
        else:
            return np.mean(self.agg_quant_bits)

    def get_chunk_quant_bits(self):
        # print(degree_list.shape)
        # print(torch.max(degree_list))
        # print(torch.mean(degree_list))
        # print(torch.min(degree_list))
        # print(degree_list[:self.n_classes[0]])
        # print(degree_list[self.n_classes[0]: self.n_classes[1]])
        # print(degree_list[self.n_classes[1]: self.n_classes[2]])
        # print(degree_list[self.n_classes[2]: self.n_classes[3]])

        if self.chunk_q_mix:
            adj = torch.clone(self.adj1).to_dense()
            degree_list = torch.sum(adj, dim=1)

            mean_in_degree_list = []
            self.nodes_in_classes_list = []
            pre_limit = 0
            for i, position in enumerate(self.n_classes):
                now_limit = position
                _degree = degree_list[pre_limit: now_limit]
                mean_in_degree_list.append(torch.mean(_degree))
                self.nodes_in_classes_list.append(now_limit - pre_limit)
                pre_limit = now_limit

            print(mean_in_degree_list)
            print(self.nodes_in_classes_list)

            # TODO:
            # map different bits w.r.t. the mean degrees
            # insights - high degree, high bits
            # act_q_max = 4
            # act_q_min = 2
            act_q_max = self.q_max
            act_q_min = self.q_min
            chunk_d_max = max(mean_in_degree_list)
            chunk_d_min = min(mean_in_degree_list)
            act_quant_bits = []
            for i in range(len(mean_in_degree_list)):
                _act_q = act_q_min + (mean_in_degree_list[i] - chunk_d_min) / (chunk_d_max - chunk_d_min) * (act_q_max - act_q_min)
                act_quant_bits.append(int(_act_q))

            # agg_q_max = 4
            # agg_q_min = 2
            agg_q_max = self.q_max
            agg_q_min = self.q_min
            agg_quant_bits = []
            for i in range(len(mean_in_degree_list)):
                _agg_q = agg_q_min + (mean_in_degree_list[i] - chunk_d_min) / (chunk_d_max - chunk_d_min) * (agg_q_max - agg_q_min)
                agg_quant_bits.append(int(_agg_q))

        else:
            act_quant_bits = []
            agg_quant_bits = []
            for i in range(len(self.n_classes)):
                act_quant_bits.append(self.num_act_bits)
                agg_quant_bits.append(self.num_agg_bits)

            assert len(act_quant_bits) == len(self.n_classes)
            assert len(agg_quant_bits) == len(self.n_classes)

        return act_quant_bits, agg_quant_bits

    def forward(self, return_time=False):
        if return_time is True:
            x, edge_index = self.data.x, self.data.edge_index
            start_time = time.time()
            x = self.conv1(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj1 -self.id), quant=self.quant,
                        num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits, num_att_bits=self.num_att_bits,
                        chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)
            x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj2 -self.id), quant=self.quant,
                        num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits, num_att_bits=self.num_att_bits,
                        chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)
            # x = F.relu(x)
            # x = self.lin(x)
            end_time = time.time()
            return end_time - start_time

        else:
            x, edge_index = self.data.x, self.data.edge_index
            x = self.conv1(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj1 -self.id), quant=self.quant,
                        num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits, num_att_bits=self.num_att_bits,
                        chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)
            x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, SparseTensor.from_torch_sparse_coo_tensor(self.adj2 -self.id), quant=self.quant,
                        num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits, num_att_bits=self.num_att_bits,
                        chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)
            # x = F.relu(x)
            # x = self.lin(x)
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


def sparse_adj_from_weights(data, model, device):
    adj1 = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=model.edges[0].to(device)).to_torch_sparse_coo_tensor().to(device)
    adj1 += model.id
    adj2 = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=model.edges[1].to(device)).to_torch_sparse_coo_tensor().to(device)
    adj2 += model.id
    return (adj1, adj2)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, data, device='cpu', quant=False,
                num_act_bits=None, num_wei_bits=None, num_agg_bits=None,
                chunk_q=False, n_classes=None, n_subgraphs=None, chunk_q_mix=None, q_max=None, q_min=None):
        super(SAGE, self).__init__()

        self.num_layers = 2
        self.quant = quant
        self.num_act_bits = num_act_bits
        self.num_wei_bits = num_wei_bits
        self.num_agg_bits = num_agg_bits
        self.chunk_q = chunk_q
        self.n_classes = n_classes
        self.n_subgraphs = n_subgraphs
        self.chunk_q_mix = chunk_q_mix
        self.q_max = q_max
        self.q_min = q_min
        self.device = device

        self.data = data
        self.edges = [torch.ones(data.edge_index.size()[1])]*2 # store the global edge weights
        ref_adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=self.edges[0].to(device)).to_torch_sparse_coo_tensor()
        self.id = SparseTensor.eye(ref_adj.shape[0]).to_torch_sparse_coo_tensor().to(device)
        del ref_adj
        self.adj1, self.adj2 = sparse_adj_from_weights(data, self, device)

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, quant=quant, chunk_q=self.chunk_q))
        self.convs.append(SAGEConv(hidden_channels, out_channels, quant=quant, chunk_q=self.chunk_q))

        # chunk-based quantization bits
        if self.chunk_q is True:
            self.act_quant_bits, self.agg_quant_bits = self.get_chunk_quant_bits()
            print(self.act_quant_bits, self.agg_quant_bits)
            if self.chunk_q_mix:
                total_act_bits = 0
                total_agg_bits = 0
                for i in range(len(self.act_quant_bits)):
                    total_act_bits += self.act_quant_bits[i] * self.nodes_in_classes_list[i]
                    total_agg_bits += self.agg_quant_bits[i] * self.nodes_in_classes_list[i]
                print('mean bits for activation: {:.3f}'.format(total_act_bits / np.sum(self.nodes_in_classes_list)))
                print('mean bits for activation: {:.3f}'.format(total_act_bits / np.sum(self.nodes_in_classes_list)))
            else:
                print('mean bits for activation: {:.3f}'.format(np.mean(self.act_quant_bits)))
                print('mean bits for activation: {:.3f}'.format(np.mean(self.agg_quant_bits)))
        else:
            self.act_quant_bits, self.agg_quant_bits = None, None
            # exit()


    def get_mean_act_bits(self):
        if self.chunk_q_mix:
            total_act_bits = 0
            for i in range(len(self.act_quant_bits)):
                total_act_bits += self.act_quant_bits[i] * self.nodes_in_classes_list[i]
            return total_act_bits / np.sum(self.nodes_in_classes_list)
        else:
            return np.mean(self.act_quant_bits)

    def get_mean_agg_bits(self):
        if self.chunk_q_mix:
            total_agg_bits = 0
            for i in range(len(self.agg_quant_bits)):
                total_agg_bits += self.agg_quant_bits[i] * self.nodes_in_classes_list[i]
            return total_agg_bits / np.sum(self.nodes_in_classes_list)
        else:
            return np.mean(self.agg_quant_bits)

    def get_chunk_quant_bits(self):
        # print(degree_list.shape)
        # print(torch.max(degree_list))
        # print(torch.mean(degree_list))
        # print(torch.min(degree_list))
        # print(degree_list[:self.n_classes[0]])
        # print(degree_list[self.n_classes[0]: self.n_classes[1]])
        # print(degree_list[self.n_classes[1]: self.n_classes[2]])
        # print(degree_list[self.n_classes[2]: self.n_classes[3]])

        if self.chunk_q_mix:
            adj = torch.clone(self.adj1).to_dense()
            degree_list = torch.sum(adj, dim=1)

            mean_in_degree_list = []
            self.nodes_in_classes_list = []
            pre_limit = 0
            for i, position in enumerate(self.n_classes):
                now_limit = position
                _degree = degree_list[pre_limit: now_limit]
                mean_in_degree_list.append(torch.mean(_degree))
                self.nodes_in_classes_list.append(now_limit - pre_limit)
                pre_limit = now_limit

            print(mean_in_degree_list)
            print(self.nodes_in_classes_list)

            # TODO:
            # map different bits w.r.t. the mean degrees
            # insights - high degree, high bits
            # act_q_max = 4
            # act_q_min = 2
            act_q_max = self.q_max
            act_q_min = self.q_min
            chunk_d_max = max(mean_in_degree_list)
            chunk_d_min = min(mean_in_degree_list)
            act_quant_bits = []
            for i in range(len(mean_in_degree_list)):
                _act_q = act_q_min + (mean_in_degree_list[i] - chunk_d_min) / (chunk_d_max - chunk_d_min) * (act_q_max - act_q_min)
                act_quant_bits.append(int(_act_q))

            # agg_q_max = 4
            # agg_q_min = 2
            agg_q_max = self.q_max
            agg_q_min = self.q_min
            agg_quant_bits = []
            for i in range(len(mean_in_degree_list)):
                _agg_q = agg_q_min + (mean_in_degree_list[i] - chunk_d_min) / (chunk_d_max - chunk_d_min) * (agg_q_max - agg_q_min)
                agg_quant_bits.append(int(_agg_q))

        else:
            act_quant_bits = []
            agg_quant_bits = []
            for i in range(len(self.n_classes)):
                act_quant_bits.append(self.num_act_bits)
                agg_quant_bits.append(self.num_agg_bits)

            assert len(act_quant_bits) == len(self.n_classes)
            assert len(agg_quant_bits) == len(self.n_classes)

        return act_quant_bits, agg_quant_bits


    def forward(self, x, adjs, n_id):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.

        # edge_weights = [self.adj1.coalesce().values(), self.adj2.coalesce().values()]

        for i, (edge_index, e_id, size) in enumerate(adjs):

            # if i == 1:
            #     edge_attr = torch.ones(edge_index[0].size(0)).to(edge_index.device)
            #     adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.clone(edge_attr)).to_torch_sparse_coo_tensor().to(edge_index.device)
            #     # adj = adj + torch.eye(adj.shape[0]).to_sparse().to(adj.device)
            #     degree_list = torch.sum(adj.cpu().to_dense(), dim=0)
            #     print(degree_list)
            #     print(size[1])
            #     print(len(degree_list))
            #     exit()
            edge_weights = [(self.adj1 -  self.id).coalesce(), (self.adj2 - self.id).coalesce()]

            # x_target = x[:size[1]]  # Target nodes are always placed first.
            x_target = x[n_id[:size[1]]]  # Target nodes are always placed first.

            real_edge_indices = self.data.edge_index[:,e_id]
            mask_adj = SparseTensor.from_edge_index(real_edge_indices, sparse_sizes=edge_weights[i].size()).to_torch_sparse_coo_tensor() # create mask
            sampled_adj = SparseTensor.from_torch_sparse_coo_tensor((edge_weights[i] * mask_adj.float().to(self.device)).t())

            # x = self.convs[i]((x, x_target), edge_index, quant=self.quant, num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
            #     chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)

            x = self.convs[i](x, sampled_adj, quant=self.quant, num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
                chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, batch_size, device='cpu', return_time=False):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # subgraph_loader = NeighborSampler(self.adj1.coalesce().indices(), node_idx=None, sizes=[-1],
        #                                 batch_size=batch_size, shuffle=False,
        #                                 num_workers=12)

        subgraph_loader = NeighborSampler(self.data.edge_index, node_idx=None, sizes=[-1],
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=12)

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.

        edge_weights = [(self.adj1 -  self.id).coalesce(), (self.adj2 - self.id).coalesce()]

        if return_time is True:
            infer_time = []
            for i in range(self.num_layers):
                xs = []
                for batch_size, n_id, adj in subgraph_loader:
                    edge_index, e_id, size = adj.to(device)
                    # x = x_all[n_id].to(device)
                    x = x_all.to(device)

                    # x_target = x[:size[1]]
                    x_target = x_all[n_id[:size[1]]].to(device)
                    real_edge_indices = self.data.edge_index[:,e_id]
                    mask_adj = SparseTensor.from_edge_index(real_edge_indices, sparse_sizes=edge_weights[i].size()).to_torch_sparse_coo_tensor() # create mask
                    sampled_adj = SparseTensor.from_torch_sparse_coo_tensor((edge_weights[i] * mask_adj.float().to(device)).t())

                    # x = self.convs[i]((x, x_target), edge_index, quant=self.quant, num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
                    #     chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)

                    _infer_start = time.time()

                    x = self.convs[i](x, sampled_adj, quant=self.quant, num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
                        chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)

                    if i != self.num_layers - 1:
                        x = F.relu(x)

                    infer_time.append(time.time() - _infer_start)

                    # xs.append(x.cpu())
                    xs.append(x[n_id[:size[1]]].cpu())
                    pbar.update(batch_size)
                x_all = torch.cat(xs, dim=0)
            pbar.close()
            return sum(infer_time)
        else:
            for i in range(self.num_layers):
                xs = []
                for batch_size, n_id, adj in subgraph_loader:
                    edge_index, e_id, size = adj.to(device)
                    # x = x_all[n_id].to(device)
                    x = x_all.to(device)
                    # x_target = x[:size[1]]

                    x_target = x_all[n_id[:size[1]]].to(device)
                    real_edge_indices = self.data.edge_index[:,e_id]
                    mask_adj = SparseTensor.from_edge_index(real_edge_indices, sparse_sizes=edge_weights[i].size()).to_torch_sparse_coo_tensor() # create mask
                    sampled_adj = SparseTensor.from_torch_sparse_coo_tensor((edge_weights[i] * mask_adj.float().to(device)).t())

                    # x = self.convs[i]((x, x_target), edge_index, quant=self.quant, num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
                    #     chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)

                    x = self.convs[i](x, sampled_adj, quant=self.quant, num_act_bits=self.num_act_bits, num_wei_bits=self.num_wei_bits, num_agg_bits=self.num_agg_bits,
                        chunk_q=self.chunk_q, n_classes=self.n_classes, n_subgraphs=self.n_subgraphs, act_quant_bits=self.act_quant_bits, agg_quant_bits=self.agg_quant_bits)

                    if i != self.num_layers - 1:
                        x = F.relu(x)
                    # xs.append(x.cpu())
                    xs.append(x[n_id[:size[1]]].cpu())
                    pbar.update(batch_size)

                x_all = torch.cat(xs, dim=0)

            pbar.close()

            return x_all