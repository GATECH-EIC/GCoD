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
from torch_geometric.datasets import Planetoid, TUDataset, Flickr, Coauthor, CitationFull, NELL
from torch_geometric.data import Data, ClusterData, ClusterLoader
from torch_geometric.utils import to_dense_adj

from utils import *
from scipy import sparse, io
from get_reddit_partition import my_partition_graph
from get_boundary import my_get_boundary
from torch_sparse import SparseTensor
# from network import GCN, GAT
from models import GCN, GAT, GIN

from datasets import get_dataset

torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GIN'])
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--dataset', type=str, default="CiteSeer", choices=['reddit'])
parser.add_argument('--num_groups', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--total_subgraphs', type=int, default=12)
parser.add_argument('--infer_only',  action='store_true', default=False)
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:8'])
parser.add_argument('--save_prefix', type=str, default='./pretrain')
parser.add_argument('--partition', action='store_true', default=False)
parser.add_argument('--repeat', type=int, default=5, help='repeat run 5 times for default')
parser.add_argument('--quant', action='store_true', default=False)
parser.add_argument('--num_act_bits', type=int, default=32, help='will quantize node features if enable')
parser.add_argument('--num_wei_bits', type=int, default=32, help='will quantize weights if enable')
parser.add_argument('--num_agg_bits', type=int, default=32, help='will quantize aggregation if enable')
parser.add_argument('--num_att_bits', type=int, default=32, help='will quantize attention module if enable')
parser.add_argument('--enable_chunk_q', action='store_true', default=False, help='enable chunk based quantization')
parser.add_argument('--enable_chunk_q_mix', action='store_true', default=False, help='enable mixed precision chunk based quantization')
parser.add_argument('--q_max', type=int, default=4)
parser.add_argument('--q_min', type=int, default=2)
args = parser.parse_args()

device = torch.device(args.device)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)

class Dataset:
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes

global dataset
dataset = Dataset(602, 41)

# load dataset
if args.dataset in ['Cora', 'CiteSeer', 'Pubmed']:
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
elif args.dataset == 'IMDB-BIN':
    dataset = get_dataset('IMDB-BINARY', sparse=True, cleaned=False)
elif args.dataset == 'NELL':
    dataset = NELL(path, pre_transform=None)
    data = dataset[0]
    data.x = data.x.to_dense()[:, :5414]
    print(data)
    # global dataset
    dataset = Dataset(5414, 210)
# print('len of dataset: {}'.format(len(dataset)))
# exit()

def main_train(device):
    # create logging file
    global save_path
    if args.quant is False:
        save_path = os.path.join(args.save_prefix, '{}_{}'.format(args.model, args.dataset))
    else:
        if args.model == 'GCN' or args.model == 'GIN':
            save_path = os.path.join(args.save_prefix, '{}_{}_ACT_{}-bit_WEI_{}-bit_AGG_{}-bit'.format(args.model, args.dataset, args.num_act_bits, args.num_wei_bits, args.num_agg_bits))
        else:
            save_path = os.path.join(args.save_prefix, '{}_{}_ACT_{}-bit_WEI_{}-bit_AGG_{}-bit_ATT_{}-bit'.format(args.model, args.dataset, args.num_act_bits, args.num_wei_bits, args.num_agg_bits, args.num_att_bits))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.logger_file = os.path.join(save_path, 'log_train_{}_{}_{}.txt'.format(args.num_groups, args.num_classes, args.total_subgraphs))
    if os.path.exists(args.logger_file):
        os.remove(args.logger_file)
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    logging.info('device: {}'.format(args.device))
    logging.info('start training {}'.format(args.model))

    # partition the graphs
    global n_subgraphs, n_classes, n_groups
    if args.partition is True:
        degree_split = [500, 1000]
        data, n_subgraphs, class_graphs = my_partition_graph(degree_split, args.total_subgraphs, args.num_groups, dataset=args.dataset)
        n_subgraphs, n_classes, n_groups = my_get_boundary(n_subgraphs, class_graphs, args.num_groups)
        data = data.to(device)

        logging.info('dataset information: {}'.format(data))
        logging.info('n_subgraphs: {}'.format(n_subgraphs))
        logging.info('n_classes  : {}'.format(n_classes))
        logging.info('n_groups   : {}'.format(n_groups))
    else:
        data = data.to(device)
        n_subgraphs = None
        n_classes = None
        n_groups = None

    return None, data

    # repeat loop
    val_acc_list = []
    test_acc_list = []
    best_model = None
    for i in range(args.repeat):
        # load model
        if args.model == 'GCN':
            model = GCN(dataset, data, args, device=device, quant=args.quant, num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits,
                        chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min).to(device)
            model.reset_parameters()
            if args.dataset == 'NELL':
                optimizer = torch.optim.Adam([
                    dict(params=model.conv1.parameters(), weight_decay=1e-5),
                    dict(params=model.conv2.parameters(), weight_decay=1e-5)
                ], lr=0.01)
            else:
                optimizer = torch.optim.Adam([
                    dict(params=model.conv1.parameters(), weight_decay=5e-4),
                    dict(params=model.conv2.parameters(), weight_decay=0)
                ], lr=0.01)  # Only perform weight-decay on first convolution.

        if args.model == 'GAT':
            model = GAT(dataset, data, hidden_unit=8, heads=8, dropout=0.5, device=device,
                        quant=args.quant, num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits, num_att_bits=args.num_att_bits,
                        chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min).to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

        if args.model == 'GIN': # 32
            model = GIN(dataset, data, num_layers=2, hidden=32, device=device,
                        quant=args.quant, num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits, num_att_bits=args.num_att_bits,
                        chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min).to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) #Pubmed/Cora 0.01, CiteSeer: 0.05/0.03

        # train loop
        best_val_acc = best_test_acc = 0
        for epoch in range(1, args.epochs):
            train(model, optimizer, data)
            train_acc, val_acc, test_acc = test(model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                if i == 0:
                    best_model = model
            if i > 0 and test_acc > max(test_acc_list):
                best_model = model
            logging.info('Pretrain Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
                          epoch, train_acc, val_acc, test_acc))
            # print('Pretrain Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
            #               epoch, train_acc, val_acc, test_acc))
        logging.info('Best val. acc : {}'.format(best_val_acc))
        logging.info('Best test acc : {}'.format(best_test_acc))

        val_acc_list.append(best_val_acc)
        test_acc_list.append(best_test_acc)

    logging.info('val_acc_list: {}'.format(val_acc_list))
    logging.info('test_acc_list: {}'.format(test_acc_list))
    logging.info('mean val acc: {}'.format(np.mean(val_acc_list)))
    logging.info('std. val acc: {}'.format(np.std(val_acc_list)))
    logging.info('mean test acc: {}'.format(np.mean(test_acc_list)))
    logging.info('std. test acc: {}'.format(np.std(test_acc_list)))

    if args.quant and args.enable_chunk_q and args.enable_chunk_q_mix:
        mean_act_bits = model.get_mean_act_bits()
        mean_agg_bits = model.get_mean_agg_bits()

    f = open(os.path.join(args.save_prefix, 'summary.txt'), 'a+')
    if args.model == 'GCN' or args.model == 'GIN':
        if args.quant and args.enable_chunk_q and args.enable_chunk_q_mix:
            f.write("Model: {}, Dataset: {}, Act bits: {:.2f}, Wei bits: {}, Agg bits: {:.2f} --- Test Acc (mean): {:.2f} (std): {:.3f} \n".format(
                    args.model, args.dataset, mean_act_bits, args.num_wei_bits, mean_agg_bits, np.mean(test_acc_list) * 100, np.std(test_acc_list) * 100))
        else:
           f.write("Model: {}, Dataset: {}, Act bits: {}, Wei bits: {}, Agg bits: {} --- Test Acc (mean): {:.2f} (std): {:.3f} \n".format(
                    args.model, args.dataset, args.num_act_bits, args.num_wei_bits, args.num_agg_bits, np.mean(test_acc_list) * 100, np.std(test_acc_list) * 100))
    elif args.model == 'GAT':
        if args.quant and args.enable_chunk_q and args.enable_chunk_q_mix:
            f.write("Model: {}, Dataset: {}, Act bits: {:.2f}, Wei bits: {}, Agg bits: {:.2f} Att bits: {} --- Test Acc (mean): {:.2f} (std): {:.3f} \n".format(
                    args.model, args.dataset, mean_act_bits, args.num_wei_bits, mean_agg_bits, args.num_att_bits, np.mean(test_acc_list) * 100, np.std(test_acc_list) * 100))
        else:
            f.write("Model: {}, Dataset: {}, Act bits: {}, Wei bits: {}, Agg bits: {} Att bits: {} --- Test Acc (mean): {:.2f} (std): {:.3f} \n".format(
                    args.model, args.dataset, args.num_act_bits, args.num_wei_bits, args.num_agg_bits, args.num_att_bits, np.mean(test_acc_list) * 100, np.std(test_acc_list) * 100))
    else:
        print('no such model!')
        exit()

    return best_model, data

# def get_degree(edge_index):
#     row = edge_index[0]
#     col = edge_index[1]
#     edge_weight = torch.ones(row.size(0)).to(row.device)
#     adj = SparseTensor(row=row, col=col, value=torch.clone(edge_weight)).to_torch_sparse_coo_tensor()
#     degree_list = torch.sum(adj.to_dense(), dim=1)
#     return degree_list

def main_infer(dataset, data, device, model=None):
    if model is None:
        # create logging file
        if args.quant is False:
            save_path = os.path.join(args.save_prefix, '{}_{}'.format(args.model, args.dataset))
        else:
            save_path = os.path.join(args.save_prefix, '{}_{}_{}-bit'.format(args.model, args.dataset, args.num_bits))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args.logger_file = os.path.join(save_path, 'log_infer_{}_{}_{}.txt'.format(args.num_groups, args.num_classes, args.total_subgraphs))
        if os.path.exists(args.logger_file):
            os.remove(args.logger_file)
        handlers = [logging.FileHandler(args.logger_file, mode='w'),
                    logging.StreamHandler()]
        logging.basicConfig(level=logging.INFO,
                            datefmt='%m-%d-%y %H:%M',
                            format='%(asctime)s:%(message)s',
                            handlers=handlers)

        logging.info('device: {}'.format(args.device))

    logging.info('start inference {}'.format(args.model))

    data = data.to(device)

    if model is None:
        # load model
        n_subgraphs = None
        n_classes = None
        n_groups = None
        if args.model == 'GCN':
            model = GCN(dataset, data, args, device=device, quant=args.quant, num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits,
                        chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min).to(device)

        if args.model == 'GAT':
            model = GAT(dataset, data, hidden_unit=8, heads=8, dropout=0.5, device=device,
                        quant=args.quant, num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits, num_att_bits=args.num_att_bits,
                        chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min).to(device)

        if args.model == 'GIN':
            model = GIN(dataset, data, num_layers=2, hidden=32, device=device,
                        quant=args.quant, num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits, num_att_bits=args.num_att_bits,
                        chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min).to(device)

    inference_time = inference(model, data)
    logging.info('inference time: {}'.format(inference_time))


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model()
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(), []
    masks = ['train_mask', 'val_mask', 'test_mask']
    for i, (_, mask) in enumerate(data('train_mask', 'val_mask', 'test_mask')):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

@torch.no_grad()
def inference(model, data):
    model.eval()
    inference_time = model(return_time=True)
    return inference_time

if args.infer_only is True:
    main_infer(dataset, data, device)
else:
    model, data = main_train(device)
    main_infer(dataset, data, device, model=model)

    if args.partition is True:
        torch.save({"state_dict":model.state_dict(),"adj":model.adj1, "data":data,
                    "n_subgraphs": n_subgraphs, "n_classes": n_classes, "n_groups": n_groups},
                    save_path + '/ckpt.pth.tar')
    else:
        torch.save({"state_dict":model.state_dict(),"adj":model.adj1, "data":data},
                    save_path + '/ckpt.pth.tar')

