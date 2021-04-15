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
from get_partition import my_partition_graph
from get_boundary import my_get_boundary
from torch_sparse import SparseTensor
# from network import GCN, GAT, SAGE
from models import GCN, GAT, SAGE
from sampler import NeighborSampler
# from torch_geometric.data.sampler import NeighborSampler
from tqdm import tqdm

torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--model', type=str, default='GCN', choices=['GraphSAGE'])
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--dataset', type=str, default="CiteSeer", choices=['Cora', 'CiteSeer', 'Pubmed', 'NELL'])
parser.add_argument('--num_groups', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--total_subgraphs', type=int, default=12)
parser.add_argument('--infer_only',  action='store_true', default=False)
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:5', 'cuda:6', 'cuda:7', 'cuda:8'])
parser.add_argument('--save_prefix', type=str, default='./pretrain')
parser.add_argument('--partition', action='store_true', default=False)
parser.add_argument('--repeat', type=int, default=5, help='repeat run 5 times for default')
parser.add_argument('--quant', action='store_true', default=False)
parser.add_argument('--num_act_bits', type=int, default=32, help='will quantize node features if enable')
parser.add_argument('--num_wei_bits', type=int, default=32, help='will quantize weights if enable')
parser.add_argument('--num_agg_bits', type=int, default=32, help='will quantize aggregation if enable')
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

# load dataset
if args.dataset in ['Cora', 'CiteSeer', 'Pubmed']:
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
elif args.dataset == 'NELL':
    dataset = NELL(path, pre_transform=None)
    data = dataset[0]
    data.x = data.x.to_dense()[:, :5414]
    print(data)
    # global dataset
    dataset = Dataset(5414, 210)
else:
    print('please check the dataset info!')
    exit()

# print('len of dataset: {}'.format(len(dataset)))

def main_train(dataset, data, device):
    # create logging file
    global save_path
    if args.quant is False:
        save_path = os.path.join(args.save_prefix, '{}_{}'.format(args.model, args.dataset))
    else:
        save_path = os.path.join(args.save_prefix, '{}_{}_ACT_{}-bit_WEI_{}-bit_AGG_{}-bit'.format(args.model, args.dataset, args.num_act_bits, args.num_wei_bits, args.num_agg_bits))
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

    # save origin adjacency matrix
    edge_attr = torch.ones(data.edge_index[0].size(0))
    eye = torch.eye(data.x.size(0)).to_sparse().to(device)
    oriadj = SparseTensor(row=data.edge_index[0],
                          col=data.edge_index[1],
                          value=torch.clone(edge_attr)).to_torch_sparse_coo_tensor().to(device)
    scipy_oriadj = SparseTensor.from_torch_sparse_coo_tensor(oriadj + eye).to_scipy()
    io.mmwrite(save_path + '/{}_oriadj.mtx'.format(args.dataset), scipy_oriadj)

    logging.info('save origin adj matrix ...')

    # logging.info('data before partition: {}'.format(data.x[data.test_mask][:, :20]))

    # partition the graphs
    if args.partition is True:
        if args.dataset == 'Cora':
            degree_split = [4]
        elif args.dataset == 'CiteSeer':
            degree_split = [3, 6]
        elif args.dataset == 'Pubmed':
            degree_split = [7, 12]
        global n_subgraphs, n_classes, n_groups
        data, n_subgraphs, class_graphs = my_partition_graph(data, degree_split, args.total_subgraphs, args.num_groups, dataset=args.dataset)
        n_subgraphs, n_classes, n_groups = my_get_boundary(n_subgraphs, class_graphs, args.num_groups)
        data = data.to(device)

        logging.info('dataset information: {}'.format(data))
        logging.info('n_subgraphs: {}'.format(n_subgraphs))
        logging.info('n_classes  : {}'.format(n_classes))
        logging.info('n_groups   : {}'.format(n_groups))

        # logging.info('data after partition: {}'.format(data.x[data.test_mask][:10, :20]))

        # exit()

    else:
        data = data.to(device)
        n_subgraphs = None
        n_classes = None
        n_groups = None

    # repeat loop
    val_acc_list = []
    test_acc_list = []
    best_model = None
    for i in range(args.repeat):
        # load model
        if args.model == 'GraphSAGE':
            global train_loader
            global subgraph_loader
            train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                sizes=[25, 10], batch_size=64, shuffle=True,
                                num_workers=12)
            # subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
            #                     batch_size=64, shuffle=False,
            #                     num_workers=12)
            model = SAGE(dataset.num_features, 32, dataset.num_classes, data=data, device=device, quant=args.quant,
                        num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits,
                        chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            x = data.x.to(device)
            y = data.y.squeeze().to(device)


        # train loop
        best_val_acc = best_test_acc = 0
        for epoch in range(1, args.epochs):
            loss, acc = train(model, data, epoch, optimizer, x, y)
            train_acc, val_acc, test_acc = test(model, data, x, y, batch_size=64)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                if i == 0:
                    best_model = model
            if i > 0 and test_acc > max(test_acc_list):
                best_model = model
            logging.info('Pretrain Epoch: {:03d}, Loss: {:.4f}, Approx. Train: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
                          epoch, loss, acc, train_acc, val_acc, test_acc))
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

    f = open(os.path.join(args.save_prefix, 'summary.txt'), 'a+')
    if args.model == 'GraphSAGE':
        f.write("Model: {}, Dataset: {}, Act bits: {}, Wei bits: {}, Agg bits: {} --- Test Acc (mean): {:.2f} (std): {:.3f} \n".format(
                args.model, args.dataset, args.num_act_bits, args.num_wei_bits, args.num_agg_bits, np.mean(test_acc_list) * 100, np.std(test_acc_list) * 100))
    else:
        print('no such model!')
        exit()

    return best_model, data

def main_infer(dataset, data, device, model=None):
    if model is None:
        # create logging file
        save_path = os.path.join(args.save_prefix, '{}_{}'.format(args.model, args.dataset))
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
        if args.model == 'GraphSAGE':
            global train_loader
            global subgraph_loader
            train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                sizes=[25, 10], batch_size=64, shuffle=True,
                                num_workers=12)
            # subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
            #                     batch_size=64, shuffle=False,
            #                     num_workers=12)
            model = SAGE(dataset.num_features, 32, dataset.num_classes, data=data, device=device, quant=args.quant,
                        num_act_bits=args.num_act_bits, num_wei_bits=args.num_wei_bits, num_agg_bits=args.num_agg_bits,
                        chunk_q=args.enable_chunk_q, n_classes=n_classes, n_subgraphs=n_subgraphs, chunk_q_mix=args.enable_chunk_q_mix, q_max=args.q_max, q_min=args.q_min)
            model = model.to(device)
    x = data.x.to(device)
    y = data.y.squeeze().to(device)

    inference_time = inference(model, x, y, batch_size=64)
    logging.info('inference time: {}'.format(inference_time))


def train(model, data, epoch, optimizer, x, y):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description('Epoch {:03d}'.format(epoch))

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        # print(batch_size)
        # print(n_id.shape)
        # print(data.edge_index[:,adjs[0][1][0]],adjs[0][1])

        optimizer.zero_grad()
        # out = model(x[n_id], adjs)
        out = model(x, adjs, n_id)

        # loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss = F.nll_loss(out[n_id[:batch_size]], y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        # total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        total_correct += int(out[n_id[:batch_size]].argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc

@torch.no_grad()
def test(model, data, x, y, batch_size):
    model.eval()

    out = model.inference(x, batch_size, device=device)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results

@torch.no_grad()
def inference(model, x, y, batch_size):
    model.eval()
    inference_time = model.inference(x, batch_size, device=device, return_time=True)
    return inference_time

if args.infer_only is True:
    main_infer(dataset, data, device)
else:
    model, data = main_train(dataset, data, device)
    main_infer(dataset, data, device, model=model)

    if args.partition is True:
        torch.save({"state_dict":model.state_dict(),"adj":model.adj1, "data":data,
                    "n_subgraphs": n_subgraphs, "n_classes": n_classes, "n_groups": n_groups},
                    save_path + '/ckpt.pth.tar')
    else:
        torch.save({"state_dict":model.state_dict(),"adj":model.adj1, "data":data},
                    save_path + '/ckpt.pth.tar')