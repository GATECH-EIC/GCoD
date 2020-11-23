import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Planetoid,TUDataset, Flickr
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv  # noga
from utils import *
from sparse_pytorch_train import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--load_path', type=str, default="./graph_train/Cora_prune_20.pth.tar")
parser.add_argument("--use_state", type=bool,default=True)
# parser.add_argument("--save", type=str, default="model.pth.tar")
parser.add_argument('--save_dir', type=str, default='./retrain')
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument("--log", type=str, default="{:.4f}")
parser.add_argument("--dataset", type=str, default="CiteSeer")
parser.add_argument('--group', type=int, default=3)
parser.add_argument('--class', type=int, default=3)
args = parser.parse_args()

# args.load_path = './pretrain/CiteSeer_model.pth.tar'

dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
lrate = 0.01
if dataset == "F":
    dataset = Flickr(path, transform=T.NormalizeFeatures())
    print(len(dataset))
    lrate = 0.1
else:
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# print(f"Number of graphs in {dataset} dataset:", len(dataset))
data = dataset[0]
checkpoint = torch.load(args.load_path)
state_dict = checkpoint["state_dict"]
adj = checkpoint["adj"]
# Preload model with pruned weights and adj
data = cluster_graph(data, num_parts=args.group, save=False)
model, data = Net(dataset, data, args, adj=adj).to(device), data.to(device)
if args.use_state:
    model.load_state_dict(state_dict)

train_acc, val_acc, tmp_test_acc = test(model, data)
print('Loaded pruned model with accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(train_acc, val_acc, tmp_test_acc))
# print(log.format(train_acc, val_acc, tmp_test_acc))

# create save dir
if not osp.exists(args.save_dir):
    os.mkdir(args.save_dir)


optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=lrate)  # Only perform weight-decay on first convolution.

def get_conv_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, GCNConv):
            total += torch.sum(m.weight.data.eq(0))
    return total

def retrain(model, data): # keep nonzeros in weights
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()

    # zero_num = get_conv_zero_param(model)
    # print(f"Number of zero parameters: {zero_num}")
    # ------ don't update those pruned grads! -
    # for k, m in enumerate(model.modules()):
    #     # print(k, m)
    #     if isinstance(m, GCNConv):
    #         weight_copy = m.weight.data.abs().clone()
    #         mask = weight_copy.gt(0).float().to(device)
    #         m.weight.grad.data.mul_(mask)
    # -----------------------------------------

    optimizer.step()

for epoch in range(1, args.epochs):
    retrain(model, data)
    train_acc, val_acc, tmp_test_acc = test(model, data)
    best_val_acc = test_acc = 0
    if tmp_test_acc > test_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Retrain Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
    if(epoch==args.epochs-1):
        log_4_test = args.log
        print(log_4_test.format(tmp_test_acc))
# print(model.state_dict().keys())
torch.save(model.state_dict(), "{}/{}_retrain.pth.tar".format(args.save_dir, args.dataset))






