import os
import torch
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from get_boundary import my_get_bd
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
device = torch.device("cpu")

def mySaveFig(pltm,fntmp,fp=0,isax=0,iseps=0,isShowPic=0):
    if isax==1:
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        pltm.yticks(size=14)
        pltm.xticks(size=14)
    fnm='%s.png'%(fntmp)
    pltm.savefig(fnm, bbox_inches='tight')
    if iseps:
        fnm='%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        pltm.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic:
        pltm.show()
    else:
        pltm.close()


def plot_bd(ax, bd, adj_size, color):
    # print(bd)
    nb = len(bd)
    bd.insert(0, 0)
    bd.append(adj_size)
    for i in range(nb):
        x, y, z = bd[i], bd[i + 1], bd[i + 2]
        ax.plot([x, z], [y, y], c=color, lw=5, alpha=0.6)
        ax.plot([y, y], [x, z], c=color, lw=5, alpha=0.6)

def plot_raw_adj(data):
    edge_attr = torch.ones(data.edge_index[0].size(0))
    eye = torch.eye(data.x.size(0)).to_sparse().to(device)
    oriadj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.clone(edge_attr)).to_torch_sparse_coo_tensor()
    oriadj = SparseTensor.from_torch_sparse_coo_tensor(oriadj + eye)
    row, col, value = oriadj.coo()
    adj_size = data.x.size(0)

    ax[0].scatter(row, col, s=10, alpha=0.5, c='b')
    ax[0].set_ylim([0, adj_size])
    ax[0].set_xlim([adj_size, 0])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Before optimization', fontsize=font_big, fontweight="bold", y=1.02)


def plot_optimzed_adj(checkpoint, bd, n_class, n_groups):
    data = checkpoint["data"].to(device)
    adj_size = data.x.size(0)
    row = data.edge_index[0]
    col = data.edge_index[1]

    bd1 = bd
    n_class = n_class
    n_groups = n_groups
    bd1, bd2, bd3 = my_get_bd(n_groups, n_class, bd1)

    # plot_bd(ax[0], bd1, adj_size, color='g')
    plot_bd(ax[1], bd2, adj_size, color='g')
    plot_bd(ax[1], bd3, adj_size, color='r')
    ax[1].scatter(row, col, s=10, alpha=0.5, c='b')
    ax[1].set_ylim([0, adj_size])
    ax[1].set_xlim([adj_size, 0])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('After optimization', fontsize=font_big, fontweight="bold", y=1.02)

def plot_hard_optimzed_adj(checkpoint, bd, n_class, n_groups):
    data = checkpoint["data"].to(device)
    adj_size = data.x.size(0)
    row = data.edge_index[0]
    col = data.edge_index[1]

    bd1 = bd
    n_class = n_class
    n_groups = n_groups
    bd1, bd2, bd3 = my_get_bd(n_groups, n_class, bd1)

    # plot_bd(ax[0], bd1, adj_size, color='g')
    plot_bd(ax[2], bd2, adj_size, color='g')
    plot_bd(ax[2], bd3, adj_size, color='r')
    ax[2].scatter(row, col, s=10, alpha=0.5, c='b')
    ax[2].set_ylim([0, adj_size])
    ax[2].set_xlim([adj_size, 0])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title('After optimization', fontsize=font_big, fontweight="bold", y=1.02)


def get_parameters_2_3_10(model, dataset):
    if model == 'GCN':
        if dataset == 'Cora':
            bd = [368, 394, 189, 181, 174, 404, 455, 193, 198, 152]
            n_class = [2, 3]
            n_groups = 2
        elif dataset == 'CiteSeer':
            bd = [1104, 192, 202, 49, 137, 1070, 237, 183, 54, 99]
            n_class = [1, 2, 2]
            n_groups = 2
        elif dataset == 'Pubmed':
            bd = [3543, 4059, 781, 728, 359, 3700, 4780, 755, 615, 397]
            n_class = [2, 1, 2]
            n_groups = 2
        else:
            print('No supports for {} dataset!'.format(dataset))
    elif model == 'GAT':
        if dataset == 'Cora':
            bd = [361, 423, 171, 199, 195, 420, 417, 177, 182, 163]
            n_class = [2, 3]
            n_groups = 2
        elif dataset == 'CiteSeer':
            bd = [1106, 192, 202, 84, 50, 1068, 237, 183, 152, 53]
            n_class = [1, 2, 2]
            n_groups = 2
        elif dataset == 'Pubmed':
            bd = [3543, 4059, 781, 361, 577, 3700, 4780, 755, 393, 768]
            n_class = [2, 1, 2]
            n_groups = 2
        else:
            print('No supports for {} dataset!'.format(dataset))
    else:
        print('No supports for {} model!'.format(model))

    return bd, n_class, n_groups


def get_parameters_2_3_12(model, dataset):
    if model == 'GCN':
        if dataset == 'Cora':
            # bd = [368, 394, 118, 128, 137, 132, 404, 455, 159, 142, 142, 129]
            bd = [361, 423, 124, 118, 125, 157, 420, 417, 131, 143, 146, 143]
            n_class = [2, 4]
            n_groups = 2
        elif dataset == 'CiteSeer':
            # bd = [565, 547, 192, 202, 58, 97, 533, 529, 237, 183, 52, 132]
            bd = [556, 526, 192, 202, 75, 134, 538, 554, 237, 183, 53, 77]
            n_class = [2, 2, 2]
            n_groups = 2
        elif dataset == 'Pubmed':
            # bd = [3768, 3867, 781, 279, 225, 444, 4292, 4155, 755, 251, 584, 316]
            bd = [3768, 3867, 781, 279, 225, 444, 4292, 4155, 755, 251, 584, 316]
            n_class = [2, 1, 3]
            n_groups = 2
        else:
            print('No supports for {} dataset!'.format(dataset))
    elif model == 'GAT':
        if dataset == 'Cora':
            bd = [361, 423, 127, 142, 155, 137, 420, 417, 142, 129, 101, 154]
            n_class = [2, 4]
            n_groups = 2
        elif dataset == 'CiteSeer':
            bd = [565, 547, 192, 202, 78, 72, 533, 529, 237, 183, 51, 138]
            n_class = [2, 2, 2]
            n_groups = 2
        elif dataset == 'Pubmed':
            bd = [3543, 4059, 781, 279, 225, 444, 3700, 4780, 755, 251, 584, 316]
            n_class = [2, 1, 3]
            n_groups = 2
        else:
            print('No supports for {} dataset!'.format(dataset))
    else:
        print('No supports for {} model!'.format(model))

    return bd, n_class, n_groups

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GIN'])
parser.add_argument('--dataset', type=str, default="CiteSeer", choices=['Cora', 'CiteSeer', 'Pubmed'])
parser.add_argument('--quant', action='store_true', default=False)
parser.add_argument('--num_bits', type=int, default=6, help='will quantize to num_bits if enable')
args = parser.parse_args()

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.dataset)
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]

if args.quant is False:
    checkpoint_1 = torch.load('./graph_tune/{}_{}/ckpt.pth.tar'.format(
                            args.model, args.dataset))
    checkpoint_2 = torch.load('./graph_tune/{}_{}/ckpt_hard.pth.tar'.format(
                            args.model, args.dataset))
else:
    if args.model != 'GAT':
        checkpoint_1 = torch.load('./graph_tune/{}_{}_ACT_{}-bit_WEI_{}-bit_AGG_{}-bit/ckpt.pth.tar'.format(
                                args.model, args.dataset, args.num_bits, args.num_bits, args.num_bits), map_location='cpu')
        checkpoint_2 = torch.load('./graph_tune/{}_{}_ACT_{}-bit_WEI_{}-bit_AGG_{}-bit/ckpt_hard.pth.tar'.format(
                                args.model, args.dataset, args.num_bits, args.num_bits, args.num_bits), map_location='cpu')
    else:
        checkpoint_1 = torch.load('./graph_tune/{}_{}_ACT_{}-bit_WEI_{}-bit_AGG_{}-bit_ATT_32-bit/ckpt.pth.tar'.format(
                                args.model, args.dataset, args.num_bits, args.num_bits, args.num_bits), map_location='cpu')
        checkpoint_2 = torch.load('./graph_tune/{}_{}_ACT_{}-bit_WEI_{}-bit_AGG_{}-bit_ATT_32-bit/ckpt_hard.pth.tar'.format(
                                args.model, args.dataset, args.num_bits, args.num_bits, args.num_bits), map_location='cpu')

font_big = 34
font_mid = 34
font_leg = 33
font_small = 30

fig, ax = plt.subplots(1, 3, figsize=(30, 10))
plt.subplots_adjust(wspace=0.1, hspace=0.25)
plt.gca().invert_yaxis()

plot_raw_adj(data)
bd, n_class, n_groups = get_parameters_2_3_12(args.model, args.dataset)
plot_optimzed_adj(checkpoint_1, bd=bd, n_class=n_class, n_groups=n_groups)
bd, n_class, n_groups = get_parameters_2_3_12(args.model, args.dataset)
plot_hard_optimzed_adj(checkpoint_2, bd=bd, n_class=n_class, n_groups=n_groups)

for i in range(3):
    ax[i].spines['bottom'].set_linewidth(4)
    ax[i].spines['bottom'].set_color('black')
    ax[i].spines['left'].set_linewidth(4)
    ax[i].spines['left'].set_color('black')
    ax[i].spines['top'].set_linewidth(4)
    ax[i].spines['top'].set_color('black')
    ax[i].spines['right'].set_linewidth(4)
    ax[i].spines['right'].set_color('black')

# fig.text(0.5, 0.025, args.dataset, fontsize=font_big, fontweight='bold', ha='center', va='center')

if os.path.exists('./adj_visual') is False:
    os.makedirs('./adj_visual')

if args.quant is False:
    mySaveFig(plt, './adj_visual_sep/{}_{}'.format(args.model, args.dataset), isax=0, fp=1, isShowPic=0)
else:
    mySaveFig(plt, './adj_visual_sep/{}_{}_{}-bit'.format(args.model, args.dataset, args.num_bits), isax=0, fp=1, isShowPic=0)