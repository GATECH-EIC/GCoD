from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

from .quantize import *

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        # print("Adj: ", torch.isnan(adj_t.to_dense()).sum())
        deg = sum(adj_t, dim=1)
        # print("Deg: ", torch.isnan(deg).sum())
        deg_inv_sqrt = deg.pow_(-0.5)
        # print("Num inf: ", (deg_inv_sqrt == float('inf')).int().sum())
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        # print("Invert: ", torch.isnan(deg_inv_sqrt).sum())
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t, deg_inv_sqrt

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col], deg_inv_sqrt


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)
    or
    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),
    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, eps: float = 0.1, train_eps: bool = False, chunk_q: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConv, self).__init__(**kwargs)
        self.nn = nn
        self.cached = False
        self.improved = False
        self.add_self_loops = False
        self.normalize = True
        self._cached_adj_t = None
        self._cached_edge_index = None
        self.initial_eps = eps

        self.chunk_q = chunk_q

        self.quantize_agg = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

        if self.chunk_q is True:
            print('register quantization function !!!')
            for i in range(6):
                _q_agg = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)
                setattr(self, 'quantize_chunk_agg_{}'.format(i), _q_agg)


        if train_eps:
            print("--------------------")
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                quant=True, num_act_bits=None, num_wei_bits=None, num_att_bits=None, num_agg_bits=None,
                chunk_q=False, n_classes=None, n_subgraphs=None, act_quant_bits=None, agg_quant_bits=None,
                size: Size = None, edge_weight:OptTensor = None) -> Tensor:
        """"""
        self.quant = quant
        self.num_act_bits = num_act_bits
        self.num_wei_bits = num_wei_bits
        self.num_agg_bits = num_agg_bits
        self.num_att_bits = num_att_bits
        self.chunk_q = chunk_q
        self.n_classes = n_classes
        self.n_subgraphs = n_subgraphs
        self.act_quant_bits = act_quant_bits
        self.agg_quant_bits = agg_quant_bits

        # return self.nn(out)
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index, deg_inverse = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # quantize the aggregation
        if self.quant:
            if self.chunk_q:
                qx_list = []
                pre_limit = 0
                for i, bit in enumerate(self.agg_quant_bits):
                    now_limit = self.n_classes[i]
                    _qx = getattr(self, 'quantize_chunk_agg_{}'.format(i))(x[pre_limit: now_limit, :], bit)
                    pre_limit = now_limit
                    qx_list.append(_qx)
                x = torch.cat(qx_list, 0)
            else:
                x = self.quantize_agg(x, self.num_agg_bits)

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        # print(out.shape)
        # print(self.nn)
        x_r = x[1]
        # print(x_r.shape, deg_inverse.pow(2)[:,None].shape)
        # print(torch.isnan(deg_inverse.pow(2)[:,None]).sum())
        # print(deg_inverse.pow(2).sum())
        if x_r is not None:
            out += (1+self.eps) * x_r * deg_inverse.pow(2)[:,None]

        if self.quant:
            if self.chunk_q:
                out = self.nn(out, num_act_bits, num_wei_bits, self.act_quant_bits, self.n_classes)
            else:
                out = self.nn(out, num_act_bits, num_wei_bits)
        else:
            out = self.nn(out)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    # def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
    #     return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        # adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)
    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    #     return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)
    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINEConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        print(train_eps)
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)