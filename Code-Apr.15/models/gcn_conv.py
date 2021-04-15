from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import glorot, zeros

from .quantize import *
# from .massage_passing import MessagePassing

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul_(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul_(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
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
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, chunk_q: bool = False, **kwargs):

        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.quantize_input = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

        self.quantize_agg = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

        self.chunk_q = chunk_q

        if self.chunk_q is True:
            for i in range(6):
                _q_act = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)
                _q_agg = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)
                setattr(self, 'quantize_chunk_act_{}'.format(i), _q_act)
                setattr(self, 'quantize_chunk_agg_{}'.format(i), _q_agg)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, quant=False, num_act_bits=None, num_wei_bits=None, num_agg_bits=None,
                chunk_q=False, n_classes=None, n_subgraphs=None, act_quant_bits=None, agg_quant_bits=None) -> Tensor:
        """"""
        self.quant = quant
        self.num_act_bits = num_act_bits
        self.num_wei_bits = num_wei_bits
        self.num_agg_bits = num_agg_bits
        self.chunk_q = chunk_q
        self.n_classes = n_classes
        self.n_subgraphs = n_subgraphs
        self.act_quant_bits = act_quant_bits
        self.agg_quant_bits = agg_quant_bits

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
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if self.quant:
            # quantize weight
            weight_qparams = calculate_qparams(
                self.weight, num_bits=num_wei_bits, flatten_dims=(1, -1), reduce_dim=None)
            qweight = quantize(self.weight, qparams=weight_qparams)
            # quantize input
            if self.chunk_q:
                # chunk-based quantization
                qx_list = []
                qbit_list = []
                pre_limit = 0
                for i, bit in enumerate(self.act_quant_bits):
                    now_limit = self.n_classes[i]
                    _qx = getattr(self, 'quantize_chunk_act_{}'.format(i))(x[pre_limit: now_limit, :], bit)
                    qbit_list.append(bit * torch.ones(now_limit - pre_limit))
                    pre_limit = now_limit
                    qx_list.append(_qx)
                    # print(_qx.shape)
                qx = torch.cat(qx_list, 0)
                qbit = torch.cat(qbit_list, 0)
                mean_act_bits = torch.mean(qbit)
                # print('mean bits: ', torch.mean(qbit))
                # print(qx.shape)
                # exit()
            else:
                # uniform quantization
                qx = self.quantize_input(x, num_act_bits)
            x = torch.matmul(qx, qweight)
        else:
            x = torch.matmul(x, self.weight)

        # quantize the aggregation
        if self.quant:
            if self.chunk_q:
                qx_list = []
                qbit_list = []
                pre_limit = 0
                for i, bit in enumerate(self.agg_quant_bits):
                    now_limit = self.n_classes[i]
                    _qx = getattr(self, 'quantize_chunk_agg_{}'.format(i))(x[pre_limit: now_limit, :], bit)
                    qbit_list.append(bit * torch.ones(now_limit - pre_limit))
                    pre_limit = now_limit
                    qx_list.append(_qx)
                x = torch.cat(qx_list, 0)
                # qbit = torch.cat(qbit_list, 0)
                # mean_agg_bits = torch.mean(qbit)
            else:
                x = self.quantize_agg(x, self.num_agg_bits)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        # if self.bias is not None:
        #     if self.quant:
        #         # quantize bias
        #         qbias = quantize(
        #             self.bias, num_bits=num_act_bits,
        #             flatten_dims=(0, -1))
        #         out += qbias
        #         # quantize output
        #         out = quantize(out, num_bits=num_act_bits)
        #     else:
        #         out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        edge_weight_view = edge_weight.view(-1, 1)
        '''
        if self.quant:
            # quantize feature
            x_j = quantize(x_j, num_bits=self.num_bits, dequantize=True)
        '''
        return edge_weight_view * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
