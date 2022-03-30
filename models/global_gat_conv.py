from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

from .quantize import *


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(GATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.quantize_input = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

        self.quant = None
        self.num_bits = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, quant=True, num_bits=None,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        self.quant = quant
        self.num_bits = num_bits

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None

        if self.quant:
            # quantize input
            x = self.quantize_input(x, num_bits)
            # quantize weight
            att_l_qparams = calculate_qparams(
                self.att_l, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
            qatt_l = quantize(self.att_l, qparams=att_l_qparams)
            att_r_qparams = calculate_qparams(
                self.att_r, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
            qatt_r = quantize(self.att_r, qparams=att_r_qparams)

            if isinstance(x, Tensor):
                assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
                x_l = x_r = self.lin_l(x).view(-1, H, C)
                alpha_l = alpha_r = (x_l * qatt_l).sum(dim=-1)
            else:
                x_l, x_r = x[0], x[1]
                assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
                x_l = self.lin_l(x_l).view(-1, H, C)
                alpha_l = (x_l * qatt_l).sum(dim=-1)
                if x_r is not None:
                    x_r = self.lin_r(x_r).view(-1, H, C)
                    alpha_r = (x_r * qatt_r).sum(dim=-1)
        else:
            if isinstance(x, Tensor):
                assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
                x_l = x_r = self.lin_l(x).view(-1, H, C)
                alpha_l = (x_l * self.att_l).sum(dim=-1)
                alpha_r = (x_r * self.att_r).sum(dim=-1)
            else:
                x_l, x_r = x[0], x[1]
                assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
                x_l = self.lin_l(x_l).view(-1, H, C)
                alpha_l = (x_l * self.att_l).sum(dim=-1)
                if x_r is not None:
                    x_r = self.lin_r(x_r).view(-1, H, C)
                    alpha_r = (x_r * self.att_r).sum(dim=-1)

        # add by haoran
        print('x_l: ', x_l.shape)
        print('x_r: ', x_r.shape)
        print('alpha_l: ', alpha_l.shape)
        print('alpha_r: ', alpha_r.shape)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                num_nodes = size[1] if size is not None else num_nodes
                num_nodes = x_r.size(0) if x_r is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        print('out: ', out.shape)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            if self.quant:
                # quantize bias
                qbias = quantize(
                    self.bias, num_bits=num_bits,
                    flatten_dims=(0, -1))
                out += qbias
            else:
                out += self.bias

        if self.quant:
            # quantize output
            out = quantize(out, num_bits=num_bits)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # x_j has shape [E, out_channels]

        print('x_j: ', x_j.shape)
        print('alpha_i', alpha_i.shape)
        print('alpha_j', alpha_j.shape)

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)

        print('alpha before softmax: ', alpha.shape)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha

        print('alpha: ', alpha.shape)
        print('alpha: ', alpha.unsqueeze(-1).shape)
        # exit()
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if self.quant:
            # quantize attention matrix
            alpha = quantize(alpha, num_bits=self.num_bits)
        return x_j * alpha.unsqueeze(-1)  # 每条边乘以权重后的embedding 每一行=每条边的target端

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
