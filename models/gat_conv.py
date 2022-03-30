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
                 add_self_loops: bool = True, bias: bool = True, quant: bool = False,  chunk_q: bool = False, **kwargs):
        super(GATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.chunk_q = chunk_q

        if quant is True:
            if self.chunk_q is True:
                if isinstance(in_channels, int):
                    self.lin_l = my_QLinear(in_channels, heads * out_channels, bias=False, chunk_q=True)
                    self.lin_r = self.lin_l
                else:
                    self.lin_l = my_QLinear(in_channels[0], heads * out_channels, False, chunk_q=True)
                    self.lin_r = my_QLinear(in_channels[1], heads * out_channels, False, chunk_q=True)
            else:
                if isinstance(in_channels, int):
                    self.lin_l = my_QLinear(in_channels, heads * out_channels, bias=False)
                    self.lin_r = self.lin_l
                else:
                    self.lin_l = my_QLinear(in_channels[0], heads * out_channels, False)
                    self.lin_r = my_QLinear(in_channels[1], heads * out_channels, False)
        else:
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

        self.quantize_agg = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

        if self.chunk_q is True:
            for i in range(6):
                _q_agg = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)
                setattr(self, 'quantize_chunk_agg_{}'.format(i), _q_agg)

        self.reset_parameters()

    def reset_parameters(self):
        if self.chunk_q:
            pass
        else:
            glorot(self.lin_l.weight)
            glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, quant=True, num_act_bits=None, num_wei_bits=None, num_att_bits=None, num_agg_bits=None,
                chunk_q=False, n_classes=None, n_subgraphs=None, act_quant_bits=None, agg_quant_bits=None,
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
        self.num_act_bits = num_act_bits
        self.num_wei_bits = num_wei_bits
        self.num_agg_bits = num_agg_bits
        self.num_att_bits = num_att_bits
        self.chunk_q = chunk_q
        self.n_classes = n_classes
        self.n_subgraphs = n_subgraphs
        self.act_quant_bits = act_quant_bits
        self.agg_quant_bits = agg_quant_bits

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None

        if self.quant:
            att_l_qparams = calculate_qparams(
                self.att_l, num_bits=num_att_bits, flatten_dims=(1, -1), reduce_dim=None)
            qatt_l = quantize(self.att_l, qparams=att_l_qparams)
            att_r_qparams = calculate_qparams(
                self.att_r, num_bits=num_att_bits, flatten_dims=(1, -1), reduce_dim=None)
            qatt_r = quantize(self.att_r, qparams=att_r_qparams)


            if self.chunk_q:
                # Chunk-based quantization of activation
                if isinstance(x, Tensor):
                    assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
                    x_l = x_r = self.lin_l(x, num_act_bits, num_wei_bits, self.act_quant_bits, self.n_classes).view(-1, H, C)
                    alpha_l = alpha_r = (x_l * qatt_l).sum(dim=-1)
                else:
                    x_l, x_r = x[0], x[1]
                    assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
                    x_l = self.lin_l(x_l, num_act_bits, num_wei_bits, self.act_quant_bits, self.n_classes).view(-1, H, C)
                    alpha_l = (x_l * qatt_l).sum(dim=-1)
                    if x_r is not None:
                        x_r = self.lin_r(x_r, num_act_bits, num_wei_bits, self.act_quant_bits, self.n_classes).view(-1, H, C)
                        alpha_r = (x_r * qatt_r).sum(dim=-1)
            else:
                # Uniform quantization of activation
                if isinstance(x, Tensor):
                    assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
                    x_l = x_r = self.lin_l(x, num_act_bits, num_wei_bits).view(-1, H, C)
                    alpha_l = alpha_r = (x_l * qatt_l).sum(dim=-1)
                else:
                    x_l, x_r = x[0], x[1]
                    assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
                    x_l = self.lin_l(x_l, num_act_bits, num_wei_bits).view(-1, H, C)
                    alpha_l = (x_l * qatt_l).sum(dim=-1)
                    if x_r is not None:
                        x_r = self.lin_r(x_r, num_act_bits, num_wei_bits).view(-1, H, C)
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

        # quantize the aggregation
        if self.quant:
            if self.chunk_q:
                qx_l_list = []
                qx_r_list = []
                pre_limit = 0
                for i, bit in enumerate(self.agg_quant_bits):
                    now_limit = self.n_classes[i]
                    _qx_l = getattr(self, 'quantize_chunk_agg_{}'.format(i))(x_l[pre_limit: now_limit, :], bit)
                    _qx_r = getattr(self, 'quantize_chunk_agg_{}'.format(i))(x_r[pre_limit: now_limit, :], bit)
                    pre_limit = now_limit
                    qx_l_list.append(_qx_l)
                    qx_r_list.append(_qx_r)
                x_l = torch.cat(qx_l_list, 0)
                x_r = torch.cat(qx_r_list, 0)
            else:
                x_l = self.quantize_agg(x_l, self.num_agg_bits)
                x_r = self.quantize_agg(x_r, self.num_agg_bits)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

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
                    self.bias, num_bits=num_act_bits,
                    flatten_dims=(0, -1))
                out += qbias
            else:
                out += self.bias

        # if self.quant:
        #     # quantize output
        #     out = quantize(out, num_bits=num_act_bits)

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

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if self.quant:
            # quantize attention matrix
            alpha = quantize(alpha, num_bits=self.num_att_bits)
        return x_j * alpha.unsqueeze(-1)  # 每条边乘以权重后的embedding 每一行=每条边的target端

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
