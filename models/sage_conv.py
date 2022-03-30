from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from .quantize import *

# class my_QLinear(nn.Linear):
#     """docstring for QConv2d."""

#     def __init__(self, in_features, out_features, bias=True):
#         super(my_QLinear, self).__init__(in_features, out_features, bias)
#         self.quantize_input = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

#     def forward(self, input, num_act_bits, num_wei_bits):
#         # self.quantize_input = QuantMeasure(num_bits)
#         qinput = self.quantize_input(input, num_act_bits)
#         weight_qparams = calculate_qparams(
#             self.weight, num_bits=num_wei_bits, flatten_dims=(1, -1), reduce_dim=None)
#         qweight = quantize(self.weight, qparams=weight_qparams)
#         if self.bias is not None:
#             qbias = quantize(
#                 self.bias, num_bits=num_act_bits,
#                 flatten_dims=(0, -1))
#         else:
#             qbias = None
#         output = F.linear(qinput, qweight, qbias)
#         return output

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


# add edge_weight

class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, quant=False, chunk_q: bool = False, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.chunk_q = chunk_q

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if quant is True:
            if self.chunk_q is True:
                self.lin_l = my_QLinear(in_channels[0], out_channels, bias=bias, chunk_q=self.chunk_q)
                self.lin_r = my_QLinear(in_channels[1], out_channels, bias=False, chunk_q=self.chunk_q)
            else:
                self.lin_l = my_QLinear(in_channels[0], out_channels, bias=bias)
                self.lin_r = my_QLinear(in_channels[1], out_channels, bias=False)
        else:
            self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.quantize_input = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

        self.quantize_agg = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

        if self.chunk_q is True:
            print('register quantization function !!!')
            for i in range(6):
                _q_agg = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)
                setattr(self, 'quantize_chunk_agg_{}'.format(i), _q_agg)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, target_id=None,
                edge_weight: OptTensor = None, size: Size = None, quant=False, num_act_bits=None, num_wei_bits=None, num_agg_bits=None,
                chunk_q=False, n_classes=None, n_subgraphs=None, act_quant_bits=None, agg_quant_bits=None,) -> Tensor:
        """"""

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        self.quant = quant
        self.num_act_bits = num_act_bits
        self.num_wei_bits = num_wei_bits
        self.num_agg_bits = num_agg_bits
        self.chunk_q = chunk_q
        self.n_classes = n_classes
        self.n_subgraphs = n_subgraphs
        self.act_quant_bits = act_quant_bits
        self.agg_quant_bits = agg_quant_bits

        if self.quant is True:
            if self.chunk_q is True:
                qx_0_list = []
                qx_1_list = []
                pre_limit = 0
                for i, bit in enumerate(self.agg_quant_bits):
                    now_limit = self.n_classes[i]
                    _qx_0 = getattr(self, 'quantize_chunk_agg_{}'.format(i))(x[0][pre_limit: now_limit, :], bit)
                    _qx_1 = getattr(self, 'quantize_chunk_agg_{}'.format(i))(x[1][pre_limit: now_limit, :], bit)
                    pre_limit = now_limit
                    qx_0_list.append(_qx_0)
                    qx_1_list.append(_qx_1)
                _x0 = torch.cat(qx_0_list, 0)
                _x1 = torch.cat(qx_1_list, 0)
                x = (_x0, _x1)
            else:
                _x0 = self.quantize_agg(x[0], num_agg_bits)
                _x1 = self.quantize_agg(x[1], num_agg_bits)
                x = (_x0, _x1)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        if target_id != None:
            out = out[target_id]

        if quant is True:
            if self.chunk_q:
                out = self.lin_l(out, num_act_bits, num_wei_bits, self.act_quant_bits, self.n_classes)
            else:
                out = self.lin_l(out, num_act_bits, num_wei_bits)
        else:
            out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            if quant is True:
                if self.chunk_q:
                    out += self.lin_r(x_r, num_act_bits, num_wei_bits, self.act_quant_bits, self.n_classes)
                else:
                    out += self.lin_r(x_r, num_act_bits, num_wei_bits)
            else:
                out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)