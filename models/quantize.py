from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,  reduce_type='mean', keepdim=False, true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)

        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        if num_bits == 32:
            return input

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**num_bits - 1.
        scale = qparams.range / (qmax - qmin)

        min_scale = torch.tensor(1e-8).expand_as(scale).to(scale.device)#.cuda()
        scale = torch.max(scale, min_scale)

        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad, flatten_dims=(1, -1))
    return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach()
                    if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

    if qparams:
        if qparams.num_bits:
            return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)
    elif num_bits:
        return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)

    return x


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
    if qparams:
        if qparams.num_bits:
            return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)
    elif num_bits:
        return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)

    return x


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.9, measure=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input, num_bits, qparams=None):

        if num_bits == 32:
            return input

        if self.training or self.measure:
            if qparams is None:
                qparams = calculate_qparams(
                    input, num_bits=num_bits, flatten_dims=self.flatten_dims, reduce_dim=0, reduce_type='extreme')
            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(
                    qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(
                    qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range,
                              zero_point=self.running_zero_point, num_bits=num_bits)
        if self.measure:
            return input
        else:
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               stochastic=self.stochastic, inplace=self.inplace)
            return q_input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, momentum=0.1, quant_act_forward=0, quant_act_backward=0, quant_grad_act_error=0, quant_grad_act_gc=0, weight_bits=0, fix_prec=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.quantize_input_fw = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), momentum=momentum)
        self.quantize_input_bw = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), momentum=momentum)
        self.quant_act_forward = quant_act_forward
        self.quant_act_backward = quant_act_backward
        self.quant_grad_act_error = quant_grad_act_error
        self.quant_grad_act_gc = quant_grad_act_gc
        self.weight_bits = weight_bits
        self.fix_prec = fix_prec
        self.stride = stride


    def forward(self, input, num_bits, num_grad_bits):
        if num_bits == 0:
            output = F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
            return output

        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=self.num_bits_weight + self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        if self.fix_prec:
            if self.quant_act_forward or self.quant_act_backward or self.quant_grad_act_error or self.quant_grad_act_gc or self.weight_bits:
                weight_qparams = calculate_qparams(self.weight, num_bits=self.weight_bits, flatten_dims=(1, -1), reduce_dim=None)
                qweight = quantize(self.weight, qparams=weight_qparams)

                qinput_fw = self.quantize_input_fw(input, self.quant_act_forward)
                qinput_bw = self.quantize_input_bw(input, self.quant_act_backward)

                error_bits = self.quant_grad_act_error
                gc_bits = self.quant_grad_act_gc
                output = self.conv2d_quant_act(qinput_fw, qinput_bw, qweight, qbias, self.stride, self.padding, self.dilation, self.groups, error_bits, gc_bits)

            else:
                qinput = self.quantize_input_fw(input, num_bits)
                weight_qparams = calculate_qparams(self.weight, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
                qweight = quantize(self.weight, qparams=weight_qparams)
                output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
                output = quantize_grad(output, num_bits=num_grad_bits, flatten_dims=(1, -1))

            return output

        weight_qparams = calculate_qparams(self.weight, num_bits=self.weight_bits, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)

        qinput = self.quantize_input_fw(input, num_bits)
        output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
        output = quantize_grad(output, num_bits=num_grad_bits, flatten_dims=(1, -1))

        # if self.quant_act_forward == -1:
        #     qinput_fw = self.quantize_input_fw(input, num_bits)
        # else:
        #     qinput_fw = self.quantize_input_fw(input, self.quant_act_forward)

        # if self.quant_act_backward == -1:
        #     qinput_bw = self.quantize_input_bw(input, num_bits)
        # else:
        #     qinput_bw = self.quantize_input_bw(input, self.quant_act_backward)

        # if self.quant_grad_act_error == -1:
        #     error_bits = num_grad_bits
        # else:
        #     error_bits = self.quant_grad_act_error

        # if self.quant_grad_act_gc == -1:
        #     gc_bits = num_grad_bits
        # else:
        #     gc_bits = self.quant_grad_act_gc

        # output = self.conv2d_quant_act(qinput_fw, qinput_bw, qweight, qbias, self.stride, self.padding, self.dilation, self.groups, error_bits, gc_bits)

        return output


    def conv2d_quant_act(self, input_fw, input_bw, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, error_bits=0, gc_bits=0):
        out1 = F.conv2d(input_fw, weight.detach(), bias.detach() if bias is not None else None,
                        stride, padding, dilation, groups)
        out2 = F.conv2d(input_bw.detach(), weight, bias,
                        stride, padding, dilation, groups)
        out1 = quantize_grad(out1, num_bits=error_bits)
        out2 = quantize_grad(out2, num_bits=gc_bits)
        return out1 + out2 - out2.detach()


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=8, num_bits_grad=8, biprecision=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1), momentum=0.1)

    def forward(self, input, num_bits, num_bits_grad):
        # self.quantize_input = QuantMeasure(num_bits)
        qinput = self.quantize_input(input, num_bits)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        # if not self.biprecision or self.num_bits_grad is None:
        output = F.linear(qinput, qweight, qbias)
        #     if self.num_bits_grad is not None:
        output = quantize_grad(
            output, num_bits=num_bits_grad)
        # else:
        #     output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output


class RangeBN(nn.Module):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(inplace=True, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, num_bits, num_grad_bits):
        x = self.quantize_input(x, num_bits)
        if x.dim() == 2:  # 1d
            x = x.unsqueeze(-1,).unsqueeze(-1)

        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()  # C x B x H x W
            y = y.view(C, self.num_chunks, (B * H * W) // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)  # C
            mean_min = y.min(-1)[0].mean(-1)  # C
            mean = y.view(C, -1).mean(-1)  # C
            scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
                                        0.5) / ((2 * math.log(y.size(-1))) ** 0.5)

            scale = (mean_max - mean_min) * scale_fix
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(
                    mean * (1 - self.momentum))

                self.running_var.mul_(self.momentum).add_(
                    scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        # scale = quantize(scale, num_bits=self.num_bits, min_value=float(
        #     scale.min()), max_value=float(scale.max()))
        out = (x - mean.view(1, -1, 1, 1)) / \
            (scale.view(1, -1, 1, 1) + self.eps)

        if self.weight is not None:
            qweight = self.weight
            # qweight = quantize(self.weight, num_bits=self.num_bits,
            #                    min_value=float(self.weight.min()),
            #                    max_value=float(self.weight.max()))
            out = out * qweight.view(1, -1, 1, 1)

        if self.bias is not None:
            qbias = self.bias
            # qbias = quantize(self.bias, num_bits=self.num_bits)
            out = out + qbias.view(1, -1, 1, 1)
        if num_grad_bits:
            out = quantize_grad(
                out, num_bits=num_grad_bits, flatten_dims=(1, -1))

        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out


class RangeBN1d(RangeBN):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeBN1d, self).__init__(num_features, dim, momentum,
                                        affine, num_chunks, eps, num_bits, num_bits_grad)
        self.quantize_input = QuantMeasure(
            self.num_bits, inplace=True, shape_measure=(1, 1), flatten_dims=(1, -1))

if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)