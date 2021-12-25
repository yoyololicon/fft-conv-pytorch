import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.fft import rfftn, rfft, fftn, ifftn, irfft
from torch.nn.modules.utils import _single, _pair, _triple
from typing import Optional, List, Tuple
from .functional import *
from .functional import _complex_matmul, _conv_shape


__all__ = [
    'FFTConv1d', 'FFTConv2d', 'FFTConv3d', 'FFTConvTranspose1d', 'FFTConvTranspose2d', 'FFTConvTranspose3d'
]


def _check_circular_optimize(input_shape: tuple, kernel_shape: tuple, stride: tuple, dilation: tuple) -> bool:
    for i in range(len(input_shape)):
        if input_shape[i] % stride[i] or input_shape[i] % dilation[i] or kernel_shape[i] * dilation[i] > input_shape[i]:
            return False
    return (input_shape[-1] // stride[-1]) % 2 == 0 and (input_shape[-1] // dilation[-1]) % 2 == 0


def _circular_fft_convnd(input: Tensor,
                         weight: Tensor,
                         bias: Optional[Tensor],
                         stride: Tuple[int],
                         dilation: Tuple[int],
                         padding: Tuple[int],
                         groups: int) -> Tensor:

    input = input.roll(shifts=padding, dims=tuple(range(2, input.dim())))
    output_size = _conv_shape(
        input.shape[2:], weight.shape[2:], stride, padding, dilation)

    weight_s = tuple(l // d for l, d in zip(input.shape[2:], dilation))

    X = rfftn(input, dim=tuple(range(2, input.dim())))
    W = rfft(weight, n=weight_s[-1])

    if dilation[-1] > 1:
        W_neg_freq = W.flip(-1)[..., 1:]
        W_neg_freq.imag.mul_(-1)

        tmp = [W]
        for i in range(1, dilation[-1]):
            if i % 2:
                tmp.append(W_neg_freq)
            else:
                tmp.append(W[..., 1:])

        W = torch.cat(tmp, -1)

    if len(weight_s) > 1:
        W = fftn(W, s=weight_s[:-1], dim=tuple(range(2, W.ndim - 1)))
        repeats = (1, 1) + dilation[:-1] + (1,)
        W.imag.mul_(-1)
        if sum(repeats) > W.ndim:
            W = W.repeat(*repeats)
    else:
        W.imag.mul_(-1)

    Y = _complex_matmul(X, W, groups)

    # handle stride
    if len(stride) > 1:
        unfold_shape = [Y.size(0), Y.size(1)]
        sum_dims: List[int] = []
        for i, st in enumerate(stride[:-1]):
            if st == 1:
                unfold_shape.append(Y.size(i + 2))
                continue
            step = Y.size(i + 2) // st
            unfold_shape += [st, step]
            sum_dims.append(len(unfold_shape) - 2)

        unfold_shape.append(-1)
        if len(sum_dims):
            Y = Y.view(*unfold_shape).mean(sum_dims)
        Y = ifftn(Y, dim=tuple(range(2, Y.ndim - 1)))

    if stride[-1] > 1:
        n_fft = Y.size(-1) * 2 - 2
        new_n_fft = n_fft // stride[-1]
        step_size = new_n_fft // 2
        strided_Y_size = step_size + 1

        unfolded_Y_real = Y.real.unfold(-1, strided_Y_size, step_size)
        unfolded_Y_imag = Y.imag[...,
                                 1:].unfold(-1, strided_Y_size - 2, step_size)
        Y_pos_real, Y_pos_imag = unfolded_Y_real[..., ::2,
                                                 :].sum(-2), unfolded_Y_imag[..., ::2, :].sum(-2)
        Y_neg_real, Y_neg_imag = unfolded_Y_real[..., 1::2, :].sum(
            -2).flip(-1), unfolded_Y_imag[..., 1::2, :].sum(-2).flip(-1)

        Y_real = Y_pos_real.add_(Y_neg_real)
        Y_imag = Y_pos_imag.add_(Y_neg_imag, alpha=-1)
        Y_imag = F.pad(Y_imag, [1, 1])

        Y = torch.view_as_complex(
            torch.stack((Y_real, Y_imag), -1)).div_(stride[-1])

    output = irfft(Y)
    idx = (slice(None), slice(None))
    for i in range(2, output.dim()):
        if output.shape[i] < output_size[i - 2]:
            output = torch.cat(
                [output, output[idx + (slice(output_size[i - 2] - output.shape[i]),)]], i)
        elif output.shape[i] > output_size[i - 2]:
            output = output[idx + (slice(0, output_size[i - 2]),)]
        idx = (slice(None),) + idx

    if bias is not None:
        output += bias[(slice(None),) + (None,) * (output.ndim - 2)]

    return output


class FFTConv1d(nn.Conv1d):
    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode == 'circular' and _check_circular_optimize(input.shape[2:], self.weight.shape[2:], self.stride, self.dilation):
            return _circular_fft_convnd(input, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups)

        if self.padding_mode != 'zeros':
            return fft_conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              self.weight, self.bias, self.stride, _single(0),
                              self.dilation, self.groups)
        return fft_conv1d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)


class FFTConv2d(nn.Conv2d):
    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode == 'circular' and _check_circular_optimize(input.shape[2:], self.weight.shape[2:], self.stride, self.dilation):
            return _circular_fft_convnd(input, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups)

        if self.padding_mode != 'zeros':
            return fft_conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              self.weight, self.bias, self.stride, _pair(0),
                              self.dilation, self.groups)
        return fft_conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)


class FFTConv3d(nn.Conv3d):
    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode == 'circular' and _check_circular_optimize(input.shape[2:], self.weight.shape[2:], self.stride, self.dilation):
            return _circular_fft_convnd(input, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups)

        if self.padding_mode != 'zeros':
            return fft_conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              self.weight, self.bias, self.stride, _triple(0),
                              self.dilation, self.groups)
        return fft_conv3d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)


class FFTConvTranspose1d(nn.ConvTranspose1d):
    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError(
                'Only `zeros` padding mode is supported for FFTConvTranspose1d')

        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore
        return fft_conv_transpose1d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class FFTConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError(
                'Only `zeros` padding mode is supported for FFTConvTranspose2d')

        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore
        return fft_conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class FFTConvTranspose3d(nn.ConvTranspose3d):
    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError(
                'Only `zeros` padding mode is supported for FFTConvTranspose3d')

        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore
        return fft_conv_transpose3d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
