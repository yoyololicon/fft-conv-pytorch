import torch
from torch import Tensor
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
from torch.fft import rfft, fftn, rfftn, ifft, irfft, ifftn, irfftn

from typing import List, Optional, Union, Tuple
from math import gcd

__all__ = [
    'fft_conv1d', 'fft_conv2d', 'fft_conv3d', 'fft_conv_transpose1d', 'fft_conv_transpose2d', 'fft_conv_transpose3d'
]


def _reverse_repeat_tuple(t: List[int], n: int) -> List[int]:
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    out: List[int] = []
    for i in range(len(t) - 1, -1, -1):
        x = t[i]
        out += [x] * n
    return out


def _lcm(x: int, y: int):
    return abs(x * y) // gcd(x, y)


def _complex_matmul(a: Tensor, b: Tensor, groups: int = 1, transpose: bool = False) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view((a.size(0), groups, -1) + a.shape[2:] + (1,))
    b = b.view((groups, -1) + b.shape[1:])
    a = torch.movedim(a, 2, a.dim() - 1)

    if transpose:
        b = torch.movedim(b, (1, 2), (b.dim() - 2, b.dim() - 1))
    else:
        b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    c = a @ b
    c = torch.movedim(c, c.dim() - 1, 2)
    return c.reshape((c.size(0), -1) + c.shape[3:-1])


def _conv_shape(L_in: List[int],
                kernel: List[int],
                stride: List[int],
                padding: List[int],
                dilation: List[int]) -> List[int]:

    L_out: List[int] = []
    for i in range(len(L_in)):
        out = (L_in[i] - 1) * stride[i] - 2 * padding[i] + \
            dilation[i] * (kernel[i] - 1) + 1
        # out = (l + 2 * p - d * (k - 1) - 1) // s + 1
        assert out > 0, "Kernel size can't be greater than input."
        L_out.append(out)

    return L_out


def _conv_transpose_shape(L_in: Tuple[int],
                          kernel: Tuple[int],
                          stride: Tuple[int],
                          padding: Tuple[int],
                          output_padding: Tuple[int],
                          dilation: Tuple[int]) -> Tuple[int]:

    L_out: List[int] = []
    for l, k, s, p, op, d in zip(L_in, kernel, stride, padding, output_padding, dilation):
        out = (l - 1) * s - 2 * p + d * (k - 1) + op + 1
        assert out > 0, "Kernel size can't be greater than input."
        L_out.append(out)

    return tuple(L_out)


def _fft_convnd(input: Tensor,
                weight: Tensor,
                bias: Optional[Tensor],
                stride: List[int],
                padding: List[int],
                dilation: List[int],
                groups: int) -> Tensor:

    output_size = _conv_shape(input.shape[2:], weight.shape[2:],
                              stride, padding, dilation)
    reversed_padding_repeated_twice = _reverse_repeat_tuple(padding, 2)
    padded_input = F.pad(input, reversed_padding_repeated_twice)

    s: List[int] = []
    weight_s: List[int] = []
    for i in range(len(dilation)):
        x_size = padded_input.shape[2 + i]
        w_size = weight.shape[2 + i]
        d = dilation[i]
        st = stride[i]
        s_size = max(x_size, w_size * d)

        # find s size that can be divided by stride and dilation
        rfft_even = 2 if i == len(stride) - 1 else 1
        factor = _lcm(st * rfft_even, d * rfft_even)

        offset = s_size % factor
        if offset:
            s_size += factor - offset
        s.append(s_size)
        weight_s.append(s_size // d)

    X = rfftn(padded_input, s=s)

    W = rfft(weight, n=weight_s[-1])
    # handle dilation
    # handle dilation for last dim
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
        W = fftn(W, s=weight_s[:-1], dim=list(range(2, W.ndim - 1)))
        repeats = (1, 1) + dilation[:-1] + (1,)
        W.imag.mul_(-1)
        if sum(repeats) > W.ndim:
            W = W.repeat(repeats)
    else:
        W.imag.mul_(-1)

    Y = _complex_matmul(X, W, groups)

    # handle stride
    if len(stride) > 1:
        for i, st in enumerate(stride[:-1]):
            if st > 1:
                tmp = Y.shape[:i+2] + (st, -1) + Y.shape[i+3:]
                Y = Y.reshape(tmp).mean(i+2)

            Y = ifft(Y, dim=i+2)
            Y_stride: List[int] = []
            for j in range(len(Y.shape)):
                Y_stride.append(Y.stride(j))
            Y = Y.as_strided(
                Y.shape[:i+2] + output_size[i:i+1] + Y.shape[i+3:], Y_stride)

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

    # Remove extra padded values
    output = output[..., :output_size[-1]].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        output += bias[:, None]

    return output


def _fft_conv_transposend(input: Tensor,
                          weight: Tensor,
                          bias: Optional[Tensor],
                          stride: Tuple[int],
                          padding: Tuple[int],
                          output_padding: Tuple[int],
                          groups: int,
                          dilation: Tuple[int],) -> Tensor:

    output_size = _conv_transpose_shape(input.shape[2:], weight.shape[2:],
                                        stride, padding, output_padding, dilation)
    padded_output_size = tuple(o + 2 * p for o, p in zip(output_size, padding))

    s: List[int] = []
    weight_s: List[int] = []
    for i, (x_size, w_size, d, st) in enumerate(zip(padded_output_size, weight.shape[2:], dilation, stride)):
        s_size = max(x_size, w_size * d)

        # find s size that can be divided by stride and dilation
        rfft_even = 2 if i == len(stride) - 1 else 1
        factor = _lcm(st * rfft_even, d * rfft_even)

        offset = s_size % factor
        if offset:
            s_size += factor - offset
        s.append(s_size // st)
        weight_s.append(s_size // d)

    X = rfft(input, n=s[-1])
    W = rfft(weight, n=weight_s[-1])

    if stride[-1] > 1:
        X_neg_freq = X.flip(-1)[..., 1:]
        X_neg_freq.imag.mul_(-1)

        tmp = [X]
        for i in range(1, stride[-1]):
            if i % 2:
                tmp.append(X_neg_freq)
            else:
                tmp.append(X[..., 1:])

        X = torch.cat(tmp, -1)

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

    if len(s) > 1:
        X = fftn(X, s=s[:-1], dim=tuple(range(2, X.ndim - 1)))
        W = fftn(W, s=weight_s[:-1], dim=tuple(range(2, W.ndim - 1)))
        repeats = (1, 1) + stride[:-1] + (1,)
        if sum(repeats) > X.ndim:
            X = X.repeat(*repeats)

        repeats = (1, 1) + dilation[:-1] + (1,)
        if sum(repeats) > W.ndim:
            W = W.repeat(*repeats)

    Y = _complex_matmul(X, W, groups, True)

    output = irfftn(Y, dim=tuple(range(2, Y.ndim)))

    # Remove extra padded values
    index = (slice(None),) * 2 + tuple(slice(p, o + p)
                                       for p, o in zip(padding, output_size))
    output = output[index].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        output += bias[(slice(None),) + (None,) * (output.ndim - 2)]

    return output


def fft_conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
               stride: Union[int, Tuple[int]] = 1,
               padding: Union[int, Tuple[int]] = 0,
               dilation: Union[int, Tuple[int]] = 1,
               groups: int = 1) -> Tensor:
    r"""
    """
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)

    return _fft_convnd(input, weight, bias, stride, padding, dilation, groups)


def fft_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
               stride: Union[int, Tuple[int]] = 1,
               padding: Union[int, Tuple[int]] = 0,
               dilation: Union[int, Tuple[int]] = 1,
               groups: int = 1) -> Tensor:
    r"""
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    return _fft_convnd(input, weight, bias, stride, padding, dilation, groups)


def fft_conv3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
               stride: Union[int, Tuple[int]] = 1,
               padding: Union[int, Tuple[int]] = 0,
               dilation: Union[int, Tuple[int]] = 1,
               groups: int = 1) -> Tensor:
    r"""
    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    return _fft_convnd(input, weight, bias, stride, padding, dilation, groups)


def fft_conv_transpose1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                         stride: Union[int, Tuple[int]] = 1,
                         padding: Union[int, Tuple[int]] = 0,
                         output_padding: Union[int, Tuple[int]] = 0,
                         groups: int = 1,
                         dilation: Union[int, Tuple[int]] = 1) -> Tensor:
    r"""
    """
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)
    output_padding = _single(output_padding)

    return _fft_conv_transposend(input, weight, bias, stride, padding, output_padding, groups, dilation)


def fft_conv_transpose2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                         stride: Union[int, Tuple[int]] = 1,
                         padding: Union[int, Tuple[int]] = 0,
                         output_padding: Union[int, Tuple[int]] = 0,
                         groups: int = 1,
                         dilation: Union[int, Tuple[int]] = 1) -> Tensor:
    r"""
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    output_padding = _pair(output_padding)

    return _fft_conv_transposend(input, weight, bias, stride, padding, output_padding, groups, dilation)


def fft_conv_transpose3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                         stride: Union[int, Tuple[int]] = 1,
                         padding: Union[int, Tuple[int]] = 0,
                         output_padding: Union[int, Tuple[int]] = 0,
                         groups: int = 1,
                         dilation: Union[int, Tuple[int]] = 1) -> Tensor:
    r"""
    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    output_padding = _triple(output_padding)

    return _fft_conv_transposend(input, weight, bias, stride, padding, output_padding, groups, dilation)
