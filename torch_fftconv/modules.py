import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from typing import Optional, List
from .functional import *


__all__ = [
    'FFTConv1d', 'FFTConv2d', 'FFTConv3d', 'FFTConvTranspose1d', 'FFTConvTranspose2d', 'FFTConvTranspose3d'
]


class FFTConv1d(nn.Conv1d):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return fft_conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              weight, bias, self.stride,
                              _single(0), self.dilation, self.groups)
        return fft_conv1d(input, weight, bias, self.stride,
                          self.padding, self.dilation, self.groups)


class FFTConv2d(nn.Conv2d):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return fft_conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              weight, bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        return fft_conv2d(input, weight, bias, self.stride,
                          self.padding, self.dilation, self.groups)


class FFTConv3d(nn.Conv3d):
    def forward(self, input: Tensor) -> Tensor:
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
