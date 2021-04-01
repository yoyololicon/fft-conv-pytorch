import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from .functional import *


__all__ = [
    'FFTConv1d', 'FFTConv2d', 'FFTConv3d',
]


class FFTConv1d(nn.Conv1d):
    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != 'zeros':
            return fft_conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              self.weight, self.bias, self.stride, _single(0),
                              self.dilation, self.groups)
        return fft_conv1d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)


class FFTConv2d(nn.Conv2d):
    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != 'zeros':
            return fft_conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              self.weight, self.bias, self.stride, _pair(0),
                              self.dilation, self.groups)
        return fft_conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)


class FFTConv3d(nn.Conv3d):
    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != 'zeros':
            return fft_conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                              self.weight, self.bias, self.stride, _triple(0),
                              self.dilation, self.groups)
        return fft_conv3d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
