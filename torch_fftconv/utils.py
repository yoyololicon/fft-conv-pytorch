import torch
from torch import nn
from .modules import *


def convert_fft_conv(module: nn.Module):
    module_output = module
    if isinstance(module, nn.Conv1d):
        module_output = FFTConv1d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                  module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode)
    elif isinstance(module, nn.Conv2d):
        module_output = FFTConv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                  module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode)
    elif isinstance(module, nn.Conv3d):
        module_output = FFTConv3d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                  module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode)
    elif isinstance(module, nn.ConvTranspose1d):
        module_output = FFTConvTranspose1d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                           module.padding, module.output_padding, module.groups, module.bias is not None, module.dilation, module.padding_mode)
    elif isinstance(module, nn.ConvTranspose2d):
        module_output = FFTConvTranspose2d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                           module.padding, module.output_padding, module.groups, module.bias is not None, module.dilation, module.padding_mode)
    elif isinstance(module, nn.ConvTranspose3d):
        module_output = FFTConvTranspose3d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                           module.padding, module.output_padding, module.groups, module.bias is not None, module.dilation, module.padding_mode)
    if not (module_output is module):
        module_output = module_output.to(module.weight.device)
        module_output.load_state_dict(module.state_dict())

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_fft_conv(child))
    del module
    return module_output
