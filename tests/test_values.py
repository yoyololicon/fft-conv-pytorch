import torch
from torch.nn import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from torch_fftconv.modules import *

import pytest
from itertools import product

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@pytest.mark.parametrize('batch', [8])
@pytest.mark.parametrize('in_channels', [32])
@pytest.mark.parametrize('out_channels', [32])
@pytest.mark.parametrize('length', [60])
@pytest.mark.parametrize('kernel_size', [9, 11])
@pytest.mark.parametrize('stride', [1, 2, 3, 5])
@pytest.mark.parametrize('dilation', [1, 2, 3, 5])
@pytest.mark.parametrize('padding', [0, 4, 5, 10])
@pytest.mark.parametrize('bias', [True, False])
def test_conv1d_circular(batch, length,
                         in_channels, out_channels,
                         kernel_size, stride, padding, dilation, bias):

    x = torch.randn(batch, in_channels, length,
                    requires_grad=True, device=device)
    conv = Conv1d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, 1, bias, 'circular').to(device)
    fft_conv = FFTConv1d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, 1, bias, 'circular').to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()


@pytest.mark.parametrize('batch', [1, 4])
@pytest.mark.parametrize('in_channels', [16, 64])
@pytest.mark.parametrize('out_channels', [8, 32])
@pytest.mark.parametrize('length', [1733])
@pytest.mark.parametrize('kernel_size', [128, 256])
@pytest.mark.parametrize('stride', [1, 3, 4])
@pytest.mark.parametrize('dilation', [1, 2, 4])
@pytest.mark.parametrize('padding', [0, 3])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('groups', [1, 4])
@pytest.mark.parametrize('padding_mode', ['zeros', 'reflect'])
def test_conv1d(batch, length,
                in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias, padding_mode):

    x = torch.randn(batch, in_channels, length,
                    requires_grad=True, device=device)
    conv = Conv1d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv = FFTConv1d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('in_channels', [8, 32])
@pytest.mark.parametrize('out_channels', [4, 16])
@pytest.mark.parametrize('length', [(101, 101)])
@pytest.mark.parametrize('kernel_size', [17, 23])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('padding', [0, 7])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('padding_mode', ['zeros', 'circular'])
def test_conv2d(batch, length,
                in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias, padding_mode):

    x = torch.randn(batch, in_channels, *length,
                    requires_grad=True, device=device)
    conv = Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv = FFTConv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('in_channels', [8])
@pytest.mark.parametrize('out_channels', [8])
@pytest.mark.parametrize('length', [(53, 53, 59)])
@pytest.mark.parametrize('kernel_size', [9, 11])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('padding', [6])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('padding_mode', ['zeros', 'replicate'])
def test_conv3d(batch, length,
                in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias, padding_mode):

    x = torch.randn(batch, in_channels, *length,
                    requires_grad=True, device=device)
    conv = Conv3d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv = FFTConv3d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode).to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()


@pytest.mark.parametrize('batch', [1, 4])
@pytest.mark.parametrize('in_channels', [16, 64])
@pytest.mark.parametrize('out_channels', [8, 32])
@pytest.mark.parametrize('length', [409])
@pytest.mark.parametrize('kernel_size', [128, 256])
@pytest.mark.parametrize('stride,dilation,output_padding',
                         [x + (0,) for x in product([1, 3, 4], [1, 2, 4])] + [x + (1,) for x in product([3, 4], [2, 4])])
@pytest.mark.parametrize('padding', [0, 3])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('groups', [1, 4])
@pytest.mark.parametrize('padding_mode', ['zeros'])
def test_conv_transpose1d(batch, length,
                          in_channels, out_channels,
                          kernel_size, stride, padding, output_padding, dilation, groups, bias, padding_mode):

    x = torch.randn(batch, in_channels, length,
                    requires_grad=True, device=device)
    conv = ConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                           padding, output_padding, groups, bias, dilation, padding_mode).to(device)
    fft_conv = FFTConvTranspose1d(in_channels, out_channels, kernel_size,
                                  stride, padding, output_padding, groups, bias, dilation, padding_mode).to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('in_channels', [8, 32])
@pytest.mark.parametrize('out_channels', [4, 16])
@pytest.mark.parametrize('length', [(31, 31)])
@pytest.mark.parametrize('kernel_size', [17, 23])
@pytest.mark.parametrize('padding', [0, 7])
@pytest.mark.parametrize('stride,dilation,output_padding',
                         [x + (0,) for x in product([1, 2, 3], [1, 2, 3])] + [x + (1,) for x in product([2, 3], [2, 3])])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('padding_mode', ['zeros'])
def test_conv_transpose2d(batch, length,
                          in_channels, out_channels,
                          kernel_size, stride, padding, output_padding, dilation, groups, bias, padding_mode):

    x = torch.randn(batch, in_channels, *length,
                    requires_grad=True, device=device)
    conv = ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                           padding, output_padding, groups, bias, dilation, padding_mode).to(device)
    fft_conv = FFTConvTranspose2d(in_channels, out_channels, kernel_size,
                                  stride, padding, output_padding, groups, bias, dilation, padding_mode).to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('in_channels', [8])
@pytest.mark.parametrize('out_channels', [8])
@pytest.mark.parametrize('length', [(29, 23, 23)])
@pytest.mark.parametrize('kernel_size', [9, 11])
@pytest.mark.parametrize('padding', [6])
@pytest.mark.parametrize('stride,dilation,output_padding',
                         [x + (0,) for x in product([1, 2, 3], [1, 2, 3])] + [x + (1,) for x in product([2, 3], [2, 3])])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('padding_mode', ['zeros'])
def test_conv_transpose3d(batch, length,
                          in_channels, out_channels,
                          kernel_size, stride, padding, output_padding, dilation, groups, bias, padding_mode):

    x = torch.randn(batch, in_channels, *length,
                    requires_grad=True, device=device)
    conv = ConvTranspose3d(in_channels, out_channels, kernel_size, stride,
                           padding, output_padding, groups, bias, dilation, padding_mode).to(device)
    fft_conv = FFTConvTranspose3d(in_channels, out_channels, kernel_size,
                                  stride, padding, output_padding, groups, bias, dilation, padding_mode).to(device)
    fft_conv.load_state_dict(conv.state_dict())

    y1 = conv(x)
    y2 = fft_conv(x)
    assert torch.allclose(
        y1, y2, atol=1e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.sum().backward()
