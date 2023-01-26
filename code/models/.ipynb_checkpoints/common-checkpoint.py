# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   common.py
@Time    :   2021/7/3 11:48
@Desc    :
"""
import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from models import switchable_norm as sn
from torch.nn.functional import softplus
from torch.nn.modules.utils import _pair
from torch.nn.functional import interpolate
from models.partialconv2d import PartialConv2d

class VConv2d(nn.modules.conv._ConvNd):
    """
    Versatile Filters
    Paper: https://papers.nips.cc/paper/7433-learning-versatile-filters-for-efficient-convolutional-neural-networks
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, delta=0, g=1, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(VConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.s_num = int(np.ceil(self.kernel_size[0] / 2))  # s in paper
        self.delta = delta  # c-\hat{c} in paper
        self.g = g  # g in paper
        self.weight = nn.Parameter(torch.Tensor(
            int(out_channels / self.s_num / (1 + self.delta / self.g)), in_channels // groups, *kernel_size))
        self.reset_parameters()

    def forward(self, x):
        x_list = []
        s_num = self.s_num
        ch_ratio = (1 + self.delta / self.g)
        ch_len = self.in_channels - self.delta
        for s in range(s_num):
            for start in range(0, self.delta + 1, self.g):
                weight1 = self.weight[:, :ch_len, s:self.kernel_size[0] - s, s:self.kernel_size[0] - s]
                if self.padding[0] - s < 0:
                    h = x.size(2)
                    x1 = x[:, start:start + ch_len, s:h - s, s:h - s]
                    padding1 = _pair(0)
                else:
                    x1 = x[:, start:start + ch_len, :, :]
                    padding1 = _pair(self.padding[0] - s)
                x_list.append(F.conv2d(x1, weight1, self.bias[int(
                    self.out_channels * (s * ch_ratio + start) / s_num / ch_ratio):int(
                    self.out_channels * (s * ch_ratio + start + 1) / s_num / ch_ratio)], self.stride,
                                       padding1, self.dilation, self.groups))
        x = torch.cat(x_list, 1)
        return x

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, inputs):
        inputs = inputs * (torch.tanh(softplus(inputs)))
        return inputs

def get_act(act):
    if act == 'PReLU':
        ret_act = torch.nn.PReLU()
    elif act == 'SELU':
        ret_act = torch.nn.SELU(True)
    elif act == 'LeakyReLU':
        ret_act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
    elif act == 'ELU':
        ret_act = torch.nn.ELU(inplace=True)
    elif act == 'ReLU':
        ret_act = torch.nn.ReLU(True)
    elif act == 'Mish':
        ret_act = Mish()
    else:
        print('ACT ERROR')
        ret_act = torch.nn.ReLU(True)
    return ret_act

class ConvBNReLU2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, partial=False, vcnn=False, act=None, norm=None):
        super(ConvBNReLU2D, self).__init__()
        if partial:
            self.layers = PartialConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif vcnn:
            self.layers = VConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.layers = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = None
        self.norm = None
        if norm == 'BN':
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm = torch.nn.InstanceNorm2d(out_channels)
        elif norm == 'GN':
            self.norm = torch.nn.GroupNorm(2, out_channels)
        elif norm == 'WN':
            self.layers = torch.nn.utils.weight_norm(self.layers)
        elif norm == 'SN':
            self.norm = sn.SwitchNorm2d(out_channels, using_moving_average=True, using_bn=True)
        elif norm == 'Adaptive':
            self.norm = AdaptiveNorm(n=out_channels)

        if act == 'PReLU':
            self.act = torch.nn.PReLU()
        elif act == 'SELU':
            self.act = torch.nn.SELU(True)
        elif act == 'LeakyReLU':
            self.act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        elif act == 'ELU':
            self.act = torch.nn.ELU(inplace=True)
        elif act == 'ReLU':
            self.act = torch.nn.ReLU(True)
        elif act == 'Tanh':
            self.act = torch.nn.Tanh()
        elif act == 'Mish':
            self.act = Mish()
        elif act == 'Sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'SoftMax':
            self.act = torch.nn.Softmax2d()

    def forward(self, *inputs):
        if len(inputs) == 1:
            out = self.layers(inputs[0])
        else:
            out = self.layers(inputs[0], inputs[1])

        if self.norm is not None:
            out = self.norm(out)

        if self.act is not None:
            out = self.act(out)
        return out


def torch_min(tensor):
    return torch.min(torch.min(tensor, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

def torch_max(tensor):
    return torch.max(torch.max(tensor, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

def torch_gaussian(channels, kernel_size=15, sigma=5):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp((-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)).float())

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

class DeConvReLU(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel=3, stride=2, padding=1):
        super(DeConvReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding=padding, output_padding=stride-1),
            nn.PReLU()
        )

    def forward(self, inputs):
        return self.layers(inputs)

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class invPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)

        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b, -1, y // ratio, x // ratio)


class ResNet(nn.Module):
    def __init__(self, num_features, act, norm):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(*[
            ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1, act=act, norm=norm),
            ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1, norm=norm)
        ])
        self.act = get_act(act=act)

    def forward(self, input_feature):
        return self.act(self.layers(input_feature) + input_feature)

class DownSample(nn.Module):
    def __init__(self, num_features, act, norm, scale=2):
        super(DownSample, self).__init__()
        if scale == 1:
            self.layers = nn.Sequential(
                ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, act=act, norm=norm, padding=1),
                ConvBNReLU2D(in_channels=num_features * scale * scale, out_channels=num_features, kernel_size=1, act=act, norm=norm)
            )
        else:
            self.layers = nn.Sequential(
                ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, act=act, norm=norm, padding=1),
                invPixelShuffle(ratio=scale),
                ConvBNReLU2D(in_channels=num_features * scale * scale, out_channels=num_features, kernel_size=1, act=act, norm=norm)
            )

    def forward(self, inputs):
        return self.layers(inputs)




class TUnet(nn.Module):
    def __init__(self, num_features, act, norm):
        super(TUnet, self).__init__()
        self.up_sample = nn.PixelShuffle(2)
        self.down_sample = invPixelShuffle(2)
        self.encoder_1 = ConvBNReLU2D(in_channels=num_features, out_channels=num_features // 2, kernel_size=3, padding=1, act=act, norm=norm)
        self.encoder_2 = ConvBNReLU2D(in_channels=num_features * 2, out_channels=num_features, kernel_size=3, padding=1, act=act, norm=norm)
        self.feature_transform = nn.Sequential(
            ConvBNReLU2D(in_channels=num_features * 4, out_channels=num_features * 4, kernel_size=3, padding=1, act=act, norm=norm),
            ConvBNReLU2D(in_channels=num_features * 4, out_channels=num_features * 4, kernel_size=3, padding=1, act=act, norm=norm)
        )

        self.decoder_1 = ConvBNReLU2D(in_channels=num_features * 4, out_channels=num_features * 4 * 2, kernel_size=3, padding=1, act=act, norm=norm)
        self.decoder_2 = ConvBNReLU2D(in_channels=num_features * 2, out_channels=num_features * 2 * 2, kernel_size=3, padding=1, act=act, norm=norm)

    def forward(self, inputs):
        enc_out_1 = self.down_sample(self.encoder_1(inputs))
        enc_out_2 = self.down_sample(self.encoder_2(enc_out_1))
        feature_out = self.feature_transform(enc_out_2) + enc_out_2
        dec_out_1 = self.up_sample(self.decoder_1(feature_out))
        dec_out_2 = self.up_sample(self.decoder_2(dec_out_1 + enc_out_1))
        return dec_out_2 + inputs

class Head(nn.Module):
    def __init__(self, num_features, act, expand_ratio=1, guide_channels=3, in_channels=1):
        super(Head, self).__init__()

        self.depth_head = nn.Sequential(
            ConvBNReLU2D(in_channels=in_channels, out_channels=num_features * expand_ratio, kernel_size=3, padding=1),
            ConvBNReLU2D(in_channels=num_features * expand_ratio, out_channels=num_features, kernel_size=1, act=act)
        )

        self.guide_head = nn.Sequential(
            ConvBNReLU2D(in_channels=guide_channels, out_channels=num_features * expand_ratio, kernel_size=3, padding=1),
            ConvBNReLU2D(in_channels=num_features * expand_ratio, out_channels=num_features, kernel_size=1, act=act)
        )

    def forward(self, depth_img, guide_img):
        return self.depth_head(depth_img), self.guide_head(guide_img)