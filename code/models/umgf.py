# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   umgf.py
@Time    :   2022/7/27 20:55
@Desc    :
"""
import torch
from torch import nn
from models.common import ConvBNReLU2D
from torchvision.transforms.functional import rgb_to_grayscale

def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

class BasicBlock(torch.nn.Module):
    def __init__(self, num_features):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(
            ConvBNReLU2D(2, num_features, 3, dilation=1, act='ReLU', padding=1),
            ConvBNReLU2D(num_features, num_features, 3, dilation=2, act='ReLU', padding=2),
            ConvBNReLU2D(num_features, num_features, 3, dilation=4, act='ReLU', padding=4),
            ConvBNReLU2D(num_features, num_features, 3, dilation=8, act='ReLU', padding=8),
            ConvBNReLU2D(num_features, 1, 3, dilation=1, act='ReLU', padding=1)
        )
        self.compress = ConvBNReLU2D(3, 2, 1, dilation=1, act='ReLU', padding=0)

    def forward(self, inputs):

        out = self.layers(inputs)
        return out, self.compress(torch.cat((inputs, out), dim=1))

class UMGF(torch.nn.Module):
    def __init__(self):
        super(UMGF, self).__init__()
        self.dep_box = BoxFilter(r=4)
        self.rgb_box = BoxFilter(r=8)
        self.guide_conv = nn.Sequential(
            ConvBNReLU2D(3, 16, kernel_size=3, padding=1, act='ReLU'),
            ConvBNReLU2D(16, 1, kernel_size=3, padding=1, act=None),
        )
        self.amount_block = BasicBlock(num_features=24)

    def forward(self, samples):
        dep_img, rgb_img = samples['lr_up'], samples['img_rgb']
        img_box = self.dep_box(dep_img)
        rgb_res = rgb_img - self.rgb_box(rgb_img)
        dep_res = dep_img - img_box
        guide_conv2 = self.guide_conv(rgb_res)
        target_concat = torch.cat((dep_res, rgb_to_grayscale(rgb_res, 1)), dim=1)

        pred_depth_list = []
        for i in range(4):

            target_conv, target_concat = self.amount_block(target_concat)
            pred_depth = target_conv * guide_conv2 + img_box
            pred_depth_list.append(pred_depth)
        pred_depth_list = torch.cat(pred_depth_list, dim=1)

        return {'img_out': torch.mean(pred_depth_list, dim=1, keepdim=True)}

def make_model(args): return UMGF()