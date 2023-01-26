# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   dgf.py
@Time    :   2021/7/3 20:14
@Desc    :
"""
import torch
from torch import nn
from models.guiuded_filter import ConvGuidedFilter

class DGF(nn.Module):
    def __init__(self, args):
        super(DGF, self).__init__()
        self.args = args
        self.layers = ConvGuidedFilter(in_channels=1, num_features=64)

    def forward(self, lr, rgb, lr_up):
        rgb = torch.mean(rgb, dim=1, keepdim=True)

        out = self.layers(rgb, lr_up)
        return [out]

def make_model(args): return DGF(args)