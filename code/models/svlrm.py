# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   svlrm.py
@Time    :   2022/7/26 11:56
@Desc    :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
import math


class SVLRM(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_layer = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        feature_block = []
        for _ in range(1, 11):
            feature_block += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                              nn.LeakyReLU(0.1)
                              ]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.data.normal_(0, math.sqrt(2.0 / n))
                nn.init.constant_(m.bias, 0)

    def forward(self, samples):
        lr_data, guided_data = samples['lr_up'], samples['img_rgb']
        guided_data = rgb_to_grayscale(guided_data, 1)
        input_tensor = torch.cat((lr_data, guided_data), dim=1)
        param = F.leaky_relu(self.first_layer(input_tensor), 0.1)
        param = self.feature_block(param)
        param = self.final_layer(param)

        param_alpha, param_beta = param[:, :1, :, :], param[:, 1:, :, :]
        output = param_alpha * guided_data + param_beta

        return {'img_out': output}
        # return output, param_alpha, param_beta


def make_model(args): return SVLRM()