# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   BALoss.py
@Time    :   2021/7/3 11:19
@Desc    :
"""
import torch
from torch.nn import functional as F


class BALoss(torch.nn.Module):
    def __init__(self):
        super(BALoss, self).__init__()

    def forward(self, output, target):
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        b, c, w, h = output.shape
        sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
        sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
        sobel_x = sobel_x.type_as(output)
        sobel_y = sobel_y.type_as(output)
        weight_x = torch.nn.Parameter(data=sobel_x, requires_grad=False)
        weight_y = torch.nn.Parameter(data=sobel_y, requires_grad=False)
        Ix1 = F.conv2d(output, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(target, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(output, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(target, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        #     loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = dx * dy * torch.abs(target - output)
        return torch.mean(loss)