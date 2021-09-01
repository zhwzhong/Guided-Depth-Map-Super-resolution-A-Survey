# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   dmsg.py
@Time    :   2021/3/22 21:05
@Desc    :
"""
from models.common import *

class DMSG(nn.Module):
    def __init__(self, args):
        super(DMSG, self).__init__()
        self.args = args
        self.scale = args.scale
        m = int(np.log2(self.scale))
        j = np.arange(2, 2 * m - 1, 2)
        j_ = np.arange(3, 2 * m, 2)

        M = 3 * (m + 1)
        k = np.arange(1, 3 * m, 3)
        k_1 = k + 1
        k_2 = k + 2
        k_3 = np.arange(3 * m + 1, M - 1, 3)

        self.branchY = nn.ModuleList()
        self.branchMain = nn.ModuleList()
        self.gaussian = torch_gaussian(channels=1, kernel_size=15, sigma=5)

        self.branchY.append(ConvBNReLU2D(1, 49, kernel_size=7, stride=1, padding=3, act='PReLU'))
        self.branchY.append(ConvBNReLU2D(49, 32, kernel_size=5, stride=1, padding=2, act='PReLU'))

        for i in range(2, 2 * m):
            if i in j_:
                self.branchY.append(nn.MaxPool2d(3, 2, padding=1))

            if i in j:
                self.branchY.append(ConvBNReLU2D(32, 32, kernel_size=5, stride=1, padding=2, act='PReLU'))

        self.feature_extra = ConvBNReLU2D(1, 64, kernel_size=5, stride=1, padding=2, act='PReLU')

        in_channels, out_channels = 64, 32
        for i in range(1, M):
            if i in k:
                self.branchMain.append(DeConvReLU(in_channels, out_channels, kernel=5, stride=2, padding=2))
            if i in k_1:
                self.branchMain.append(DeConvReLU(in_channels * 2, out_channels, kernel=5, stride=1, padding=2))
            if (i in k_2) or (i in k_3):
                self.branchMain.append(ConvBNReLU2D(in_channels, out_channels, 5, stride=1, padding=2, act='PReLU'))
            in_channels, out_channels = 32, 32

        self.branchMain.append(ConvBNReLU2D(32, 1, 5, stride=1, padding=2, act='PReLU'))

    def forward(self, lr, rgb, lr_up):
        rgb_img = torch.mean(rgb, dim=1, keepdim=True)
        h_Yh = rgb_img - self.gaussian(rgb_img)
        h_Yh = (h_Yh - torch_min(h_Yh)) / (torch_max(h_Yh) - torch_min(h_Yh))

        m = int(np.log2(self.scale))
        k = np.arange(0, 3 * m - 1, 3)

        outputs_Y = [h_Yh]

        for layer in self.branchY:
            outputs_Y.append(layer(outputs_Y[-1]))

        outputs_Main = [self.feature_extra(lr - self.gaussian(lr))]

        for i, layer in enumerate(self.branchMain):
            outputs_Main.append(layer(outputs_Main[-1]))
            if i in k:
                y_ind = 2 * (m - i // 3)
                outputs_Main.append(torch.cat((outputs_Y[y_ind], outputs_Main[-1]), dim=1))
        return [outputs_Main[-1] + self.gaussian(lr_up)]

def make_model(args): return DMSG(args)
