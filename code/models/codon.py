# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   codon.py
@Time    :   2022/7/27 17:25
@Desc    :
"""
import torch.nn
from torch import nn
from models.common import ConvBNReLU2D, pool_func
from torchvision.transforms.functional import rgb_to_grayscale

class MCUnit(torch.nn.Module):
    def __init__(self, num_features):
        super(MCUnit, self).__init__()
        self.conv_5x5 = ConvBNReLU2D(num_features, num_features, 5, padding=2, act='ReLU')
        self.conv_3x3 = ConvBNReLU2D(num_features, num_features, 3, padding=1, act='ReLU')

        self.cat = nn.Sequential(*[
            nn.ReLU(),
            ConvBNReLU2D(num_features * 2, num_features * 2, 5, padding=2, act=None),
            ConvBNReLU2D(num_features * 2, num_features, 1, padding=0, act=None)
        ])

    def forward(self, inputs):
        out = torch.cat((self.conv_3x3(inputs), self.conv_5x5(inputs)), dim=1)
        return self.cat(out)


class RMC(torch.nn.Module):
    def __init__(self, num_features, num_blocks):
        super(RMC, self).__init__()
        self.num_blocks = num_blocks
        self.layer = MCUnit(num_features)

    def forward(self, inputs):
        out = inputs
        for _ in range(self.num_blocks):
            out = self.layer(out) + inputs
        return out

class CAC(torch.nn.Module):
    def __init__(self, num_features):
        super(CAC, self).__init__()
        self.avg_layers = nn.Sequential(
            ConvBNReLU2D(num_features * 2, 8, 1, act='ReLU'),
            ConvBNReLU2D(8, num_features, 1, act=None)
        )
        self.max_layers = nn.Sequential(
            ConvBNReLU2D(num_features * 2, 8, 1, act='ReLU'),
            ConvBNReLU2D(8, num_features, 1, act=None)
        )
        self.conv_5x5 = ConvBNReLU2D(2, 1, 5, padding=2, act='ReLU')
        self.func = nn.Sigmoid()

    def forward(self, dep, rgb):
        out = torch.cat((dep, rgb), dim=1)
        channel_att = self.func(self.max_layers(pool_func(out, 'max')) + self.avg_layers(pool_func(out, 'avg')))
        spatial_att = torch.cat((torch.mean(out, dim=1, keepdim=True), torch.max(out, dim=1, keepdim=True)[0]), dim=1)
        spatial_att = self.func(self.conv_5x5(spatial_att))
        att = channel_att * spatial_att
        return att * dep, att * rgb

class Codon(torch.nn.Module):
    def __init__(self):
        super(Codon, self).__init__()
        num_features = 64
        self.input_d = nn.Sequential(
            ConvBNReLU2D(1, num_features, kernel_size=3, act='ReLU', padding=1),
            ConvBNReLU2D(num_features, num_features, kernel_size=3, act='ReLU', padding=1)
        )
        self.input_c = nn.Sequential(
            ConvBNReLU2D(1, num_features, kernel_size=3, act='ReLU', padding=1),
            ConvBNReLU2D(num_features, num_features, kernel_size=3, act='ReLU', padding=1)
        )

        self.rmc_d = MCUnit(num_features)
        self.rmc_c = MCUnit(num_features)

        self.cac = CAC(num_features)

        self.rec = nn.Sequential(
            ConvBNReLU2D(num_features * 2, num_features, kernel_size=1, act='ReLU'),
            RMC(num_features=num_features, num_blocks=3),
            ConvBNReLU2D(num_features, num_features, kernel_size=3, act='ReLU', padding=1),
            ConvBNReLU2D(num_features, 1, kernel_size=3, act='ReLU', padding=1),
        )

    def forward(self, samples):
        dep, rgb = samples['lr_up'], samples['img_rgb']
        rgb = rgb_to_grayscale(rgb, num_output_channels=1)

        dep, rgb = self.input_d(dep), self.input_c(rgb)

        _dep, _rgb = dep, rgb
        for i in range(5):

            _dep, _rgb = self.rmc_d(_dep), self.rmc_c(_rgb)
            _dep, _rgb = self.cac(_dep, _rgb)
            _dep, _rgb = dep + _dep, rgb + _rgb

        out = self.rec(torch.cat((_dep, _rgb), dim=1)) + samples['lr_up']
        return {'img_out': out}

def make_model(args): return Codon()
# from lib import utils
# sample = {
#     'lr_up': torch.randn(1, 1, 128, 128),
#     'img_rgb': torch.randn(1, 3, 128, 128)
# }
# net = Codon()
# out = net(sample)
# # print(out['img_out'].shape, out['img_out'].grad)
# out['img_out'].sum().backward()
# for para in net.parameters():
#     print(para.grad.sum())
# print(utils.get_parameter_number(net))

