# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   __init__.py.py
@Time    :   2021/6/10 16:05
@Desc    :
"""
import torch
from torch import nn
from loss.BALoss import BALoss

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.loss = []

        self.loss_module = nn.ModuleList()

        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_func = nn.MSELoss(reduction='mean')
            elif loss_type == 'L1':
                loss_func = nn.L1Loss(reduction='mean')
            elif loss_type == 'Huber':
                loss_func = nn.HuberLoss(reduction='mean', delta=args.hdelta)
            elif loss_type == 'Charbonnier':
                loss_func = Charbonnier()
            elif loss_type == 'BALoss':
                loss_func = BALoss()
            elif loss_type == 'SmoothL1':
                loss_func = nn.SmoothL1Loss(reduction='mean')
            else:
                raise NotImplementedError
            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_func})

        for l in self.loss:
            print('===> Loss Function: {:.3f} * {}'.format(l['weight'], l['type']))
            self.loss_module.append(l['function'])

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if not args.cpu and args.num_gpus > 1:
            self.loss_module = nn.DataParallel(self.loss_module, list(range(args.num_gpus)))

    def forward(self, out, gt):
        losses = []
        for i, l in enumerate(self.loss):
            loss = l['function'](out, gt)
            effective_loss = l['weight'] * loss
            losses.append(effective_loss)
        return sum(losses)

class Charbonnier(torch.nn.Module):
    def __init__(self):
        super(Charbonnier, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
