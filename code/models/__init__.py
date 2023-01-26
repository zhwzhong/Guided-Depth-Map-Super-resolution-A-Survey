# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   __init__.py.py
@Time    :   2021/6/10 16:06
@Desc    :
"""
import torch
import torch.nn as nn
from importlib import import_module

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making Model...')
        self.args = args
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        module = import_module('models.' + args.model_name.lower())
        self.model = module.make_model(args).to(self.device)
        self.model = torch.nn.parallel.DataParallel(self.model, device_ids=list(range(self.args.num_gpus)))

    def forward(self, lr, rgb):
        self.model.forward(lr, rgb)