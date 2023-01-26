# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   __init__.py.py
@Time    :   2021/6/8 11:27
@Desc    :
"""
import os
import math
import glob
import data
import torch
import numpy as np
from importlib import import_module
from torch.utils import data as u_data
from prefetch_generator import BackgroundGenerator

class DataLoaderX(u_data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Data:
    def __init__(self, args, attr):
        self.loader_train = None
        self.data_loader = None
        self.load_npy = True
        self.attr = attr
        if self.attr == 'train':
            train_data = getattr(import_module('data.' + args.dataset_name.lower()), args.dataset_name)(args, attr='train')
            self.loader_train = DataLoaderX(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=not args.cpu, drop_last=True)
            print('===> Length of Train Data: {}, Number of Training Batch: {}'.format(len(train_data), len(self.loader_train)))

        else:
            data_set = getattr(import_module('data.' + args.dataset_name.lower()), args.dataset_name)(args, attr=attr)
            self.data_loader = DataLoaderX(dataset=data_set, batch_size=args.test_batch_size if attr == 'test' else 1,
                                                 shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

def get_dataloader(args, attr):
    return Data(args=args, attr=attr)