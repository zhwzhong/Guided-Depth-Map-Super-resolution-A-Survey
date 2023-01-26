# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   rd.py
@Time    :   2021/8/27 09:34
@Desc    :
"""
import os
import math
import glob
import tqdm
from torch.utils.data.dataset import T_co

import utility
import random
import numpy as np
from scipy import io as sio
from torch.utils import data

random.seed(60)

root_path = '/data/zhwzhong/Data/Depth/RGB-D-D/process/'

if not os.path.exists(root_path):
    root_path = '/userhome/MyData/Depth/RGB-D-D/process/'

class RD(data.Dataset):
    def __init__(self, args, attr='train'):
        self.args = args
        self.attr = attr
        self.val_ratio = args.val_ratio
        self.train_data = {}
        img_list = sorted(glob.glob(root_path + '*.mat'))
        num_train = int((1 - args.val_ratio) * len(img_list))

        train_list = random.sample(img_list, num_train)

        val_list = [item for item in img_list if item not in train_list]

        assert len(train_list) + len(val_list) == len(img_list)

        self.img_list = train_list if attr == 'train' else val_list

    def __len__(self):
        return len(self.img_list) * self.args.show_every if self.attr == 'train' else len(self.img_list)

    def __getitem__(self, item):
        item = item % len(self.img_list)
        if self.attr == 'train':
            if self.img_list[item] in self.train_data.keys():
                img_mat = self.train_data[self.img_list[item]]
            else:
                img_mat = sio.loadmat(self.img_list[item])
                self.train_data[self.img_list[item]] = img_mat
        else:
            img_mat = sio.loadmat(self.img_list[item])
        lr_img, gt_img, rgb_img = img_mat['lr_up'], img_mat['gt_img'], img_mat['rgb_img']
        lr_img = np.expand_dims(lr_img, axis=0)
        gt_img = np.expand_dims(gt_img, axis=0)
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        if self.attr == 'train':
            lr_img, gt_img, rgb_img = utility.get_patch(lr_img, gt_img, rgb_img, patch_size=self.args.patch_size, scale=1)
            lr_img, gt_img, rgb_img = utility.augment(lr_img, gt_img, rgb_img, hflip=True, rot=True)


        lr_img, gt_img, rgb_img = utility.np_to_tensor(lr_img / 5732.0, gt_img / 5732.0, rgb_img / 255.0)

        img_name = os.path.basename(self.img_list[item]) if self.attr != 'train' else 'train'
        return {'lr_img': lr_img.float(), 'gt_img': gt_img.float(), 'rgb_img': rgb_img.float(), 'img_name': img_name}


# from option import args as arg
# data_loader = data.DataLoader(RD(arg, attr='test'))
#
# for _, samples in enumerate(data_loader):
#     print(samples['lr_img'].shape, samples['lr_img'].max(), samples['gt_img'].mean())