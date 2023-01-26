# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   nyu.py
@Time    :   2020/1/13 20:43
@Desc    :
"""
import os
import math
import glob
import utility
import random
import numpy as np
from torch.utils import data

random.seed(60)
root_path = '/home/zhwzhong/Data/'

if not os.path.exists(root_path):
    root_path = '/userhome/MyData/'

data_dict = {
    'Lu': root_path + 'Depth/NYU/npy/Lu/gt/',
    'SOD': root_path + 'Depth/NYU/npy/SOD/gt/',
    'test': root_path + 'Depth/NYU/npy/nyu/test/gt/',
    'train': root_path + 'Depth/NYU/npy/nyu/train/gt/',
    'Sintel': root_path + 'Depth/NYU/npy/Sintel/gt/',
    'Middlebury': root_path + 'Depth/NYU/npy/Middlebury/gt/',
    'A': root_path + 'Depth/MPI/npy/A/gt/', 'B': root_path + 'Depth/MPI/npy/B/gt/', 'C': root_path + 'Depth/MPI/npy/C/gt/'
}

class NYU(data.Dataset):
    def __init__(self, args, attr='train'):
        self.args = args
        self.attr = attr
        self.val_ratio = args.val_ratio

        self.gt_imgs = []
        self.rgb_imgs = []

        if self.attr == 'val':
            self.img_list = sorted(glob.glob(data_dict['train'] + '*.npy'))
        else:
            self.img_list = sorted(glob.glob(data_dict['{}'.format(attr)] + '*.npy'))

        if self.attr == 'train':
            self.img_list = self.img_list[: int((1 - self.val_ratio) * len(self.img_list))]

            if self.args.train_ratio < 1:
                train_number  = int(self.args.train_ratio * len(self.img_list))
                self.img_list = random.sample(self.img_list, train_number)

            for img_name in self.img_list:
                self.gt_imgs.append(np.expand_dims(np.load(img_name), 0))
                tmp_rgb_img = np.load(img_name.replace('gt', 'rgb'))

                if self.args.guide_channels == 1:
                    tmp_rgb_img = np.expand_dims(utility.rgb2gray(tmp_rgb_img), 2)

                self.rgb_imgs.append(np.float32(np.transpose(tmp_rgb_img, (2, 0, 1))) / 255.)
                # self.rgb_imgs.append(np.transpose( np.load(img_name.replace('gt', 'rgb')), axes=(2, 0, 1)) / 255.)

        elif self.attr == 'val':
            self.img_list = self.img_list[int((1 - self.val_ratio) * len(self.img_list)):]

    def __len__(self):
        if self.attr == 'train':
            return len(self.img_list) * self.args.show_every
        else:
            return len(self.img_list)

    def __getitem__(self, item):
        item = item % len(self.img_list)
        rgb_name = 'test'
        if self.attr == 'train':
            gt_img, rgb_img = utility.mod_crop(self.gt_imgs[item], modulo=int(math.pow(2, 1 + self.args.num_pyramid))), \
                              utility.mod_crop(self.rgb_imgs[item], modulo=int(math.pow(2, 1 + self.args.num_pyramid)))

            gt_img, rgb_img = utility.get_patch(gt_img, rgb_img, patch_size=self.args.patch_size, scale=1)

            if self.args.data_augment:
                gt_img, rgb_img = utility.augment(gt_img, rgb_img)
            if self.args.down_type == 'nearest':
                lr_img = utility.get_lowers(gt_img, factor=self.args.scale, mode=self.args.down_direction)
            else:
                lr_img = utility.get_lowers(gt_img, factor=self.args.scale, mode='bicubic')
            lr_up = utility.get_lowers(lr_img, factor=1 / self.args.scale, mode='bicubic')
        else:
            gt_name = self.img_list[item]
            rgb_name = gt_name.replace('gt', 'rgb')

            gt_img, rgb_img = np.load(gt_name), np.load(rgb_name)

            module = max(int(math.pow(2, 1 + self.args.num_pyramid)), self.args.scale)
            tmp_gt = utility.mod_crop(gt_img, modulo=module)
            if self.args.down_type == 'nearest':

                tmp_gt = (tmp_gt - np.min(tmp_gt)) / (np.max(tmp_gt) - np.min(tmp_gt))
                lr_img = utility.get_lowers(tmp_gt, factor=self.args.scale, mode=self.args.down_direction)
            else:
                tmp_gt = (tmp_gt - np.min(tmp_gt)) / (np.max(tmp_gt) - np.min(tmp_gt))
                lr_img = utility.get_lowers(tmp_gt, factor=self.args.scale, mode='bicubic')
            lr_up = utility.get_lowers(lr_img, factor=1 / self.args.scale, mode='bicubic')
            lr_img, gt_img = np.expand_dims(lr_img, 0), np.expand_dims(gt_img, 0)
            lr_up = np.expand_dims(lr_up, 0)
            if self.args.guide_channels == 1:
                rgb_img = np.expand_dims(utility.rgb2gray(rgb_img), 2)

            rgb_img = np.float32(np.transpose(rgb_img, axes=(2, 0, 1))) / 255.

            gt_img, rgb_img = utility.mod_crop(gt_img, modulo=module), utility.mod_crop(rgb_img, modulo=module)

            # print(gt_img.shape, rgb_img.shape, 'ppp')
            rgb_name = os.path.basename(rgb_name)

        if self.args.with_noisy:
            lr_img = np.random.normal(0, 5 / 255, lr_img.shape) + lr_img
            lr_img = np.clip(lr_img, 0, 1)

        lr_img, lr_up, gt_img, rgb_img = utility.np_to_tensor(lr_img, lr_up, gt_img, rgb_img)

        # assert lr_img.max() < 1.1, 'LR'
        # assert gt_img.max() < 1.1, 'GT'
        assert rgb_img.max() < 1.1, 'Guide'
        sample = {'lr_img': lr_img, 'lr_up': lr_up, 'gt_img': gt_img, 'rgb_img': rgb_img, 'img_name': rgb_name}
        return sample

# from option import args
# args.test_set = 'test'
# args.down_type = 'nearest'
# args.with_noisy = 'False'
# args.val_ratio = 0.1
# args.scale = 16
# from torch.nn.functional import interpolate
# data_loader = data.DataLoader(NYU(args, attr='train'))
# print(len(data_loader))
# sum_num = 0
# sum_rmse = []
# for _, samples in enumerate(data_loader):
#     print(samples['lr_up'].shape, samples['gt_img'].shape)
#     # rmse = utility.root_mean_sqrt_error(im_pred=samples['lr_up'], im_true=samples['gt_img'], border=0, mul_ratio=100)[0]
#     rmse = utility.root_mean_sqrt_error(im_pred=interpolate(samples['lr_img'], scale_factor=args.scale, mode='bicubic',
#                                         align_corners=False), im_true=samples['gt_img'] , border=0, mul_ratio=100)[0]
#
#     sum_rmse.append(rmse)
#
# print(np.mean(sum_rmse), len(sum_rmse), sum_num)