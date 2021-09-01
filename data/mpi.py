# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   nyu.py
@Time    :   2021/6/8 11:27
@Desc    :
"""
import os
import cv2
import math
import glob
import torch
import utility
import numpy as np
from torch.utils import data

root_path = '/data/zhwzhong/Data/'

if os.path.exists('/data/zhwzhong/Data/'):
    root_path = '/data/zhwzhong/Data/'
elif os.path.exists('/data/zhwzhong/Data/'):
    root_path = '/data/zhwzhong/Data/'
elif os.path.exists('/userhome/MyData/'):
    root_path = '/userhome/MyData/'

data_dict = {
    'train': root_path + 'Depth/MPI/npy/train/gt/',
    'test': root_path + 'Depth/MPI/npy/test/gt/',
    'Lu': root_path + 'Depth/NYU/npy/Lu/gt/',  # Lu
    'NYU': root_path + 'Depth/NYU/npy/nyu/test/gt/',  # NYU
    '2003': root_path + 'Depth/MPI/npy/2003/gt/'
}


def mod_crop(img, modulo):
    h, w = img.shape[: 2]
    return img[: h - (h % modulo), :w - (w % modulo), :]


def img_resize(gt_img, rgb_img, scale):
    rh, rw = rgb_img.shape[: 2]
    dh, dw = gt_img.shape[: 2]
    if rh != dh:
        crop_h = (rh - dh) // 2
        crop_w = (rw - dw) // 2
        rgb_img = rgb_img[crop_h: rh - crop_h, crop_w: rw - crop_w, :]

    gt_img, rgb_img = mod_crop(gt_img, modulo=scale), mod_crop(rgb_img, scale)
    return gt_img, rgb_img


def min_max_norm(img):
    img_min, img_max = np.min(img), np.max(img)
    return (img - img_min) / (img_max - img_min)


class MPI(data.Dataset):
    def __init__(self, args, attr='train'):
        self.args = args
        self.attr = attr

        self.gt_imgs = []
        self.lr_imgs = []
        self.rgb_imgs = []

        self.img_list = sorted(glob.glob(data_dict['train'] + '*.npy'))
        val_num = int(len(self.img_list) * self.args.val_ratio)

        if self.attr == 'val':
            self.img_list = self.img_list[len(self.img_list) - val_num:]
        elif self.attr == ' train':
            self.img_list = self.img_list[: len(self.img_list) - val_num]
        else:
            self.img_list = sorted(glob.glob(data_dict['{}'.format(attr)] + '*.npy'))

        for img_name in self.img_list:
            gt_img = np.load(img_name)
            rgb_img = np.load(img_name.replace('gt', 'rgb'))
            # print(gt_img.shape, rgb_img.shape)
            gt_img = np.expand_dims(gt_img, 2)  # ==> (H, W, C)
            gt_img, rgb_img = img_resize(gt_img=gt_img, rgb_img=rgb_img,
                                         scale=int(math.pow(2, 1 + self.args.num_pyramid)))
            if args.guide_channels == 1:
                rgb_img = np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), 2)

            gt_norm, rgb_img = min_max_norm(gt_img), min_max_norm(rgb_img)
            gt_img = gt_img if (attr in ['test', 'NYU', 'Lu', '2003']) else gt_norm

            lr_img = cv2.resize(gt_norm, None, fx=1 / self.args.scale, fy=1 / self.args.scale,
                                interpolation=cv2.INTER_CUBIC)

            if args.with_noisy:
                lr_img = np.random.normal(0, 5 / 255, lr_img.shape) + lr_img
                lr_img = np.clip(lr_img, 0, 1)

            lr_img, rgb_img, gt_img = np.expand_dims(lr_img, 0), np.transpose(rgb_img, (2, 0, 1)), np.transpose(gt_img,
                                                                                                                (2, 0,
                                                                                                                 1))

            if self.attr == 'val':
                _, h, w = gt_img.shape

                patch_size = (1024 if h > 1024 and w > 1024 else min(h, w)) // self.args.scale
                if self.args.model_name == 'PMPAN':
                    patch_size = patch_size - (patch_size % (2 * self.args.scale))

                if patch_size > 0:

                    _, gt_patch = utility.img_to_patch_sr(lr_img, gt_img, patch_size=patch_size, scale=self.args.scale,
                                                          stride=patch_size // 2)

                    lr_patch, rgb_patch = utility.img_to_patch_sr(lr_img, rgb_img, patch_size=patch_size,
                                                                  scale=self.args.scale, stride=patch_size // 2)

                    for num_i in range(lr_patch.shape[0]):
                        self.lr_imgs.append(lr_patch[num_i])
                        self.gt_imgs.append(gt_patch[num_i])
                        self.rgb_imgs.append(rgb_patch[num_i])
            else:
                self.lr_imgs.append(lr_img)
                self.gt_imgs.append(gt_img)
                self.rgb_imgs.append(rgb_img)

        self.sample_length = len(self.lr_imgs) * self.args.show_every

        if self.attr != 'train':
            self.sample_length = len(self.lr_imgs)

    def __len__(self):
        return self.sample_length

    def __getitem__(self, item):
        item = item % self.sample_length
        lr_img, gt_img, rgb_img = self.lr_imgs[item], self.gt_imgs[item], self.rgb_imgs[item]
        if self.attr == 'train':
            lr_img, gt_img, rgb_img = utility.get_patch(lr_img, gt_img, rgb_img,
                                                        patch_size=self.args.patch_size // self.args.scale,
                                                        scale=self.args.scale)
            if self.args.data_augment:
                lr_img, gt_img, rgb_img = utility.augment(lr_img, gt_img, rgb_img)
        if self.attr not in ['train', 'val']:
            rgb_name = os.path.basename(self.img_list[item])
        else:
            rgb_name = self.attr

        mask = np.where(gt_img <= 0, gt_img, 1)

        lr_img, gt_img, rgb_img, mask = utility.np_to_tensor(lr_img, gt_img, rgb_img, mask)
        assert lr_img.max() < 1.5, 'LR_{}'.format(lr_img.max())
        # assert gt_img.max() < 1.1, 'GT'
        assert rgb_img.max() < 1.1, 'Guide_{}'.format(rgb_img.max())
        # lr_img = torch.FloatTensor(lr_img.size()).normal_(mean=0, std=5 * (1 / 255.))
        sample = {'lr_img': lr_img, 'gt_img': gt_img, 'rgb_img': rgb_img, 'mask': mask, 'img_name': rgb_name}
        return sample
