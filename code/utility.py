# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   utility.py
@Time    :   2021/6/8 20:56
@Desc    :
"""
import os
import re
import math
import time
import torch
import shutil
import random
import numpy as np
from PIL import Image
import torch.optim as optim
from collections import Iterable
from skimage.transform import resize


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def mod_crop(img, modulo):
    if len(img.shape) == 2:
        h, w = img.shape
        return img[: h - (h % modulo), :w - (w % modulo)]
    else:
        _, h, w = img.shape
        return img[:, : h - (h % modulo), :w - (w % modulo)]


def get_patch(*args, patch_size=32, scale=4):
    """

    :param args: (LR, HR, ..)
    :param patch_size: LR Patch Size
    :param scale: HR // LR
    :return: (LR, HR, ..)
    """
    ih, iw = args[0].shape[1:]
    tp = scale * patch_size
    ip = tp // scale

    iy = random.randrange(0, ih - ip + 1)
    ix = random.randrange(0, iw - ip + 1)
    tx, ty = scale * ix, scale * iy
    ret = [
        args[0][:, iy:iy + ip, ix:ix + ip],
        *[a[:, ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]

    return ret


def augment(*args, hflip=True, rot=True):
    """
    Input: (C, H, W)
    :param args:
    :param hflip:
    :param rot:
    :return:
    """
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, :, ::-1]
        if vflip: img = img[:, ::-1, :]
        if rot90: img = img.transpose(0, 2, 1)
        return np.ascontiguousarray(img)

    return [_augment(a) for a in args]

def get_lowers(im_np, factor, mode='last'):
    """
    mode: 'bicubic', 'bilinear', 'nearest', 'last', 'center'
    """
    if im_np.ndim == 3:
        im_np = im_np.transpose(1, 2, 0)

    h0, w0 = im_np.shape[:2]
    h, w = int(math.ceil(h0 / float(factor))), int(math.ceil(w0 / float(factor)))

    if h0 != h * factor or w0 != w * factor:
        im_np = resize(im_np, (h * factor, w * factor), order=1, mode='reflect', clip=False, preserve_range=True,
                       anti_aliasing=True)

    if mode in ('last', 'center'):
        if mode == 'last':
            idxs = (slice(factor - 1, None, factor),) * 2
        else:
            assert mode == 'center'
            idxs = (slice(int((factor - 1) // 2), None, factor),) * 2
        lowers = im_np[idxs].copy()
    else:
        if len(im_np.shape) == 3:
            im_np = im_np[:, :, 0]
            lowers = np.expand_dims(np.array(Image.fromarray(im_np).resize((w, h), Image.BICUBIC)), 2)
        else:
            lowers = np.array(Image.fromarray(im_np).resize((w, h),Image.BICUBIC))

    if lowers.ndim == 3:
        lowers = lowers.transpose((2, 0, 1))

    return lowers


def np_to_tensor(*args, input_data_range=1.0, process_data_range=1.0):
    def _np_to_tensor(img):
        np_transpose = img.astype(np.float32)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(process_data_range / input_data_range)
        return tensor.float()

    return [_np_to_tensor(a) for a in args]


def root_mean_sqrt_error(im_pred, im_true, border=6, mul_ratio=100, is_train=False):
    b, c, h, w = im_true.size()
    if not is_train:
        im_pred, im_true = im_pred.view(im_pred.size(0), -1), im_true.view(im_true.size(0), -1)
        img_min, _ = torch.min(im_true, dim=1, keepdim=True)
        img_max, _ = torch.max(im_true, dim=1, keepdim=True)
        im_pred = im_pred * (img_max - img_min) + img_min
    if border != 0:
        im_pred = im_pred.view(b, c, h, w)[:, :, border: -border, border: -border]
        im_true = im_true.view(b, c, h, w)[:, :, border: -border, border: -border]

    return round(torch.sqrt(torch.mean(((im_true * mul_ratio) - (im_pred * mul_ratio)) ** 2)).item(), 5), im_pred

def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)

def set_checkpoint_dir(args):
    if args.test_only is False and args.re_load is False:
        #print('Removing Previous Checkpoints and Get New Checkpoints Dir')
        create_dir('./logs/{}'.format(args.file_name))
        create_dir('./logfile/{}'.format(args.file_name))
        create_dir('./checkpoints/{}'.format(args.file_name))


def init_state():
    np.random.seed(60)
    torch.manual_seed(60)
    torch.cuda.manual_seed(60)
    torch.cuda.manual_seed_all(60)
    torch.backends.cudnn.deterministic = True

def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_num, trainable_num = clever_format([total_num, trainable_num])
    return {'Total': total_num, 'Trainable': trainable_num}


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def make_optimizer(args, targets):

    if args.optimizer == 'AMSGrad':
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    else:
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer

def print_learning_rate(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr

def get_max_epoch(list_name):
    max_number = 0
    for name in list_name:
        if name.find("best") < 0:
            tmp = int(re.findall(r"\d+", os.path.basename(name))[0])
            if max_number < tmp:
                max_number = tmp
    return max_number

def get_lr(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr