# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   quick_test.py
@Time    :   2021/8/31 20:32
@Desc    :
"""
import glob
import math
import tqdm
import numpy as np
import torch

import utility
from option import args
from data import get_dataloader
from importlib import import_module

args.scale = 8
args.down_type = 'nearest'

device = torch.device('cpu' if args.cpu else 'cuda')
module = import_module('models.' + args.model_name.lower())
model = module.make_model(args).to(device)

model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.num_gpus)))
print(utility.get_parameter_number(model))
device_id = torch.cuda.current_device()


load_name = './pre_trained/net_{}_x{}.pth'.format(args.down_type, args.scale)
print(load_name)
checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage.cuda(device_id))
model.load_state_dict(checkpoint['state'])
model.eval()

test_set = ['Middlebury', 'Lu', 'test', 'Sintel'] # , 'Middlebury', 'test', 'Sintel'
for test_name in test_set:
    sum_rmse = []
    for gt_name in tqdm.tqdm(sorted(glob.glob('./test_data/{}/gt/*.npy'.format(test_name)))):
        gt_img = np.load(gt_name)
        rgb_img = np.load(gt_name.replace('gt', 'rgb'))

        # Following DKN, we use bicubic in PIL to degrade GT image (for bicubic), and crop the border
        # before calculate the RMSE values. (reference: https://github.com/cvlab-yonsei/dkn/issues/1)
        module = max(int(math.pow(2, 1 + args.num_pyramid)), args.scale)
        tmp_gt = utility.mod_crop(gt_img, modulo=module)
        if args.down_type == 'nearest':
            tmp_gt = (tmp_gt - np.min(tmp_gt)) / (np.max(tmp_gt) - np.min(tmp_gt))
            lr_img = utility.get_lowers(tmp_gt, factor=args.scale, mode=args.down_direction)
        else:
            tmp_gt = (tmp_gt - np.min(tmp_gt)) / (np.max(tmp_gt) - np.min(tmp_gt))
            lr_img = utility.get_lowers(tmp_gt, factor=args.scale, mode='bicubic')
        lr_up = utility.get_lowers(lr_img, factor=1 / args.scale, mode='bicubic')
        lr_img, gt_img = np.expand_dims(lr_img, 0), np.expand_dims(gt_img, 0)
        lr_up = np.expand_dims(lr_up, 0)
        if args.guide_channels == 1:
            rgb_img = np.expand_dims(utility.rgb2gray(rgb_img), 2)

        rgb_img = np.float32(np.transpose(rgb_img, axes=(2, 0, 1))) / 255.

        gt_img, rgb_img = utility.mod_crop(gt_img, modulo=module), utility.mod_crop(rgb_img, modulo=module)

        lr_img, lr_up, gt_img, rgb_img = utility.np_to_tensor(lr_img, lr_up, gt_img, rgb_img)

        lr_img, lr_up, gt_img, rgb_img = lr_img.unsqueeze(0), lr_up.unsqueeze(0), gt_img.unsqueeze(0), rgb_img.unsqueeze(0)
        lr_img, lr_up, gt_img, rgb_img = lr_img.to(device), lr_up.to(device), gt_img.to(device), rgb_img.to(device)

        out = model(lr=lr_img.contiguous(), rgb=rgb_img.contiguous(), lr_up=lr_up.contiguous())[-1]


        if test_name == 'test':
            mul_ratio = 100
        elif test_name == 'Sintel':
            mul_ratio = 255
        else:
            mul_ratio = 1

        rmse, _ = utility.root_mean_sqrt_error(im_pred=out.contiguous(), im_true=gt_img.contiguous(), border=6, mul_ratio=mul_ratio, is_train=False)
        sum_rmse.append(rmse)

    print('{}: {:.2f}'.format(test_name, np.mean(sum_rmse)))
