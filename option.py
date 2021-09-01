# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   option.py
@Time    :   2021/7/3 10:32
@Desc    :
"""
import os
import argparse
import get_gpu_info
from set_template import set_template
parser = argparse.ArgumentParser(description='Depth Image Super-Resolution')

# HardWare
parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--cuda_name', type=int, default=0)

# Data
parser.add_argument('--pre_up', action='store_true')
parser.add_argument('--data_max', type=float, default=5732)
parser.add_argument('--data_mean', type=float, default=0.3457)
parser.add_argument('--val_ratio', type=float, default=0.1)
parser.add_argument('--train_ratio', type=float, default=1) # different training sets
parser.add_argument('--data_range', type=float, default=1)
parser.add_argument('--data_augment', type=bool, default=True)
parser.add_argument('--dataset_name', type=str, default='RD') # Flash NYU CAVE Flash NIR MPI
parser.add_argument('--down_type', type=str, default='bicubic') # nearest bic
parser.add_argument('--down_direction', type=str, default='last')  # center  last
parser.add_argument('--test_set', type=str, default='test+Lu+Middlebury')
parser.add_argument('--num_res', type=int, default=2)
parser.add_argument('--with_noisy', action='store_true')
parser.add_argument('--noisy_level', type=float, default=25)


# Learning Rate
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--lr_scheduler', type=str, default='step') # "step", "cosine"
parser.add_argument('--lr_decay_rate', type=float, default=0.5)

parser.add_argument('--patience', type=int, default=7)
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--warmup_multiplier', type=int, default=100)
parser.add_argument('--lr_decay_epochs', type=str, default='100')


parser.add_argument('--show_every', type=int, default=32)

parser.add_argument('--scale', type=int, default=4)

parser.add_argument('--num_epochs', type=int, default=121)
parser.add_argument('--num_features', type=int, default=32)

parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--guide_channels', type=int, default=3)

parser.add_argument('--act', type=str, default='PReLU')
parser.add_argument('--norm', type=str, default='None')
parser.add_argument('--batch_size', type=int, default=32)  # DKN 为8  BFTRes  16
parser.add_argument('--patch_size', type=int, default=256)  # HR 的大小  256
parser.add_argument('--test_batch_size', type=int, default=1)

parser.add_argument('--loss', type=str, default='1*L1')
parser.add_argument('--hdelta', type=float, default=1)
parser.add_argument('--change_weight', action='store_true')
parser.add_argument('--num_pyramid', type=int, default=3)
parser.add_argument('--pyramid_loss', action='store_true')
parser.add_argument('--filter_size', type=int, default=3)  # 生成的kernel的大小

parser.add_argument('--re_load', action='store_true')
parser.add_argument('--load_best', action='store_true')
parser.add_argument('--model_name', type=str, default='DAGF')  # SVLRM FDKN DKN BestNet PacLite PacJointUpsample DJF DJFR DGN PMPAN GF


parser.add_argument('--file_name', type=str, default='')
parser.add_argument('--save_path', type=str, default='')

args = parser.parse_args()
# args = parser.parse_args([])

set_template(args)

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
    elif vars(args)[arg] == 'None':
        vars(args)[arg] = None


args.lr_decay_epochs = [int(num) for num in args.lr_decay_epochs.split('_')]
if get_gpu_info.get_memory(num_gpu=args.num_gpus) is False:
    print('Out of the memory')
    while True:
        i = 999 * 9132877

else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(get_gpu_info.get_memory(num_gpu=args.num_gpus))

if len(args.file_name) == 0:
    args.file_name = args.model_name + '_' + args.dataset_name + '_' + str(args.scale) + '_' + args.down_type.upper() + \
                     '_' + args.loss

    if args.pyramid_loss:
        args.file_name += '_PY'

    if args.with_noisy:
        args.file_name += '_Noisy'

    if args.dataset_name == 'RD':
        args.file_name += '_Mean_' + str(args.data_mean) if args.data_mean > 0 else ''

print(args)
print('===> Save File Name: ', args.file_name)