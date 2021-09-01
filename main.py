# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   main.py
@Time    :   2021/6/26 09:34
@Desc    :
"""
import warnings
warnings.filterwarnings("ignore")

import loss
import torch
import utility
from option import args
from trainer import Trainer
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter


utility.init_state()
utility.set_checkpoint_dir(args)

writer = SummaryWriter('./logs/{}'.format(args.file_name))

loss = loss.Loss(args=args)

device = torch.device('cpu' if args.cpu else 'cuda')
module = import_module('models.' + args.model_name.lower())
model = module.make_model(args).to(device)
if not args.cpu:
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.num_gpus)))

print('===> Parameter Number:', utility.get_parameter_number(model))

train_process = Trainer(args=args, my_model=model, my_loss=loss, writer=writer)

if args.test_only:
    train_process.test()
else:
    train_process.train()