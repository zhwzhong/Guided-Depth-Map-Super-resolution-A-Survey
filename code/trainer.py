# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   trainer.py
@Time    :   2021/6/26 09:34
@Desc    :
"""
import os
import tqdm
import time
import torch
import utility
import numpy as np
import torchnet as tnt
from data import get_dataloader
from prettytable import PrettyTable
from lr_scheduler import get_scheduler
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import MultiStepLR


class Trainer():
    def __init__(self, args, my_model, my_loss, writer):
        self.args = args
        self.loss = my_loss
        self.writer = writer
        self.model = my_model
        self.start_time = time.time()
        self.best_rmse = float('inf')
        self.epoch_num = self.step = 0
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.optimizer = utility.make_optimizer(self.args, self.model)
        if not args.test_only:
            self.loader_train = get_dataloader(args=self.args, attr='train').loader_train
        # self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.learning_rate_decay, gamma=0.5)
        self.scheduler = get_scheduler(self.optimizer, n_iter_per_epoch=len(self.loader_train), args=args)
        if self.args.re_load:
            self.load()

    def train(self):
        self.model.train()
        log_file = open('./logfile/{}.log'.format(self.args.file_name), 'w')
        log_file.close()
        train_loss = tnt.meter.AverageValueMeter()
        train_rmse = tnt.meter.AverageValueMeter()
        for epoch_num in range(self.epoch_num, self.args.num_epochs):
            self.epoch_num = epoch_num

            show_lr = utility.print_learning_rate(self.optimizer)
            if os.path.exists('/userhome/MyData'):
                p_bar = self.loader_train
            else:
                p_bar = tqdm.tqdm(self.loader_train)

            for _, sample in enumerate(p_bar):
                self.step += 1
                self.optimizer.zero_grad()
                lr_img, gt_img, rgb_img = self.prepare(sample['lr_img'], sample['gt_img'], sample['rgb_img'])
                if self.args.dataset_name == 'RD':
                    lr_up = lr_img
                elif self.args.dataset_name == 'NYU' and self.args.pre_up:
                    lr_up = self.prepare(sample['lr_up'])[0]
                else:
                    lr_up = interpolate(lr_img, scale_factor=self.args.scale, mode='bicubic', align_corners=False)

                out_img = self.model(lr=lr_img, rgb=rgb_img, lr_up=lr_up)
                loss = self.loss(out_img[-1], gt_img)

                if self.args.pyramid_loss:
                    loss1 = 0
                    for num_j in range(len(out_img) - 1):

                        if self.args.pyramid_dir == 'Up':
                            loss1 += self.loss(out_img[num_j], gt_img) / (len(out_img) - 1)
                        else:
                            if self.args.pyramid_way == 'nearest':
                                inter_gt = interpolate(gt_img, size=out_img[num_j].size()[2:], mode=self.args.pyramid_way)
                            else:
                                inter_gt = interpolate(gt_img, size=out_img[num_j].size()[2:], mode=self.args.pyramid_way, align_corners=False)

                            loss1 += self.loss(out_img[num_j], inter_gt) / (len(out_img) - 1)
                    if self.args.change_weight:
                        loss1 = loss1 * (self.args.num_epochs - self.epoch_num) / self.args.num_epochs
                    loss = loss + loss1
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                rmse, _ = utility.root_mean_sqrt_error(im_pred=out_img[-1], im_true=gt_img, border=0, is_train=True)
                train_rmse.add(rmse)
                train_loss.add(loss.item())
                if not os.path.exists('/userhome/MyData'):
                    p_bar.set_description('===> Epoch: {}'.format(str(self.epoch_num)).zfill(3))
                    p_bar.set_postfix(LR=show_lr, RMSE=rmse)

            self.writer.add_scalar('loss', train_loss.value()[0], self.epoch_num)
            self.writer.add_scalar('rmse/train', train_rmse.value()[0], self.epoch_num)
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.step)
            print('===> Epoch: {}, Step: {:<5d}, {:<5s}_loss: {:.4f}, {:<5s}_rmse: {:.4f}, time_spend: {}, LR: {}'
                  .format(self.epoch_num, self.step, 'train', 10000 * train_loss.value()[0], 'train',
                          train_rmse.value()[0], utility.time_since(self.start_time), utility.get_lr(self.optimizer)))

            train_rmse.reset()
            train_loss.reset()
            self.val()
            if self.args.dataset_name != 'RD':   self.test()

    def test_model(self, attr, border, mul_ratio, is_train):
        self.model.eval()
        test_loader = get_dataloader(self.args, attr).data_loader

        sum_times = 0
        rmse_list = []
        name_list = []

        test_rmse = tnt.meter.AverageValueMeter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _, sample in enumerate(test_loader):
            lr_img, gt_img, rgb_img = self.prepare(sample['lr_img'], sample['gt_img'], sample['rgb_img'])
            if self.args.dataset_name == 'RD':
                lr_up = lr_img
            elif self.args.dataset_name == 'NYU' and self.args.pre_up:
                lr_up = self.prepare(sample['lr_up'])[0]
            else:
                lr_up = interpolate(lr_img, scale_factor=self.args.scale, mode='bicubic', align_corners=False)
            start.record()
            out = self.model(lr=lr_img, rgb=rgb_img, lr_up=lr_up)[-1]
            end.record()
            torch.cuda.synchronize()
            sum_times += start.elapsed_time(end)
            rmse, im_pred = utility.root_mean_sqrt_error(im_pred=out, im_true=gt_img, border=border, mul_ratio=mul_ratio,
                                                         is_train=is_train)

            if len(self.args.save_path) > 2 and attr not in ['val']:
                save_file = './{}/{}/{}'.format(self.args.save_path, attr, sample['img_name'][0])
                np.save(file=save_file, arr=im_pred.squeeze().detach().cpu().numpy())
                print('===> Image Saved to {}...'.format(save_file))

            rmse_list.append(rmse)
            name_list.append(sample['img_name'][0])
            test_rmse.add(rmse)
        return test_rmse.value()[0], round(sum_times / 1000, 5), name_list, rmse_list

    def val(self):
        with torch.no_grad():
            rmse, time_cost, _, _= self.test_model('val', border=0, mul_ratio=100, is_train=True)

        if self.best_rmse > rmse:
            self.best_rmse = rmse
            self.save(self.epoch_num, last_name='best')
        else:
            self.save(self.epoch_num, last_name='final')
        print('===> Val Average RMSE: {}'.format(round(rmse, 4)))
        with open('./logfile/{}.log'.format(self.args.file_name), 'a') as f:
            f.write('===> Val Average RMSE: {}\n'.format(round(rmse, 4)))
        self.writer.add_scalar('rmse/val', round(rmse, 4), self.epoch_num)

    def test(self):
        print("===> Testing model...")
        test_data_name = []
        test_data_rmse = []
        if self.args.test_only:  self.load()
        with torch.no_grad():
            test_set = self.args.test_set.split('+')
            for test_name in test_set:
                if len(self.args.save_path) > 2:
                    utility.create_dir('./{}/{}'.format(self.args.save_path, test_name))
                mul_ratio = 1
                if test_name == 'test' and self.args.dataset_name == 'NYU':
                    mul_ratio = 100
                if test_name == 'Sintel' and self.args.dataset_name == 'NYU':
                    mul_ratio = 255

                test_rmse, test_time, name_list, rmse_list = self.test_model(test_name, 6, mul_ratio, is_train=False)
                test_data_name.append(test_name)
                test_data_rmse.append(round(test_rmse, 4))
                self.writer.add_scalar('rmse/{}'.format(test_name), round(test_rmse, 4), self.epoch_num)

        with open('./logfile/{}.log'.format(self.args.file_name), 'a') as f:
            f.write('===> Test Average RMSE: {}\n'.format(round(test_rmse, 4)))
        table = PrettyTable(test_data_name)
        table.add_row(test_data_rmse)
        print(table)

    def prepare(self, *args):
        def _prepare(tensor):
            return tensor.to(self.device).contiguous()

        return [_prepare(a) for a in args]

    def save(self, epoch_num, last_name=''):
        print('===> Saving {} models...'.format(last_name))
        state = {
            'state': self.model.state_dict(),
            'epoch': epoch_num
        }
        if last_name == 'best':
            torch.save(state, './checkpoints/{}/net_best.pth'.format(self.args.file_name))
        else:
            torch.save(state, './checkpoints/{}/net_{}.pth'.format(self.args.file_name, str(epoch_num)))

    def load(self):
        print('===> Loading from checkpoints...')
        device_id = torch.cuda.current_device()
        checkpoint_file = os.path.join('./checkpoints/{}'.format(self.args.file_name))
        if os.path.exists(checkpoint_file):
            file_name = os.listdir(checkpoint_file)
            if self.args.load_best:
                load_name = './checkpoints/{}/net_{}.pth'.format(self.args.file_name, 'best')
                if os.path.exists(load_name):
                    checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage.cuda(device_id))
                    self.epoch_num = checkpoint['epoch']
                    print('===> Load best checkpoint data, Epoch: {}'.format(checkpoint['epoch']))
                    self.model.load_state_dict(checkpoint['state'])
                else:
                    print('No Best Model {}'.format(load_name))
            else:
                max_num = utility.get_max_epoch(file_name)
                if os.path.exists('./checkpoints/{}/net_{}.pth'.format(self.args.file_name, str(max_num))):
                    checkpoint = torch.load('./checkpoints/{}/net_{}.pth'.format(self.args.file_name, str(max_num)),
                                            map_location=lambda storage, loc: storage.cuda(device_id))
                    self.epoch_num = checkpoint['epoch']
                    print('===> Load last checkpoint data, Epoch: {}'.format(checkpoint['epoch']))
                    self.model.load_state_dict(checkpoint['state'])
                else:
                    print('No Max model')
        else:
            print('No Model file ...')

