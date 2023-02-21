# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   ahmf.py
@Time    :   2020/1/2 17:06
@Desc    :
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.common import ConvBNReLU2D, invPixelShuffle, InvUpSampler, get_act, UpSampler


class FeatureInitialization(nn.Module):
    # 提取Depth和RGB的特征，变为64个通道
    def __init__(self, num_features, scale, guidance_channel=1):
        super(FeatureInitialization, self).__init__()

        self.rgb_shuffle = invPixelShuffle(ratio=scale)

        self.depth_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_features, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.guidance_in = nn.Sequential(
            nn.Conv2d(in_channels=guidance_channel, out_channels=num_features, kernel_size=3, padding=1),
            nn.PReLU(),
            InvUpSampler(scale=scale, n_feats=num_features)
        )

    def forward(self, depth, guidance):
        # guide_shuffle = self.rgb_shuffle(guidance)
        return self.depth_in(depth), self.guidance_in(guidance), None


class Compress(nn.Module):
    def __init__(self, num_features, act, norm, fuse_way='add'):
        super(Compress, self).__init__()
        self.fuse_way = fuse_way

        self.layers = ResNet(num_features=num_features, act=act, norm=norm)

        if self.fuse_way == 'cat':
            self.compress_out = ConvBNReLU2D(in_channels=2 * num_features, out_channels=num_features, kernel_size=1,
                                             padding=0, act=act)

    def forward(self, *inputs):
        if len(inputs) == 2:
            if self.fuse_way == 'add':
                out = inputs[0] + inputs[1]
            else:
                out = self.compress_out(torch.cat(([inputs[0], inputs[1]]), dim=1))
        else:
            out = inputs[0]
        return self.layers(out)


class ResNet(nn.Module):
    def __init__(self, num_features, act, norm):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(*[
            ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1,
                         act=act, norm=norm),
            ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1,
                         norm=norm)
        ])
        self.act = get_act(act=act)

    def forward(self, input_feature):
        return self.act(self.layers(input_feature) + input_feature)


def variance_pool(x):
    my_mean = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    return (x - my_mean).pow(2).mean(dim=3, keepdim=False).mean(dim=2, keepdim=False).view(x.size()[0], x.size()[1], 1,
                                                                                           1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


def pool_func(x, pool_type=None):
    b, c = x.size()[:2]
    if pool_type == 'avg':
        ret = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    elif pool_type == 'max':
        ret = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    elif pool_type == 'lp':
        ret = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
    else:
        ret = variance_pool(x)
    return ret.view(b, c)


class GateConv2D(nn.Module):
    def __init__(self, num_features):
        super(GateConv2D, self).__init__()
        self.Attention = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.Feature = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1),
            nn.PReLU()
        )

    def forward(self, inputs):
        return self.Attention(inputs) * self.Feature(inputs)


class ConvGRUCell(nn.Module):
    """
    Basic CGRU cell.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, bias):
        super(ConvGRUCell, self).__init__()

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.update_gate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=self.hidden_dim,
                                     kernel_size=self.kernel_size, padding=self.padding,
                                     bias=self.bias)
        self.reset_gate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=self.hidden_dim,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    bias=self.bias)

        self.out_gate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=self.hidden_dim,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur = cur_state
        # data size is [batch, channel, height, width]
        x_in = torch.cat([input_tensor, h_cur], dim=1)
        update = torch.sigmoid(self.update_gate(x_in))
        reset = torch.sigmoid(self.reset_gate(x_in))
        x_out = torch.tanh(self.out_gate(torch.cat([input_tensor, h_cur * reset], dim=1)))
        h_new = h_cur * (1 - update) + x_out * update

        return h_new

    def init_hidden(self, b, h, w):
        return torch.zeros(b, self.hidden_dim, h, w).cuda()


class ConvGRU(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers=2,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvGRUCell(in_channels=cur_input_dim,
                                         hidden_channels=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvGRU
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class MMAB(nn.Module):
    def __init__(self, num_features, reduction_ratio=4):
        super(MMAB, self).__init__()

        self.squeeze = ConvBNReLU2D(in_channels=num_features * 2, out_channels=num_features * 2 // reduction_ratio,
                                    kernel_size=3, act='PReLU', padding=1)

        self.excitation1 = ConvBNReLU2D(in_channels=num_features * 2 // reduction_ratio, out_channels=num_features,
                                        kernel_size=1, act='Sigmoid')
        self.excitation2 = ConvBNReLU2D(in_channels=num_features * 2 // reduction_ratio, out_channels=num_features,
                                        kernel_size=1, act='Sigmoid')

    def forward(self, depth, guidance):
        fuse_feature = self.squeeze(torch.cat((depth, guidance), 1))
        fuse_statistic = pool_func(fuse_feature, 'avg') + pool_func(fuse_feature)
        squeeze_feature = fuse_statistic.unsqueeze(2).unsqueeze(3)
        depth_out = self.excitation1(squeeze_feature)
        guidance_out = self.excitation2(squeeze_feature)
        return (depth_out * depth).div(2), (guidance_out * guidance).div(2)


class FuseNet(nn.Module):
    def __init__(self, num_features, reduction_ratio, act, norm):
        super(FuseNet, self).__init__()

        self.filter_conv = GateConv2D(num_features=num_features)
        self.filter_conv1 = GateConv2D(num_features=num_features)
        self.attention_layer = MMAB(num_features=num_features, reduction_ratio=reduction_ratio)
        self.res_conv = ResNet(num_features=num_features, act=act, norm=norm)

    def forward(self, depth, guide):
        guide = self.filter_conv(guide)
        depth = self.filter_conv1(depth)
        depth, guide = self.attention_layer(depth=depth, guidance=guide)

        fuse_feature = self.res_conv(depth + guide)

        return fuse_feature


class AHMF(nn.Module):
    def __init__(self, scale=4, act='PReLU'):
        super(AHMF, self).__init__()

        self.head = FeatureInitialization(num_features=64, scale=scale, guidance_channel=3)

        # Forward Backward None ALL
        self.act = act
        self.scale = scale
        self.rgb_conv = nn.ModuleList()
        self.fuse_conv = nn.ModuleList()
        self.depth_conv = nn.ModuleList()
        self.compress_out = nn.ModuleList()

        self.forward_gru_cell = nn.ModuleList()
        self.reverse_gru_cell = nn.ModuleList()

        for _ in range(3):
            self.rgb_conv.append(
                ConvBNReLU2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, act=self.act)
            )

        for _ in range(3):
            self.depth_conv.append(
                ConvBNReLU2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, act=self.act)
            )

        for _ in range(4):
            self.fuse_conv.append(
                FuseNet(num_features=64, reduction_ratio=4, act=self.act, norm=None)
            )

            self.compress_out.append(
                Compress(num_features=64, act=self.act, norm=None)
            )

        self.forward_gru_cell = ConvGRU(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)

        self.reverse_gru_cell = ConvGRU(in_channels=64, hidden_channels=64, kernel_size=(3, 3), batch_first=True)

        self.up_conv = nn.Sequential(
            ConvBNReLU2D(in_channels=64 * 4, out_channels=64,
                         kernel_size=1, padding=0, act=self.act),
            *UpSampler(scale=scale, n_feats=64),
            ConvBNReLU2D(in_channels=64, out_channels=1, kernel_size=3, padding=1, norm=None)
        )



    def forward(self, samples):
        lr, rgb, lr_up = samples['img_lr'], samples['img_rgb'], samples['lr_up']
        depth_feature, guide_feature, _ = self.head(lr, rgb)

        depth_out = [depth_feature]
        guide_out = [guide_feature]

        for i in range(3):
            guide_feature = self.rgb_conv[i](guide_feature)
            guide_out.append(guide_feature)

        for i in range(3):
            depth_feature = self.depth_conv[i](depth_feature)
            depth_out.append(depth_feature)

        fuse_feature = []
        for i in range(4):
            tmp = self.fuse_conv[i](depth=depth_out[3 - i],
                                    guide=guide_out[3 - i])
            fuse_feature.append(tmp)

        forward_hidden_list, _ = self.forward_gru_cell(torch.stack(fuse_feature, dim=1))
        forward_hidden_list = forward_hidden_list[-1]

        reversed_idx = list(reversed(range(4)))

        reverse_hidden_list, _ = self.reverse_gru_cell(torch.stack(fuse_feature, dim=1)[:, reversed_idx, ...])
        reverse_hidden_list = reverse_hidden_list[-1]
        reverse_hidden_list = reverse_hidden_list[:, reversed_idx, ...]

        fuse_out = []

        for i in range(4):
            tmp_out = self.compress_out[i](forward_hidden_list[:, i], reverse_hidden_list[:, i])
            fuse_out.append(tmp_out)

        out = self.up_conv(torch.cat(tuple(fuse_out), dim=1))
        return {'img_out': out + lr_up}


def make_model(args): return AHMF(args.scale, act=args.act)


# from lib import utils
# sample = {
#     'img_lr': torch.randn(1, 1, 32, 32).cuda(),
#     'lr_up': torch.randn(1, 1, 128, 128).cuda(),
#     'img_rgb': torch.randn(1, 3, 128, 128).cuda()
# }
# net = AHMF().cuda()
# out = net(sample)
# # print(out['img_out'].shape, out['img_out'].grad)
# out['img_out'].sum().backward()
# for para in net.parameters():
#     print(para.grad.sum())
# print(utils.get_parameter_number(net))