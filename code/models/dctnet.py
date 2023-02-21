# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   dctnet.py
@Time    :   2022/8/1 18:25
@Desc    :
"""
import torch
import kornia
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale


def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.rfft(v, 1, onesided=False)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    # v = torch.irfft(V, 1, onesided=False)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)

class Weight_Prediction_Network(nn.Module):
    def __init__(self, n_feats=64):
        super(Weight_Prediction_Network, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv_dilation = nn.Conv2d(f, f, kernel_size=3, padding=1,
                                       stride=3, dilation=2)

    def forward(self, x):  # x is the input feature
        x = self.conv1(x)
        shortCut = x
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=7, stride=3)
        x = self.relu(self.conv_max(x))
        x = self.relu(self.conv3(x))
        x = self.conv3_(x)
        x = F.interpolate(x, (shortCut.size(2), shortCut.size(3)),
                          mode='bilinear', align_corners=False)
        shortCut = self.conv_f(shortCut)
        x = self.conv4(x + shortCut)
        x = self.sigmoid(x)
        return x


class Coupled_Layer(nn.Module):
    def __init__(self,
                 coupled_number=32,
                 n_feats=64,
                 kernel_size=3):
        super(Coupled_Layer, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size
        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_rgb_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_rgb_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

    def forward(self, feat_dlr, feat_rgb):
        shortCut = feat_dlr
        feat_dlr = F.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_1, self.kernel_depth_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_depth_1], dim=0),
                            padding=1)
        feat_dlr = F.relu(feat_dlr, inplace=True)
        feat_dlr = F.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_2, self.kernel_depth_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_depth_2], dim=0),
                            padding=1)
        feat_dlr = F.relu(feat_dlr + shortCut, inplace=True)
        shortCut = feat_rgb
        feat_rgb = F.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_1, self.kernel_rgb_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_rgb_1], dim=0),
                            padding=1)
        feat_rgb = F.relu(feat_rgb, inplace=True)
        feat_rgb = F.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_2, self.kernel_rgb_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_rgb_2], dim=0),
                            padding=1)
        feat_rgb = F.relu(feat_rgb + shortCut, inplace=True)
        return feat_dlr, feat_rgb


class Coupled_Encoder(nn.Module):
    def __init__(self,
                 n_feat=64,
                 n_layer=4):
        super(Coupled_Encoder, self).__init__()
        self.n_layer = n_layer
        self.init_deep = nn.Sequential(
            nn.Conv2d(1, n_feat, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(True),
        )
        self.init_rgb = nn.Sequential(
            nn.Conv2d(3, n_feat, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(True),
        )
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer() for i in range(self.n_layer)])

    def forward(self, feat_dlr, feat_rgb):
        feat_dlr = self.init_deep(feat_dlr)
        feat_rgb = self.init_rgb(feat_rgb)
        for layer in self.coupled_feat_extractor:
            feat_dlr, feat_rgb = layer(feat_dlr, feat_rgb)
        return feat_dlr, feat_rgb


class Decoder_Deep(nn.Module):
    def __init__(self,
                 n_feats=64):
        super(Decoder_Deep, self).__init__()
        self.Decoder_Deep = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(True),
            nn.Conv2d(n_feats // 2, n_feats // 4, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(True),
            nn.Conv2d(n_feats // 4, 1, kernel_size=3, padding=1),  # in_channels, out_channels, kernel_size
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.Decoder_Deep(x)


class DCTNet(nn.Module):
    '''
    Solver for the problem: min_{x} |x-d|_2^2+lambd|L(x)-L(r).*w|_2^2
    d - input low-resolution image
    r - guidance image (we want transfer the gradient of r into d)
        input RGB image
    z - output super-resolution image
    L - Laplacian operator
    w - Edge weight matrix (to be learned by WeightLearning Network)
        *Note: the solution of this problem is idct(p/c)
               p = dct(lambd*L(L(r)).*w + d)
               c = lambd*K^2+1
               K = self.get_K()
    '''

    def __init__(self, scale, lambd=3., n_feats=64):
        super(DCTNet, self).__init__()
        self.n_feats = n_feats
        self.scale = scale
        self.lambd = nn.Parameter(
            torch.nn.init.normal_(
                torch.full(size=(1, self.n_feats, 1, 1), fill_value=lambd), mean=0.1, std=0.3))
        #                torch.nn.init.kaiming_normal(
        #                        torch.full(size=(1,self.n_feats,1,1),fill_value=lambd)))
        self.WPNet = Weight_Prediction_Network()

        self.Encoder_coupled = Coupled_Encoder()
        self.Decoder_depth = Decoder_Deep()

    def get_K(self, H, W, dtype, device):
        pi = torch.acos(torch.Tensor([-1]))
        cos_row = torch.cos(pi * torch.linspace(0, H - 1, H) / H).unsqueeze(1).expand(-1, W)
        cos_col = torch.cos(pi * torch.linspace(0, W - 1, W) / W).unsqueeze(0).expand(H, -1)
        kappa = 2 * (cos_row + cos_col - 2)
        kappa = kappa.to(dtype).to(device)
        return kappa[None, None, :, :]  # shape [1,1,H,W]

    def get_Lap(self, dtype, device):
        laplacian = kornia.filters.Laplacian(3)
        f = laplacian
        return f

    def forward(self, samples):
        x, y = samples['lr_up'], samples['img_rgb']
        # x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        # BGR2RGB
        # permute = [2, 1, 0]
        # y = y[:, permute]
        # x - input depth image d, shape [N,C,H,W]
        # y - guidance RGB image r, shape [N,1,H,W] or [N,C,H,W]

        if len(y.shape) == 3:
            y = y[:, None, :, :]
        N, C, H, W = x.shape

        high_Dim_D, high_Dim_R = self.Encoder_coupled(x, y)

        # get weight
        weight = self.WPNet(high_Dim_R)
        # weight=self.WPNet(y)

        # get SR image (64 channel)
        lambd = torch.exp(self.lambd).to(x.device)
        k2 = self.get_K(H, W, x.dtype, x.device).pow(2)
        L = self.get_Lap(x.dtype, x.device)
        # P = dct_2d(
        #         torch.mul(lambd*L(L(high_Dim_R)),weight)+high_Dim_D
        #         )
        P = dct_2d(
            lambd * L(torch.mul(L(high_Dim_R), weight)) + high_Dim_D
        )
        C = lambd * k2 + 1

        z = idct_2d(P / C)
        SR_deepth = self.Decoder_depth(z)
        return {'img_out': SR_deepth}

def make_model(args): return DCTNet(args.scale)