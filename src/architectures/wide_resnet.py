# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wide ResNet with SNGP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_conv2d_layer(use_spec_norm=True, 
                      spec_norm_iteration=1, 
                      generator=None,
                      init_weight=True):
    """Defines type of Conv2D layer to use based on spectral normalization."""
    if use_spec_norm:
        def Conv2d(*args, **kwargs):
            _device = generator.device if generator is not None else None
            Conv2DBase = nn.Conv2d(*args, device=_device, **kwargs)
            if init_weight:
                Conv2DBase.weight = nn.init.kaiming_normal_(Conv2DBase.weight, generator=generator)
            nn.utils.parametrizations.spectral_norm(
                    Conv2DBase,
                    n_power_iterations=spec_norm_iteration,
                    )
            return Conv2DBase
        return Conv2d
    else:
        def Conv2d(*args, **kwargs):
            _device = generator.device if generator is not None else None
            Conv2DBase = nn.Conv2d(*args, device=_device, **kwargs)
            if init_weight:
                Conv2DBase.weight = nn.init.kaiming_normal_(Conv2DBase.weight, generator=generator)
            return Conv2DBase
        return Conv2d


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_p=0.0, 
                       use_spec_norm=True, spec_norm_iteration=1, activation=nn.ReLU, generator=None):
        super(BasicBlock, self).__init__()

        Conv2d = make_conv2d_layer(use_spec_norm, spec_norm_iteration, generator)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = activation(inplace=True)
        self.conv1 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act2 = activation(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout_p = dropout_p
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.act1(self.bn1(x))
        else:
            out = self.act1(self.bn1(x))
        out = self.act2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.dropout_p > 0:
            out = F.dropout2d(out, p=self.dropout_p, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_p=0.0, **kwargs):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_p, **kwargs)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_p, **kwargs):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout_p, **kwargs))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
        WideResNet from https://github.com/xternalz/WideResNet-pytorch/tree/master
        made like Wide Residual Networks (BMVC 2016) http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis
        but with additional SpectralNorm over all Convs.

        out-shape: num_classes if num_linear else 64 * widen_factor

      Args:
        num_conv: Total number of convolutional layers..
        num_classes: Number of output classes.
        widen_factor: Integer to multiply the number of typical filters by. "k"
            in WRN-n-k
        dropout_rate: Dropout rate.
        use_spec_norm: Whether to apply spectral normalization.
        spec_norm_iteration: Number of power iterations to perform for estimating
            the spectral norm of weight matrices.
        spec_norm_bound: Upper bound to spectral norm of weight matrices.
    """
    def __init__(self,
                 channels_in=3,
                 num_conv=28,
                 num_classes=10,
                 widen_factor=10,
                 dropout_p=0.1,
                 use_spec_norm=True, 
                 spec_norm_iteration=1, 
                 num_linear=0,
                 activation=nn.ReLU,
                 generator=None,
                 ):
        super(WideResNet, self).__init__()

        Conv2d = make_conv2d_layer(use_spec_norm, spec_norm_iteration, generator)
        self.num_linear = num_linear
        self.num_cov = num_conv
        self.use_sn = use_spec_norm
        self.activation_str = activation.__name__

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((num_conv - 4) % 6 == 0)
        n = (num_conv - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = Conv2d(channels_in, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_p, activation=activation, generator=generator)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_p, activation=activation, generator=generator)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_p, activation=activation, generator=generator)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.activation = activation(inplace=True)
        if self.num_linear:
            if use_spec_norm:
                self.fc = nn.utils.parametrizations.spectral_norm(nn.Linear(nChannels[3], num_classes), n_power_iterations=spec_norm_iteration)
            else:
                self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    @property
    def name(self) -> str:
        _sn_str = 'SN' if self.use_sn else ''
        return f"WideResNet{_sn_str}_{self.activation_str}_{self.num_cov}_{self.num_linear}"

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.activation(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        if self.num_linear:
            return self.fc(out)
        else:
            return out

