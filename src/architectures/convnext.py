# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.architectures.drop import DropPath
from src.architectures.wide_resnet import make_conv2d_layer

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        kernel_size (int): Kernel size for depthwise convolution. Default: 7.
        padding (int): Padding for depthwise convolution. Default: 3.
        activation (nn.Module): Activation function. Default: nn.GELU.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, 
                 kernel_size=7, padding=3, activation=nn.GELU, Conv2d=nn.Conv2d):
        super().__init__()
        self.dwconv = Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = activation()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        kernel_size (int): Kernel size for the initial downsampling conv layer. Default: 4.
        stride (int): Stride for the initial downsampling conv layer. Default: kernel_size.
        downsample_layers (int): Number of generic downsampling layers. Must be one less than the number of stages. Default: -1.
        stages (int): Number of stages. Default: length of `dims` and `depths`.
        activation (nn.Module): Activation function for blocks. Default: nn.GELU.
    """   
    def __init__(self, 
                 in_chans=3, 
                 num_classes=1000, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 layer_scale_init_value=1e-6, 
                 head_init_scale=1., 
                 kernel_size=4, 
                 stride=-1, 
                 downsample_layers=-1, 
                 stages=-1, 
                 activation=nn.GELU,
                 include_head=True,
                 generator=None,
                 use_spec_norm=False,
                 spec_norm_iteration=1):  
        super().__init__()
        self.activation_str = activation.__name__ 
        self.num_cov = sum(depths)
        self.include_head = include_head
        self.generator = generator
        self.use_spec_norm = use_spec_norm
        self.spec_norm_iteration = spec_norm_iteration

        Conv2d = make_conv2d_layer(use_spec_norm, spec_norm_iteration, generator, init_weight=False)
        
        if len(dims) != len(depths):
            stages = min(len(depths), len(dims))
            print(f"Warning: Expected 'dims' and 'depths' to have the same length but they are different!\n\
                   Warning: Falling back to the smaller length: {stages}")
        elif stages > 0 and stages != len(depths):
            print(f"Warning: Got explicit parameter 'stages'={stages}, but length of 'depths' is {len(depths)}!")
        elif stages == -1:
            stages = len(depths)

        if downsample_layers > 0 and downsample_layers != stages - 1:
            print(f"Warning: Got explicit parameter 'downsample_layers'={downsample_layers}, but expected {stages - 1}!")
        downsample_layers = stages - 1

        self.downsample_layers = nn.ModuleList()
        _first_stride = stride if stride > 0 else kernel_size
        stem = nn.Sequential(
            Conv2d(in_chans, dims[0], kernel_size=kernel_size, stride=_first_stride),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(downsample_layers):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(stages):
            stage = nn.Sequential(
                *[self.make_block(dim=dims[i], drop_path=dp_rates[cur + j], 
                                  layer_scale_init_value=layer_scale_init_value, 
                                  activation=activation, Conv2d=Conv2d) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        if self.include_head:
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
            if use_spec_norm:
                self.head = nn.utils.parametrizations.spectral_norm(self.head, n_power_iterations=spec_norm_iteration)
        else:
            self.head = None

        self.apply(self._init_weights)

    def make_block(self, dim, drop_path, layer_scale_init_value, activation, Conv2d):
        return Block(dim, drop_path, layer_scale_init_value, activation=activation, Conv2d=Conv2d)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if self.generator is not None and self.generator.device != m.weight.device:
                generator = torch.Generator(device=m.weight.device).manual_seed(self.generator.initial_seed())
            else:
                generator = self.generator
            torch.nn.init.trunc_normal_(m.weight, std=.02, generator=generator)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = x.float()  
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x
    
    def forward(self, x): 
        x = self.forward_features(x)
        x = self.norm(x.mean([-2, -1]))
        if self.include_head:
            x = self.head(x)
        return x

    @property
    def name(self) -> str:
        sn_str = f"SN{self.spec_norm_iteration}" if self.use_spec_norm else ""
        return f"ConvNeXt{self.activation_str}_{self.num_cov}{sn_str}"

# Testing code
if __name__ == "__main__":
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    model = ConvNeXt(in_chans=3, num_classes=1000, activation=nn.GELU, include_head=True, 
                     generator=torch.Generator(), use_spec_norm=True, spec_norm_iteration=1).to(device) 
    print("Model name:", model.name)

    input_tensor = torch.randn(4, 3, 224, 224).to(device)  
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)