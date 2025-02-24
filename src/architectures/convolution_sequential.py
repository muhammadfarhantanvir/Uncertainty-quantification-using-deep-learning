import torch.nn as nn
from src.architectures.spectral_conv import SpectralConv


def convolution_sequential(input_dims, hidden_dims, output_dim, kernel_dim, k_lipschitz=None, p_drop=None, activation=nn.ReLU):
    channel_dim = input_dims[2]
    dims = [channel_dim] + hidden_dims
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralConv(dims[i], dims[i + 1], kernel_dim, (kernel_dim - 1) // 2, k_lipschitz ** (1./num_layers))
            layers.append(l)
        else:
            layers.append(nn.Conv2d(dims[i], dims[i + 1], kernel_dim, padding=(kernel_dim - 1) // 2))
        layers.append(activation())
        if p_drop is not None:
            layers.append(nn.Dropout(p=p_drop))
        layers.append(nn.MaxPool2d(2, padding=0))
    return nn.Sequential(*layers)
