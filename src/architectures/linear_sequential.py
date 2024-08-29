import numpy as np
import torch.nn as nn
from src.architectures.spectral_linear import SpectralLinear


def linear_sequential(input_dims, hidden_dims, output_dim, k_lipschitz=None, p_drop=None, activation=nn.ReLU):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralLinear(dims[i], dims[i + 1], k_lipschitz ** (1./num_layers))
            layers.append(l)
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(activation())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)
