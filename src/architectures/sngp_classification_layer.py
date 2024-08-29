import torch
from torch import nn
from src.densities.gaussian_process import RandomFeatureGaussianProcess, orthogonal_random_features_initializer, CrossEntropyRandomFeatureCovariance

import inspect
from functools import partial

class SNGP(nn.Module):
    """
    A Spectral Normalized Gaussian Process classification head. Given an input (..., H),
    this applies three layers
    Linear: (..., H) --> (..., H) + activation + dropout
    Projection: (..., H) --> (..., R), for R < H, and R=128 as default
    Random features: (..., R) --> (..., D) for D >> R and D=1024 as default

    In order to approximate the Gaussian Process (GP) kernel, we should have D >> R. This works when
    we reduce the dimension R, which here is done by a linear projection layer.

    See RandomFeatureGaussianProcess documentation for details about the GP kernel settings.

    Args:
        in_features: (int) the last dimension of the input
        num_classes: (int) number of classes to predict on
        reduction_dim: (int) the dimension of the lower dimensional embedding that represents the final hidden state
        classif_dropout: (float) classifier dropout rate. defaults to 0.2
        activation: (str) the name of the pytorch activation function layer (e.g. ReLU, Tanh). Defaults to Tanh.
    Returns:
        A  tuple or dict of logits, covariance, for logits a (..., C) dimensional logits tensor and covariance a
        (batch_size, batch_size) covariance tensor.
    """
    def __init__(
        self,
        pre_clas_features,
        num_classes,
        reduction_dim=128,
        classif_dropout=0.2,
        activation='Identity',
        pre_classifier=None,
        pre_clas_spec_norm=True,
        pre_clas_sn_iterations=1,
        reduce_dim_trainable=False,
        generator:torch.Generator=None,
        **kwargs,
    ):
        super(SNGP, self).__init__()

        # these are some good defaults for the gp layer
        gp_kwargs = {
            'random_features': 1024,
            'normalize_input':True, #True in paper
            'kernel_type':'gaussian',
            'kernel_scale':2.0, # 2. in paper
            'init_output_bias':0.0,
            'output_weight_trainable':True, #False in paper?
            'output_bias_trainable':True, #True in paper
            'use_custom_random_features':True, #True in paper
            'covariance_momentum':0.999, #-1 in tf-code, but paper says 0.999
            'covariance_likelihood':'gaussian',
            'custom_random_features_initializer':partial(orthogonal_random_features_initializer, 
                                                         random_norm=True, stddev=0.05, generator=generator), #OrthogonalRandomFeatures in paper
            # 'custom_random_features_initializer':nn.init.orthogonal_,
            'covariance_estimator':CrossEntropyRandomFeatureCovariance,
            'return_dict':True,
            'generator':generator,
        }
        
        for k, v in kwargs.items():
            if k != 'self' and k in inspect.getfullargspec(
                RandomFeatureGaussianProcess.__init__
            ).args:
                gp_kwargs[k] = v
        
        _device = generator.device if generator is not None else None
        if pre_classifier is not None:
            self.pre_classifier = pre_classifier
        else:
            self.pre_classifier = nn.Linear(pre_clas_features, pre_clas_features, device=_device)
            if pre_clas_spec_norm:
                self.pre_classifier = nn.utils.parametrizations.spectral_norm(self.pre_classifier, n_power_iterations=pre_clas_sn_iterations)

        self.reduce_dim_layer = nn.Linear(pre_clas_features, reduction_dim, bias=False, device=_device) #like in tf code
        # self.reduce_dim_layer.weight = nn.init.normal_(self.reduce_dim_layer.weight) doesnt accept generator so make it like below
        with torch.no_grad():
            self.reduce_dim_layer.weight.normal_(generator=generator)
        if pre_clas_spec_norm:
            self.reduce_dim_layer = nn.utils.parametrizations.spectral_norm(self.reduce_dim_layer, n_power_iterations=pre_clas_sn_iterations)
        if not reduce_dim_trainable:
            self.reduce_dim_layer.weight = self.reduce_dim_layer.weight.detach()

        self.gp_classifier = RandomFeatureGaussianProcess(
            in_features=reduction_dim,
            out_features=num_classes,
            **gp_kwargs,
        )
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.Tanh()
        self.dropout = nn.Dropout(p=classif_dropout)

    @property
    def name(self) -> str:
        if self.pre_classifier is not None:
            if hasattr(self.pre_classifier, 'name') and self.pre_classifier.name is not None:
                return f"SNGP_{self.pre_classifier.name}"
            else:
                return f"SNGP_{type(self.pre_classifier).__name__}"
        else:
            return "SNGP"

    
    def forward(self, x, last_epoch=None):
        x = self.dropout(self.activation(self.pre_classifier(x)))
        x = self.reduce_dim_layer(x)
        return self.gp_classifier(x, last_epoch)
