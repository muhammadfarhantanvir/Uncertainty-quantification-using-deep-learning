import torch
import torch.nn as nn
from src.architectures.convnext import ConvNeXt, Block
from src.architectures.wide_resnet import make_conv2d_layer

class SNGPBlock(Block):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
                 kernel_size=7, padding=3, activation=nn.GELU,
                 use_spec_norm=True, spec_norm_iteration=1, generator=None):
        super().__init__(dim, drop_path, layer_scale_init_value,
                         kernel_size, padding, activation)

        Conv2d = make_conv2d_layer(use_spec_norm, spec_norm_iteration, generator, init_weight=False)
        self.dwconv = Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.pwconv1 = nn.utils.parametrizations.spectral_norm(self.pwconv1, n_power_iterations=spec_norm_iteration)
        self.pwconv2 = nn.utils.parametrizations.spectral_norm(self.pwconv2, n_power_iterations=spec_norm_iteration)

class ConvNeXtSNGP(ConvNeXt):
    def make_block(self, dim, drop_path, layer_scale_init_value, activation, Conv2d):
        return SNGPBlock(dim, drop_path, layer_scale_init_value, activation=activation,
                         use_spec_norm=self.use_spec_norm,
                         spec_norm_iteration=self.spec_norm_iteration,
                         generator=self.generator)

    @property
    def name(self) -> str:
        sn_str = f"SN{self.spec_norm_iteration}" if self.use_spec_norm else ""
        return f"ConvNeXt{self.activation_str}_{self.num_cov}_SNGP{sn_str}"

# Testing code
if __name__ == "__main__":
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    model = ConvNeXtSNGP(in_chans=3, num_classes=1000, activation=nn.GELU, include_head=True,
                         generator=torch.Generator(), use_spec_norm=True, spec_norm_iteration=1
                         ).to(device)

    print("Model name:", model.name)

    input_tensor = torch.randn(4, 3, 224, 224).to(device)
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)