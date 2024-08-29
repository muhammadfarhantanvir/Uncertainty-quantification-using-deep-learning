import torch
import torch.distributions.normal as dist_norm

class Normal(dist_norm.Normal):
    def __init__(self, loc, scale, validate_args=None, generator=None, **kwargs):
        super(Normal, self).__init__(loc, scale, validate_args=validate_args, **kwargs)

        self.generator = generator

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape), generator=self.generator)
