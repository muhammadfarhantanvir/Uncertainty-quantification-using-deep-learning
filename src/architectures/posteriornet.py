from torch import nn
from src.architectures.linear_sequential import linear_sequential
from src.densities.NormalizingFlowDensity import NormalizingFlowDensity
from src.densities.BatchedNormalizingFlowDensity import BatchedNormalizingFlowDensity
from src.densities.MixtureDensity import MixtureDensity
from src.net_utils import init_modules_recursive, init_kaiming_uniform_


class PosteriorNetwork(nn.Module):
    def __init__(self,
                 encoder:nn.Module,
                 output_dim,  # Output dimension. int
                 latent_dim=10,  # Latent dimension. int
                 no_density=False,  # Use density estimation or not. boolean
                 density_type='radial_flow',  # Density type. string
                 n_density=8,  # Number of density components. int
                 hidden_dim_class=64,  # Hidden dimensions. list of ints
                 k_lipschitz_class=None,  # Lipschitz constant. float or None (if no lipschitz)
                 init_weights:bool=True,
                 generator=None,
                ):
        super().__init__()

        self.encoder = encoder

        # Architecture parameters
        self.output_dim, self.hidden_dim, self.latent_dim = output_dim, hidden_dim_class, latent_dim
        self.k_lipschitz = k_lipschitz_class
        self.no_density, self.density_type, self.n_density = no_density, density_type, n_density

        if self.no_density:
            self.linear_classifier = linear_sequential(input_dims=[self.latent_dim],  # Linear classifier for sequential training
                                                    hidden_dims=[self.hidden_dim],
                                                    output_dim=self.output_dim,
                                                    k_lipschitz=self.k_lipschitz)
        else:
            # Normalizing Flow -- Normalized density on latent space
            if self.density_type == 'planar_flow':
                self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type) for _ in range(self.output_dim)])
            elif self.density_type == 'radial_flow':
                self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type) for _ in range(self.output_dim)])
            elif self.density_type == 'batched_radial_flow':
                self.density_estimation = BatchedNormalizingFlowDensity(c=self.output_dim, dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type.replace('batched_', ''))
            elif self.density_type == 'iaf_flow':
                self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type) for _ in range(self.output_dim)])
            elif self.density_type == 'normal_mixture':
                self.density_estimation = nn.ModuleList([MixtureDensity(dim=self.latent_dim, n_components=n_density, mixture_type=self.density_type) for _ in range(self.output_dim)])
            else:
                raise NotImplementedError
            
            self.batch_norm = nn.BatchNorm1d(num_features=self.latent_dim)

        if init_weights:
            init_modules_recursive(self, init_func=init_kaiming_uniform_, generator=generator)

    @property
    def name(self) -> str:
        _dens_str = "no_density" if self.no_density else self.density_type
        _enc_str = getattr(self.encoder, 'name', type(self.encoder).__name__)
        return f"PosteriorNet_{_enc_str}_{_dens_str}" if self.encoder is not None else "PosteriorNet"

    def forward(self, input):
        zk = self.encoder(input)
        if self.no_density:  # Ablated model without density estimation
            logits = self.linear_classifier(zk)
            return logits
        return self.batch_norm(zk)
