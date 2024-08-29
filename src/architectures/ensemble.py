from fastai.vision.all import *
# torch.func requires PyTorch 2.0
from torch.func import stack_module_state, functional_call
import copy
from src.net_utils import init_kaiming_uniform_, init_modules_recursive
import torch.nn as nn

class VectorizedEnsemble(nn.Module):
    def __init__(self, models:list, init_weights:bool=True, generator=None):
        super(VectorizedEnsemble, self).__init__()

        self._num_models = 0
        self._base_models = None
        self._mparams = None
        self._mbuffers = None
        self.use_temp = False
        self._tmp_params = None
        self._tmp_buffers = None

        self._models = []
        self.models = models
        
        if init_weights: 
            self.init_model_weights(generator)

        # Initialize X_add and pred_k attributes
        self.X_add = None
        self.pred_k = None

    @property
    def models(self):
        return self._models
    
    @models.setter
    def models(self, models:list):
        if len(models):
            self.base_model = models
            self._num_models = len(models)
        else:
            self._base_models = None
            self._num_models = 0
        
        # register as child modules 
        self._models = nn.ModuleList(models)

        self.mparams, self.mbuffers = stack_module_state(models)

    @property
    def base_model(self):
        return self._base_models[0] if self._base_models is not None else None
    
    @base_model.setter
    def base_model(self, m: list):
        # wrap in list to prevent "cannot copy out of meta tensor; no data" error
        self._base_models = [self.get_stateless(m[0])]

    @property
    def mparams(self):
        return self._tmp_params if self.use_temp else self._mparams

    @mparams.setter
    def mparams(self, params):
        self._mparams = params
    
    @property
    def mbuffers(self):
        return self._tmp_buffers if self.use_temp else self._mbuffers
    
    @mbuffers.setter
    def mbuffers(self, buffers):
        self._mbuffers = buffers

    @property
    def particles(self):
        return torch.cat([_v.view(self.ensemble_size, -1) for _v in self.mparams.values()], dim=1)

    @property
    def name(self) -> str:
        return f"VecEnsemble_{self.base_model.name}" if self.base_model is not None else "VecEnsemble"

    @property
    def num_params(self):
        """Returns the number of trainable parameters in the model architecture."""
        return sum(p.numel() for p in self.base_model.parameters() if p.requires_grad) if self.base_model is not None else 0
    
    @property
    def ensemble_size(self):
        return self._num_models
    
    @property
    def param_shapes(self):
        return [param.shape for param in self.base_model.parameters() if param.requires_grad]

    def reshape_particles(self, particles):
        _weight_split = [np.prod(w) for w in self.param_shapes]
        _split_particles = torch.split(particles, _weight_split, 1)

        return [_split_particles[i].reshape((particles.shape[0], *shape)) for i, shape in enumerate(self.param_shapes)]

    def set_tmp_weights_from_flattened(self, flat_weights, device=None):
        _w_split = self.reshape_particles(flat_weights)

        self._tmp_params = {_kv[0]: _w_split[_i].to(device) for _i, _kv in enumerate(self._mparams.items())}
        self._tmp_buffers = {}
    
    def init_model_weights(self, generator=None):
        """Implements He (uniform) initialization, as used in the original implementation."""

        # this implementation is not compatible with the init_ensemble callback
        for _model in self.models: 
            # for each ensemble member, iterate over all layers
            init_modules_recursive(_model, init_func=init_kaiming_uniform_, generator=generator)                        

        _params, _buffers = stack_module_state(self.models)

        self.mparams = {k: v.detach().requires_grad_(True) for k,v in _params.items()}
        self.mbuffers = {k: v.detach().requires_grad_(True) for k,v in _buffers.items()}

    def get_stateless(self, model):
            """Construct a "stateless" version of one of the models. It is "stateless" in
            the sense that the parameters are meta Tensors and do not have storage."""
            _base_model = copy.deepcopy(model)
            return _base_model.to('meta')

    def vec_forward(self, params, buffers, x):
        return functional_call(self.base_model, (params, buffers), (x,))

    def forward(self, x):
        # shape (num_models, batch_size, out_features)
        return torch.vmap(self.vec_forward, 
                            in_dims=(0,0,None),  # in_dims=None here tells vmap to map the same minibatch over all models 
                            out_dims=0,
                            randomness="different",
                            )(self.mparams, self.mbuffers, x)
