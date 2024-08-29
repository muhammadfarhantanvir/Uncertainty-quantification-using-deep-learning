import torch
import torch.nn as nn
import math
import numpy as np
import re
from typing import Dict, Any
import sys
import ast

def stacked_trainable_params(m):
    "Return all stacked parameters of `m`"
    return [p for p in m.mparams.values()]

def get_normal_param_priors(base_model, prior_variance):
    _priors = {}
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            _num = param.numel()
            _priors[name] = torch.distributions.normal.Normal(torch.zeros(_num),
                                            torch.ones(_num) * prior_variance)

    return _priors

def check_param_list_or_scalar(param_list:list|int, length:int):
    if isinstance(param_list, list):
        if len(param_list) == 1:
            param_list = param_list[0]
        elif len(param_list) != length:
            print(f"length of list `param_list` must match `length`! Using param_list = {param_list[0]}...")
            param_list = param_list[0]

    return param_list

def init_kaiming_uniform_(m, generator):
    try:
        device = next(m.parameters()).device
    except StopIteration:
        device = generator.device

    if generator.device != device:
        if device.type == 'cuda':
            generator = torch.Generator(device=device).manual_seed(generator.initial_seed())
        else:
            generator = torch.Generator(device='cpu').manual_seed(generator.initial_seed())

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5.0), generator=generator)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound, generator=generator)
    elif isinstance(m, (nn.LayerNorm, getattr(sys.modules[__name__], 'LayerNorm', nn.LayerNorm))):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        pass
    else:
        if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
            if m.weight.dim() > 1:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5.0), generator=generator)
            else:
                nn.init.normal_(m.weight, mean=1.0, std=0.02, generator=generator)
        
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_modules_recursive(m, init_func, generator):
    if next(m.children(), False):  # check if the module has registered children
        for c in m.children():
            init_modules_recursive(c, init_func, generator)
    else:
        for sm in m.modules():
            try:
                init_func(sm, generator)
            except ValueError as e:
                print(f"unable to init weights for module of type {type(sm)}: {e}")

def count_class_instances(dataset):
    # fastai has no clean option for class counts so this must be it
        pos = 0
        for i, x in enumerate(dataset.items[0].parts):
            if "train" in x:
                pos = i

        return torch.tensor(np.unique([path.parts[pos+1] for path in dataset.items], return_counts=True)[1])

def get_act_from_str(activation):
    if isinstance(activation, str):
        activation = activation.upper()
        if activation == 'RELU':
            return nn.ReLU
        elif activation == 'GELU':
            return nn.GELU
        elif activation == 'SELU':
            return nn.SELU
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    else:
        return activation

def read_int_list_from_run_data(run, path):
    return [int(g) for g in re.findall("\d+", run[path].fetch())]

def get_param_cond(run, param_name:str, na_str:str='None', na_val=None, convert_type=str):
    """Fetches the parameter and compares against the `na_str` value.
    Returns the value cast to `convert_type` or `na_val` if it is not specified.
    
    Careful: Do not specify a usable value for `na_val` as it may mask missing values in the run.
    Let the code fail in this case!"""

    value = run[param_name].fetch()

    if value == na_str:
        return na_val
    
    if isinstance(value, (bool, dict, set)):
        return value 
    if isinstance(value, list) or convert_type == list:
        value = ast.literal_eval(value)
    else:
        value = convert_type(value)

    return value

def get_param_if_exists(run, param_name:str, na_str:str='None', na_val=None, convert_type=str):
    if run.exists(param_name):
        return get_param_cond(run, param_name, na_str, na_val, convert_type)
    else:
        return na_val


def sample_hyperparameters(search_config: Dict[str, Any], rgen: torch.Generator) -> Dict[str, Any]:
    """Sample hyperparameters from the given search configuration using the provided random generator."""
    sampled_params = {}
    for param, config in search_config.items():
        if config['type'] == 'continuous':
            sampled_params[param] = torch.rand(1, generator=rgen).item() * (config['range'][1] - config['range'][0]) + config['range'][0]
        elif config['type'] == 'discrete':
            sampled_params[param] = config['values'][torch.randint(len(config['values']), (1,), generator=rgen).item()]
        else:
            raise ValueError(f"Unsupported search type for parameter {param}")
    return sampled_params
