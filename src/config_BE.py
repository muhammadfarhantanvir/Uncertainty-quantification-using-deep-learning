from typing import List, Dict, Any, Callable, Type, Optional, Set
import torch
import sys
from fastai.vision.all import *
from src.base_config import BaseConfig
from src.constants import REPULSIVE_ENSEMBLE, PROJECT_NAME
from src.architectures.selunet import ConvNet
from src.architectures.convnext import ConvNeXt
from src.architectures.ensemble import VectorizedEnsemble
from src.optimizer import wrapped_partial
from src.densities.kernel import RBF
from src.architectures.gradient_estim import SpectralSteinEstimator
from src.densities.distributions import Normal
from src.callback import Unorm_post, f_WGD, auto_repulse, repulse_additional_points, stack_module_states, predict_on_prior_weights, activate_logits
from src.net_utils import get_param_cond, get_param_if_exists

class BEConfig(BaseConfig):
    def __init__(self, config: Dict[str, Any], use_random_search: bool = False):
        super().__init__(config, use_random_search)

        be_config = config['UQMethods'][REPULSIVE_ENSEMBLE]
        self.num_models = be_config['training']['num_models']
        self.prior_variance = be_config['training']['prior_variance']
        self.pred_dist_std = be_config['training']['pred_dist_std']
        self.eta = be_config['training']['eta']
        self.density_method = be_config['training']['density_method']
        self.do_auto_repulse = be_config['training']['do_auto_repulse']
        self.annealing_steps = be_config['training']['annealing_steps']
        self.gamma = be_config['training']['gamma']

    def create_model(self, dls: DataLoaders) -> nn.Module:
        device = self.device if dls is None else dls.device
        if self.base_network == "ConvNet":
            base_net = [ConvNet(**self.net_params).to(device) for _ in range(self.num_models)]
        elif self.base_network == "ConvNeXt":
            base_net = [ConvNeXt(**self.net_params).to(device) for _ in range(self.num_models)]
        else:
            raise ValueError(f"Unsupported network.base_network: {self.base_network}")

        self.model = VectorizedEnsemble(models=base_net, init_weights=True, generator=self.wgen)

        return self.model

    def get_optimizer(self) -> Callable:
        return wrapped_partial(Adam, decouple_wd=self.use_wd)

    def get_callbacks(self, train: bool=True) -> List[Callback]:
        if train:
            prior = Normal(torch.zeros(self.model.num_params, device=self.device),
                        torch.ones(self.model.num_params, device=self.device) * self.prior_variance,
                        generator=self.wgen)

            kernel = RBF()
            grad_estim_kernel = RBF()
            grad_estim = SpectralSteinEstimator(eta=self.eta, kernel=grad_estim_kernel)

            filename = self.get_filename()

            callbacks = [
                stack_module_states(),
                activate_logits(),
                Unorm_post(prior, self.pred_dist_std, self.prior_variance),
                predict_on_prior_weights(),
                f_WGD(kernel, grad_estim, self.gamma, self.annealing_steps, self.density_method),
                GradientClip(),
                SaveModelCallback(monitor='valid_loss', min_delta=2e-3, fname=filename, every_epoch=False, at_end=False, with_opt=True),
            ]

            if self.density_method == "sge" or self.density_method == "kde":
                callbacks.append(auto_repulse() if self.do_auto_repulse else repulse_additional_points())
        else:
            callbacks = [
                stack_module_states(),
                GradientClip(),
            ]

        return callbacks

    def get_additional_tags(self) -> Set[str]:
        return {
            "BE",
            "AdamW" if self.use_wd else "Adam",
            self.net_params['activation'].__name__
        }

    def get_additional_eval_callbacks(self) -> List[Callback]:
        prior = Normal(torch.zeros(self.model.num_params, device=self.device),
                       torch.ones(self.model.num_params, device=self.device) * self.prior_variance,
                       generator=self.wgen)
        return [
            stack_module_states(),
            Unorm_post(prior, self.pred_dist_std),
            GradientClip(),
        ]

    def get_model_params(self) -> Dict[str, Any]:
        return None  # BE doesn't have specific model params to log

    def get_base_net(self) -> Type[nn.Module]:
        return self.model.base_model

    def should_reduce_preds(self) -> bool:
        return True

    def get_neptune_model_name(self) -> str:
        
        neptune_model_name = ""

        if self.dataset_sub_directory is not None:
            neptune_model_name = self.dataset_name + "/" + self.dataset_sub_directory
        else:
            neptune_model_name = self.dataset_name
            
        match neptune_model_name:
            case "domainnet/quickdraw":
                 return "CS24-DMNTQDRE"
            case "domainnet/sketch":
                 return "CS24-DMNTSKRE"
            case "domainnet/real":
                 return "CS24-DMNTRLRE"
            case "domainnet/infograph":
                 return "CS24-DMNTIGRE"
            case "domainnet/painting":
                 return "CS24-DMNTPTRE"
            case "domainnet/clipart":
                 return "CS24-DMNTCARE"
            case "cifar10":
                return "CS24-CF10CNRE" if self.base_network == "ConvNeXt" else "CS24-CF10RE"
            case "cifar10_2":
                return "CS24-CF102CNRE"  
            case "dr250":
                return "CS24-DRCNRE" if self.base_network == "ConvNeXt" else "CS24-DR250RE"
            case "bgraham_dr":
                return "CS24-BGDRCNRE"
            case "HAM10000":
                return "CS24-HAMCNRE"  
            case "HAM10000_2":
                return "CS24-HAM2CNRE"
            case _:
                raise ValueError("Unknown dataset in configuration file.")

    @staticmethod
    def load_config_from_run(run):
        config = super(BEConfig, BEConfig).load_config_from_run(run)

        config["training"]["num_models"] = get_param_cond(run, f"config/ensemble_size", convert_type=int)
        config["training"]["prior_variance"] = get_param_cond(run, f"config/optimizer/initial_hyperparameters/prior_var", convert_type=float)
        config["training"]["pred_dist_std"] = get_param_cond(run, f"config/optimizer/initial_hyperparameters/pred_dist_std", convert_type=float)
        config["training"]["eta"] = get_param_cond(run, f"config/optimizer/initial_hyperparameters/grad_estim_eta", convert_type=float)
        config["training"]["density_method"] = get_param_cond(run, f"config/optimizer/initial_hyperparameters/density_method", convert_type=str)
        config["training"]["do_auto_repulse"] = get_param_if_exists(run, f"config/optimizer/initial_hyperparameters/do_auto_repulse", na_val=True, convert_type=bool)
        config["training"]["annealing_steps"] = get_param_cond(run, f"config/optimizer/initial_hyperparameters/annealing_steps", convert_type=int)
        config["training"]["gamma"] = get_param_cond(run, f"config/optimizer/initial_hyperparameters/gamma", convert_type=float)

        config["evaluation"]["callbacks"] = [{
            "callback_name": "calc_validation_total_entropy",
            "callback_params": {
              "do_activate": true,
              "log_base_str": "two"
            }
        }]

        return config

if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = 'src/config_default.json'
    currentConfig = BEConfig.load_config(config_path)
    be_config = BEConfig(currentConfig)
    be_config.train_and_run_model()