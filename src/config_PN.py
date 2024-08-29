import torch
from typing import List, Dict, Any, Callable, Type, Optional, Set
from fastai.vision.all import *
from src.base_config import BaseConfig
from src.constants import POSTERIOR_NETWORK, PROJECT_NAME
from src.architectures.convolution_linear_sequential import convolution_linear_sequential
from src.architectures.convnext import ConvNeXt
from src.architectures.posteriornet import PosteriorNetwork
from src.optimizer import wrapped_partial
from src.callback import PosteriorDensity, NormalizeAlphaPred, DataModCallback
from src.net_utils import count_class_instances, get_param_cond

class PNConfig(BaseConfig):
    def __init__(self, config: Dict[str, Any], use_random_search: bool = False, use_cross_validation: bool = False):
        super().__init__(config, use_random_search)
        self.use_cross_validation = use_cross_validation
        
        pn_config = config['UQMethods'][POSTERIOR_NETWORK]
        self.post_net_params = pn_config['post_net_params']
        self.no_density = self.post_net_params['no_density']
        self.post_net_params['output_dim'] = self.net_params['num_classes']
        self.post_net_params['k_lipschitz_class'] = self.net_params.get('k_lipschitz', None)

        if self.base_network == "ConvNeXt":
            self.post_net_params['latent_dim'] = self.net_params['num_classes']
            self.post_net_params['hidden_dim_class'] = self.net_params['dims'][-1]
        elif self.base_network == "ConvLinSeq":
            self.post_net_params['latent_dim'] = self.net_params.get('output_dim', None)
            self.post_net_params['hidden_dim_class'] = self.net_params['linear_hidden_dims'][-1]
        else:
            raise ValueError(f"Unsupported base_network: {self.base_network}")

    def create_model(self, dls: DataLoaders) -> nn.Module:
        if self.base_network == "ConvLinSeq":
            net_params = self.net_params.copy()
            del net_params["num_classes"]
            base_net = convolution_linear_sequential(**net_params).to(dls.device)
        elif self.base_network == "ConvNeXt":
            base_net = ConvNeXt(**self.net_params).to(dls.device)
        else:
            raise ValueError(f"Unsupported network.base_network: {self.base_network}")

        self.model = PosteriorNetwork(encoder=base_net, generator=self.wgen, **self.post_net_params)

        return self.model

    def get_optimizer(self) -> Callable:
        return wrapped_partial(Adam, mom=1.0 - self.one_minus_momentum, wd=self.wd, decouple_wd=self.use_wd)

    def get_callbacks(self, train: bool=True) -> List[Callback]:
        if train:
            filename = self.get_filename()

            callbacks = [
                DataModCallback(),
                GradientClip(),
                SaveModelCallback(monitor='valid_loss', min_delta=2e-3, fname=filename, every_epoch=False, at_end=False, with_opt=True),
            ]

        else:
            callbacks = [
                DataModCallback(),
                GradientClip(),
            ]
            
        if not self.no_density:
            num_class_samples = count_class_instances(self.dls.train_ds)
            callbacks.extend([
                PosteriorDensity(output_dim=self.post_net_params['output_dim'],
                                num_class_samples=num_class_samples),
                NormalizeAlphaPred(normalize_before_loss=False),
            ])

        return callbacks

    def get_additional_tags(self) -> Set[str]:
        return {"PN", "AdamW" if self.use_wd else "Adam"}

    def get_additional_eval_callbacks(self) -> List[Callback]:
        return []

    def get_model_params(self) -> Dict[str, Any]:
        return self.post_net_params

    def get_base_net(self) -> Type[nn.Module]:
        return self.model.encoder

    def should_reduce_preds(self) -> bool:
        return False

    def get_neptune_model_name(self) -> str:
        
        neptune_model_name = ""

        if self.dataset_sub_directory is not None:
            neptune_model_name = self.dataset_name + "/" + self.dataset_sub_directory
        else:
            neptune_model_name = self.dataset_name
            
        match neptune_model_name:
            case "domainnet/quickdraw":
                 return "CS24-DMNTQDPON"
            case "domainnet/sketch":
                 return "CS24-DMNTSKPON"
            case "domainnet/real":
                 return "CS24-DMNTRLPON"
            case "domainnet/infograph":
                 return "CS24-DMNTIGPON"
            case "domainnet/painting":
                 return "CS24-DMNTPTPON"
            case "domainnet/clipart":
                 return "CS24-DMNTCAPON"
            case "cifar10":
                return "CS24-CF10CNPN" if self.base_network == "ConvNeXt" else "CS24-CF10PN"
            case "cifar10_2":
                return "CS24-CF102CNPN"  if self.base_network == "ConvNeXt" else "CS24-CF102PN"
            case "dr250":
                return "CS24-DRCNPN" if self.base_network == "ConvNeXt" else "CS24-DR250PN"
            case "bgraham_dr":
                return "CS24-BGDRCNPN"
            case "HAM10000":
                return "CS24-HAMCNPN"  
            case "HAM10000_2":
                return "CS24-HAM2CNPN"
            case _:
                raise ValueError("Unknown dataset in configuration file.")

    @staticmethod
    def load_config_from_run(run):
        config = super(PNConfig, PNConfig).load_config_from_run(run)

        config["training"]["one_minus_momentum"] = 1 - get_param_cond(run, "config/optimizer/initial_hyperparameters/mom", convert_type=float)
        config["training"]["no_density"] = get_param_cond(run, "config/model/model_params", convert_type=bool)
        
        config["post_net_params"] = get_param_cond(run, "config/model/model_params", convert_type=dict)

        config["evaluation"]["callbacks"] = [{
            "callback_name": "calc_dirichlet_uncertainty",
            "callback_params": {
              "eps": 1e-20,
              "normalize_uncertainty": False
            }
        }]

        return config

    def prepare_run(self, add_tags_list: Set[str]):
        if self.base_network == "ConvLinSeq":
            torch.set_default_dtype(torch.double)
        elif self.base_network == "ConvNeXt":
            torch.set_default_dtype(torch.float32)
        else:
            raise ValueError(f"Unsupported base_network: {self.base_network}")

        return super().prepare_run(add_tags_list)

if __name__ == "__main__":
    import sys
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = 'src/config_default.json'
    currentConfig = PNConfig.load_config(config_path)
    pn_config = PNConfig(currentConfig)
    pn_config.train_and_run_model()