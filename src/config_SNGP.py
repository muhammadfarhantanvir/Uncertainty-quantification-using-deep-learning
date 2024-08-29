from typing import List, Dict, Any, Callable, Type, Optional, Set
from fastai.vision.all import *
import sys
sys.path.append("./")
from src.base_config import BaseConfig
from src.constants import SNGP_CONSTANT
from src.architectures.wide_resnet import WideResNet
from src.architectures.convnext_sngp import ConvNeXtSNGP
from src.architectures.sngp_classification_layer import SNGP
from src.optimizer import wrapped_partial, SGD_with_nesterov
from src.callback import unpack_model_output_dict, gp_classifier_helper
from src.net_utils import get_param_cond, get_param_if_exists
from src.run_utils import parse_list_params
import neptune

class SNGPConfig(BaseConfig):
    def __init__(self, config: Dict[str, Any], use_random_search: bool = False):
        super().__init__(config, use_random_search)
        sngp_config = config['UQMethods'][SNGP_CONSTANT]
        self.model_params = sngp_config['model_params']
        self.feature_multiplier = self.model_params.get('feature_multiplier', 64)

        self.model_params['num_classes'] = self.net_params['num_classes']
        if self.base_network == "WideResNet":
            self.model_params['pre_clas_features'] = self.net_params['num_classes'] * self.feature_multiplier
        elif self.base_network == "ConvNeXtSNGP":
            self.model_params['pre_clas_features'] = self.net_params['dims'][-1]
        else:
            raise ValueError(f"Unsupported base_network: {self.base_network}")

    def create_model(self, dls: DataLoaders) -> nn.Module:
        if self.base_network == "WideResNet":
            base_net = WideResNet(**self.net_params, generator=self.wgen).to(dls.device)
        elif self.base_network == "ConvNeXtSNGP":
            base_net = ConvNeXtSNGP(**self.net_params, generator=self.wgen).to(dls.device)
        else:
            raise ValueError(f"Unsupported network.base_network: {self.base_network}")

        self.model = SNGP(**self.model_params, pre_classifier=base_net, generator=self.wgen)

        return self.model

    def get_optimizer(self) -> Callable:
        if self.use_nesterov:
            return wrapped_partial(SGD_with_nesterov, 
                                mom=1.0 - self.one_minus_momentum, 
                                wd=self.wd,
                                decouple_wd=self.use_wd, 
                                nesterov=self.use_nesterov)
        else:
            return wrapped_partial(SGD_with_nesterov, mom=1-self.one_minus_momentum, wd=self.wd, nesterov=False)

    def get_callbacks(self, train: bool=True) -> List[Callback]:
        if train:
            filename = self.get_filename()

            callbacks = [
                unpack_model_output_dict(),
                gp_classifier_helper(),
                GradientClip(),
                SaveModelCallback(monitor='valid_loss', min_delta=2e-3, fname=filename, every_epoch=False, at_end=False, with_opt=True),
            ]
        else:
            callbacks = [
                unpack_model_output_dict(),
                GradientClip(),
            ]

        return callbacks

    def get_additional_tags(self) -> Set[str]:
        return {
            "SNGP",
            "SGD_nesterov" if self.use_nesterov else "SGD",
            self.net_params['activation'].__name__
        }

    def log_config(self, run: neptune.Run):
        super().log_config(run)

        run["config/optimizer/initial_hyperparameters/use_nesterov"] = str(self.use_nesterov)

    def get_additional_eval_callbacks(self) -> List[Callback]:
        return []

    def get_model_params(self) -> Dict[str, Any]:
        return self.model_params

    def get_base_net(self) -> Type[nn.Module]:
        return self.model.pre_classifier

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
                 return "CS24-DMNTQDSNGP"
            case "domainnet/sketch":
                 return "CS24-DMNTSKSNGP"
            case "domainnet/real":
                 return "CS24-DMNTRLSNGP"
            case "domainnet/infograph":
                 return "CS24-DMNTIGSNGP"
            case "domainnet/painting":
                 return "CS24-DMNTPTSNGP"
            case "domainnet/clipart":
                 return "CS24-DMNTCASNGP"
            case "cifar10":
                return "CS24-CF10CNSG" if self.base_network == "ConvNeXtSNGP" else "CS24-CF10SG"
            case "cifar10_2":
                return "CS24-CF102CNSG" if self.base_network == "ConvNeXtSNGP" else "CS24-CF102SG"
            case "dr250":
                return "CS24-DRCNSG" if self.base_network == "ConvNeXtSNGP" else "CS24-DR250SG"
            case "bgraham_dr":
                return "CS24-BGDRCNSG"
            case "HAM10000":
                return "CS24-HAMCNSG"  
            case "HAM10000_2":
                return "CS24-HAM2CNSG"
            case _:
                raise ValueError("Unknown dataset in configuration file.")

    @staticmethod
    def load_config_from_run(run):
        config = super(SNGPConfig, SNGPConfig).load_config_from_run(run)

        config["training"]["one_minus_momentum"] = 1 - get_param_cond(run, "config/optimizer/initial_hyperparameters/mom", convert_type=float)
        config["training"]["use_nesterov"] = get_param_if_exists(run, "config/optimizer/initial_hyperparameters/use_nesterov", na_val=True, convert_type=bool)

        config["model_params"] = parse_list_params(get_param_cond(run, "config/model/model_params", convert_type=dict))

        config["evaluation"]["callbacks"] = [{
            "callback_name": "calc_sngp_uncertainty",
            "callback_params": {
              "log_base_str": "natural"
            }
        }]

        print(f"sngp_config:\n{config}")

        return config

if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = 'src/config_default.json'
    currentConfig = SNGPConfig.load_config(config_path)
    sngp_config = SNGPConfig(currentConfig)
    sngp_config.train_and_run_model()