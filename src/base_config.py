import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Literal, Type, Optional
import torch
import neptune
from pathlib import Path
from contextlib import nullcontext
from fastai.vision.all import *
from src.constants import (
    ALLOWED_NEPTUNE_MODES, DATASETS, LR_SCHEDULERS, LOSSES, DATASET_DIRECTORY, 
    BASE_NETWORKS, ACTIVATION_METHODS, EVALUATION_CALLBACKS, 
    PROJECT_NAME, MODEL_PATH
)
from src.optimizer import CosineDecayWithWarmup, WarmUpPieceWiseConstantSchedStep
from src.neptune_callback import NeptuneCallbackBayesianEnsemble as NeptuneCallback
from src.evaluation import eval_model
from src.net_utils import get_act_from_str, get_param_cond, get_param_if_exists
from src.run_utils import get_source_file_paths_from_run, is_wd_config, process_callbacks, parse_list_params
from src.datasets import get_data_block, mixed_sampling
from neptune.types.mode import Mode as RunMode
from neptune.utils import stringify_unsupported
from src.loss import (
    EnsembleClassificationLoss, CrossEntropyClassificationLoss, UCELoss,
    WeightedEnsembleClassificationLoss, WeightedCrossEntropyClassificationLoss,
    WeightedUCELoss, FocalEnsembleClassificationLoss,
    FocalCrossEntropyClassificationLoss, FocalUCELoss
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseConfig(ABC):
    def __init__(self, config: Dict[str, Any], use_random_search: bool = False):
        self.use_random_search = use_random_search
        self.config = config
        self.method = config.get('currentMethod')
        self.method_config = config['UQMethods'][self.method]
        
        # General settings
        ds_name = config['dataset']['name'].split("/")
        self.dataset_name: str = ds_name[0]
        self.dataset_sub_directory: Optional[str] = config['dataset'].get('sub_directory') if len(ds_name) < 2 else ds_name[1] 
        self.neptune_mode: str = config['neptune_mode']
        self.device: torch.device = torch.device(config['device'])
        self.tags: Set[str] = set(config["tags"])

        # Training settings
        self.training = self.method_config['training']
        self.batch_size: int = self.training['batch_size']
        self.num_epochs: int = self.training['num_epochs']
        self.learning_rate: float = self.training['learning_rate']
        self.weights_seed: Optional[int] = self.training.get('weights_seed')
        self.train_valid_split_seed: Optional[int] = config['dataset'].get('train_valid_split_seed', None)
        if self.train_valid_split_seed is None:
            self.train_valid_split_seed = config['UQMethods'][self.method]['training'].get('train_valid_split_seed', None)
            if self.train_valid_split_seed is not None:
                print("Deprecation warning: Found `train_valid_split_seed` in the training parameters section. "
                      "This is deprecated, please move the seed to `dataset/train_valid_split_seed`!")
        self.random_seed = self.config.get('random_seed', None)
        self.use_random_flip = self.training.get('use_random_flip', False)
        self.use_random_erasing = self.training.get('use_random_erasing', False)
        self.use_randaugment = self.training.get('use_randaugment', False)
        self.rand_flip_prob = self.training.get('rand_flip_prob', 0.5)
        self.random_erasing_prob = self.training.get('random_erasing_prob', 0.3)
        self.use_wd: bool = self.training.get('use_wd', False)
        self.wd: float = self.training.get('wd', 0.0)
        self.use_nesterov: bool = self.training.get('use_nesterov', False)
        self.one_minus_momentum: float = self.training.get('one_minus_momentum', 0.1)

        # Dataset settings
        self.include_classes: Optional[List[str]] = config['dataset'].get('include_classes')
        self.exclude_classes: Optional[List[str]] = config['dataset'].get('exclude_classes')
        self.use_index_splitter: bool = config['dataset'].get('use_index_splitter', False)
        self.valid_indices: Optional[List[int]] = config['dataset'].get('valid_indices')
        self.valid_pct: float = 0.2

        # Loss function settings
        self.loss_func: str = self.method_config['loss']["loss_func"]
        self.loss_params: Dict[str, Any] = self.method_config['loss']["loss_params"]
        self.source_files: List[str] = self.method_config['source_files']
        
        # Network settings
        self.base_network: str = self.method_config['network']['base_network']
        self.net_params: Dict[str, Any] = self.method_config['network']['net_params']
        self.net_params['activation'] = get_act_from_str(self.net_params['activation'])
        
        # Evaluation settings
        self.evaluation_callbacks: List[Dict[str, Any]] = self.method_config['evaluation']['callbacks']

        # Learning rate scheduler settings
        self.use_lr_scheduler: bool = self.method_config['training']['use_lr_scheduler']
        self.lr_sched: str = "None"
        self.lr_sched_params: Dict[str, Any] = {}
        self.lr_fit_one_cycle_params: Dict[str, Any] = {}
        if self.use_lr_scheduler:
            self.lr_sched = self.method_config['schedulers']["lr_sched"]
            if self.lr_sched == "FitOneCycle":
                self.lr_fit_one_cycle_params = self.method_config['schedulers']["FitOneCycle_params"]
            else:
                self.lr_sched_params = self.method_config['schedulers']["lr_sched_params"]
        
        # Early stopping settings
        self.early_stopping: Dict[str, Any] = self.method_config['training'].get('early_stopping', {})
        self.early_stopping_monitor: str = self.early_stopping.get('monitor', 'valid_loss')
        self.early_stopping_min_delta: float = self.early_stopping.get('min_delta', 0.001)
        self.early_stopping_patience: int = self.early_stopping.get('patience', 15)

        # Random parameter search settings
        self.parameter_search: Dict[str, Any] = self.method_config.get('parameter_search', {})

        self.dataset_test: str = self.dataset_name
        self.FitOneCycle: bool = (self.lr_sched == "FitOneCycle")
        
        self.dls: Optional[DataLoaders] = None
        self.dl_test: Optional[DataLoader] = None

        # Cross-validation settings
        self.current_fold: int = 0
        self.fold_0_id = config.get("fold_0_id", None)

        # Validate properties
        self.validate_Property('dataset.name', self.dataset_name, DATASETS)
        self.validate_Property('neptune_mode', self.neptune_mode, ALLOWED_NEPTUNE_MODES)
        self.validate_Property('loss.loss_func', self.loss_func, LOSSES)
        self.validate_Property('network.base_network', self.base_network, BASE_NETWORKS)
        self.validate_Property('network.net_params.activation', self.net_params['activation'], ACTIVATION_METHODS)
        self.validate_Array('callback_name', self.evaluation_callbacks, EVALUATION_CALLBACKS)
        if self.use_lr_scheduler:
            self.validate_Property('schedulers.lr_sched', self.lr_sched, LR_SCHEDULERS)

    def validate_Property(self, property_key: str, value: Any, accepted_values: Literal):
        if property_key == 'network.net_params.activation':
            if isinstance(value, str):
                if value.upper() not in accepted_values.__args__:
                    raise ValueError(f"Invalid value for {property_key}: {value}, can accept only following values:\n {accepted_values.__args__}")
            elif isinstance(value, type) and value.__name__.upper() not in accepted_values.__args__:
                raise ValueError(f"Invalid value for {property_key}: {value.__name__}, can accept only following values:\n {accepted_values.__args__}")
        else:
            if value not in accepted_values.__args__:
                raise ValueError(f"Invalid value for {property_key}: {value}, can accept only following values:\n {accepted_values.__args__}")

    def validate_Array(self, property_key: str, array: Optional[List[Dict[str, Any]]], accepted_values: Literal):
        if array is not None:
            for configObject in array:
                property_value = configObject.get(property_key)
                if property_value is not None and property_value not in accepted_values.__args__:
                    raise ValueError(f"Invalid {property_key}: {property_value}, can accept only the following values: {accepted_values.__args__}")

    @property
    def model_name(self):
        return self.model.name if self.model is not None else None
    
    @abstractmethod
    def create_model(self, dls: DataLoaders) -> nn.Module:
        pass

    @abstractmethod
    def get_optimizer(self) -> Callable:
        pass

    def get_loss_func(self) -> nn.Module:
        loss_func_map = {
            "EnsembleClassificationLoss": EnsembleClassificationLoss,
            "CrossEntropyClassificationLoss": CrossEntropyClassificationLoss,
            "UCELoss": UCELoss,
            "WeightedEnsembleClassificationLoss": WeightedEnsembleClassificationLoss,
            "WeightedCrossEntropyClassificationLoss": WeightedCrossEntropyClassificationLoss,
            "WeightedUCELoss": WeightedUCELoss,
            "FocalEnsembleClassificationLoss": FocalEnsembleClassificationLoss,
            "FocalCrossEntropyClassificationLoss": FocalCrossEntropyClassificationLoss,
            "FocalUCELoss": FocalUCELoss,
        }
        if self.loss_func in loss_func_map:
            if "UCELoss" in self.loss_func:
                self.loss_params["output_dim"] = self.net_params["num_classes"]
            if "EnsembleClassificationLoss" in self.loss_func:
                self.loss_params["num_classes"] = self.net_params["num_classes"]
            if self.loss_params.get("alpha", None) is not None:
                self.loss_params["alpha"] = torch.tensor(self.loss_params["alpha"], device=self.dls.device)
            if self.loss_params.get("weight", None) is not None:
                self.loss_params["weight"] = torch.tensor(self.loss_params["weight"], device=self.dls.device)
            return loss_func_map[self.loss_func](**self.loss_params)
        else:
            raise ValueError(f"Unsupported loss_func: {self.loss_func}")

    @abstractmethod
    def get_callbacks(self, train: bool=True) -> List[Callback]:
        pass

    def prepare_generators(self):
        torch.cuda.set_device(self.device)
        
        # seed the generator for the train/valid split
        self.tgen = torch.Generator()
        if self.train_valid_split_seed is not None:
            self.tgen = self.tgen.manual_seed(self.train_valid_split_seed)
        else:
            self.train_valid_split_seed = self.tgen.seed()

        # seed the generator for the weights
        self.wgen = torch.Generator(device=self.device)
        if self.weights_seed is not None:
            self.wgen = self.wgen.manual_seed(self.weights_seed)
        else:
            self.weights_seed = self.wgen.seed()

    def prepare_run(self, add_tags_list: Set[str]) -> Tuple[neptune.Run, DataLoaders, DataLoader]:
        self.prepare_generators()

        if self.use_random_search:
            add_tags_list.add("random search")
        
        self.dls, self.dl_test = self.prepare_data()
        
        self.tags.update(add_tags_list)
        run = neptune.init_run(
            source_files=self.source_files,
            project=PROJECT_NAME,
            tags=self.tags,
            mode=self.neptune_mode
        )

        self.run_id = run['sys/id'].fetch()
        self.log_config(run)
        
        return run, self.dls, self.dl_test

    def log_config(self, run: neptune.Run):
        # seeds
        run["config/train_valid_split_seed"] = str(self.train_valid_split_seed)
        run["config/weights_seed"] = str(self.weights_seed)
        run["config/random_seed"] = str(self.random_seed) or "None"

        # fold
        run["config/fold_index"] = self.current_fold
        run["config/fold_0_id"] = self.fold_0_id if self.fold_0_id is not None else self.run_id

        # dataset
        run["io_files/resources/dataset/name"] = self.dataset_name
        run["io_files/resources/dataset/sub_directory"] = self.dataset_sub_directory or "None"
        run["io_files/resources/dataset/include_classes"] = stringify_unsupported(self.include_classes) if self.include_classes is not None else "None"
        run["io_files/resources/dataset/exclude_classes"] = stringify_unsupported(self.exclude_classes) if self.exclude_classes is not None else "None"

        # learning rate
        run["config/optimizer/use_lr_sched"] = self.use_lr_scheduler
        if self.use_lr_scheduler:
            run[f"config/optimizer/lr_sched/name"] = self.lr_sched
            lr_items = self.lr_fit_one_cycle_params.items() if self.FitOneCycle else self.lr_sched_params.items()
            for key, value in lr_items:
                run[f"config/optimizer/lr_sched/{key}"] = str(value)

        # loss
        run["config/loss/name"] = self.loss_func
        for key, value in self.loss_params.items():
            run[f"config/loss/{key}"] = str(value)

        # augmentation
        run["config/augmentation/use_random_flip"] = self.use_random_flip
        run["config/augmentation/use_random_erasing"] = self.use_random_erasing
        run["config/augmentation/use_randaugment"] = self.use_randaugment
        run["config/augmentation/rand_flip_prob"] = self.rand_flip_prob
        run["config/augmentation/random_erasing_prob"] = self.random_erasing_prob

        # early stopping
        run["config/optimizer/early_stopping/monitor"] = self.early_stopping_monitor
        run["config/optimizer/early_stopping/min_delta"] = self.early_stopping_min_delta
        run["config/optimizer/early_stopping/patience"] = self.early_stopping_patience

    def log_after_fit(self, run: neptune.Run, learn, nspace, fit_idx):
        # log the actual number of epochs for which the model trained
        actual_epochs = len(learn.recorder.values)
        run[f"{nspace}/metrics/fit_{fit_idx}/training/actual_epochs"] = actual_epochs

        # log the specific Adam optimizer version
        if run[f"{nspace}/config/optimizer/name"].fetch() == "Adam":
            run[f"{nspace}/config/optimizer/name"] = "AdamW" if self.use_wd else "Adam"

    def get_filename(self):
        return f"{self.dataset_name}_{self.model_name}_{self.run_id}"

    def train_and_run_model(self, do_eval_default: bool = True, model_version: Optional[str] = None, fold: int = 0):
        logger.info(f"All properties loaded successfully from configuration file for method: {self.method}.")

        self.current_fold = fold
        
        try:
            run, dls, dl_test = self.prepare_run(self.get_additional_tags())
            model = self.create_model(dls)
            loss_func = self.get_loss_func()
            opt_func = self.get_optimizer()
            
            neptune_model = self.get_neptune_model_name() if model_version is None else model_version
            
            neptune_callback = NeptuneCallback(run=run,
                                            model_name=type(model).__name__,
                                            model_params=self.get_model_params(),
                                            net_name=type(self.get_base_net()).__name__,
                                            net_params=self.net_params,
                                            upload_saved_models=None)
            
            callbacks = self.get_callbacks()
            callbacks.append(neptune_callback)
            callbacks.append(EarlyStoppingCallback(monitor=self.early_stopping_monitor, 
                                                min_delta=self.early_stopping_min_delta, 
                                                patience=self.early_stopping_patience))
            
            path = MODEL_PATH if run._mode != RunMode.DEBUG else None
            
            learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, 
                            lr=self.learning_rate, cbs=callbacks, path=path)
            
            with learn.no_bar() if run._mode != RunMode.DEBUG else nullcontext():
                self.fit_model(learn)
            
            self.log_after_fit(run, learn, neptune_callback.base_namespace, neptune_callback.fit_index)
            
            if do_eval_default:
                self.evaluate_model(learn, run, neptune_callback, dl_test, fold, neptune_model)
            
            run.stop()
        except Exception as e:
            e.add_note(f"Error occured while training fold {fold}.")
            raise


    def fit_model(self, learn: Learner):
        if not self.use_lr_scheduler:
            logger.info("No scheduler applied")
            learn.fit(self.num_epochs)
        elif self.FitOneCycle:
            logger.info("FitOneCycle applied")
            # Ensure the correct parameters are used
            fit_one_cycle_params = {
                "div": self.lr_fit_one_cycle_params.get("div", 25.0),
                "div_final": self.lr_fit_one_cycle_params.get("div_final", 1e4),
                "pct_start": self.lr_fit_one_cycle_params.get("pct_start", 0.3),
                "wd": self.lr_fit_one_cycle_params.get("weight_decay", 0.1)
            }
            learn.fit_one_cycle(self.num_epochs, lr_max=self.learning_rate, **fit_one_cycle_params)
        else:
            logger.info("Custom scheduler applied")
            sched = {'lr': self.get_lr_scheduler()}
            lr_scheduler = ParamScheduler(sched)
            learn.fit(self.num_epochs, cbs=lr_scheduler)

    def get_lr_scheduler(self) -> Callable:
        if self.lr_sched == "CosineDecayWithWarmup":
            lr_sched_params = {
                "start": self.learning_rate,
                "lr_warmup_epochs": self.lr_sched_params["lr_warmup_epochs"],
                "num_epochs": self.num_epochs,
                "min_lr": self.lr_sched_params.get("min_lr", 0.0)  
            }
            return CosineDecayWithWarmup(**lr_sched_params)
        elif self.lr_sched == "WarmUpPieceWiseConstantSchedStep":
            lr_sched_params = {
                "start": self.learning_rate * self.batch_size / self.lr_sched_params.get("reference_batch_size", self.batch_size),
                "lr_warmup_epochs": self.lr_sched_params["lr_warmup_epochs"],
                "lr_decay_epochs": self.lr_sched_params["lr_decay_epochs"],
                "lr_decay_ratio": self.lr_sched_params["lr_decay_ratio"],
                "num_epochs": self.num_epochs,
                "pct_offs": 1.0 / (len(self.dls.train) * self.num_epochs)
            }
            return WarmUpPieceWiseConstantSchedStep(**lr_sched_params)
        elif self.lr_sched == "FitOneCycle":
            return None
        else:
            raise ValueError(f"Unsupported lr_sched: {self.lr_sched}")

    def evaluate_model(self, learn: Learner, run: neptune.Run, neptune_callback: NeptuneCallback, dl_test: DataLoader, fold: int, neptune_model: str):
        if fold == 0:
            log_fields = {
                "data/validation/name": self.dataset_name,
                "data/validation/size": learn.dls[1].n,
                "data/testing/name": self.dataset_test,
                "data/testing/size": dl_test.n,
            }
        else:
            log_fields = None

        learn.loss_func.is_logits = True

        _log_scale = torch.log2(torch.tensor(10, dtype=torch.float))

        _eval_callbacks = process_callbacks(self.evaluation_callbacks, [], _log_scale)
        _eval_callbacks.extend(self.get_additional_eval_callbacks())

        model_version_id = eval_model(
            learn=learn,
            neptune_model=neptune_model,
            neptune_project=PROJECT_NAME,
            run=run,
            dl_valid=learn.dls[1],
            dl_test=dl_test,
            neptune_fit_idx=neptune_callback.fit_index,
            do_validate=True,
            do_test=True,
            neptune_base_namespace=neptune_callback.base_namespace,
            log_additional=log_fields,
            mode=self.neptune_mode,
            reduce_preds=self.should_reduce_preds(),
            uncertainty_callbacks=_eval_callbacks,
            fold=fold,
            inner=True,
        )

        return model_version_id

    @abstractmethod
    def get_additional_eval_callbacks(self) -> List[Callback]:
        pass

    @abstractmethod
    def get_additional_tags(self) -> Set[str]:
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_base_net(self) -> Type[nn.Module]:
        pass

    @abstractmethod
    def should_reduce_preds(self) -> bool:
        pass

    @abstractmethod
    def get_neptune_model_name(self) -> str:
        pass

    def prepare_data(self) -> Tuple[DataLoaders, DataLoader]:
        """Prepare and return data loaders."""
        try:
            if self.dataset_name == "domainnet":
                self.data_path = Path(DATASET_DIRECTORY) / self.dataset_name / self.dataset_sub_directory
            else:
                self.data_path = Path(DATASET_DIRECTORY) / self.dataset_name


            # Common parameters for all datasets        
            common_params = {
                'valid_pct': self.valid_pct,
                'use_random_flip': self.use_random_flip,
                'use_random_erasing': self.use_random_erasing,
                'use_randaugment': self.use_randaugment,
                'rand_flip_prob': self.rand_flip_prob,
                'random_erasing_prob': self.random_erasing_prob,
                'include_classes': self.include_classes,
                'exclude_classes': self.exclude_classes,
                'generator': self.tgen,
                'useIndexSplitter': self.use_index_splitter,
                'valid_indices': self.valid_indices,
            }
            
            if self.dataset_name in ["cifar10", "cifar10_2", "HAM10000", "HAM10000_2", "bgraham_dr_as_OOD","domainnet"]:
                self.data_block = get_data_block(self.dataset_name, **common_params)
            
            elif self.dataset_name in ["dr250", "bgraham_dr"]:
                self.data_block = get_data_block(self.dataset_name,
                                            train_labels_path=self.data_path/"trainLabels.csv",
                                            test_labels_path=self.data_path/"testLabels.csv",
                                            **common_params)
            
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
            train_path = self.data_path / "train"
            self.dls = self.data_block.dataloaders(train_path, bs=self.batch_size, device=self.device)
            
            # Mixed sampling for HAM10000 and HAM10000_2
            if self.dataset_name in ["HAM10000", "HAM10000_2"]:
                self.dls = mixed_sampling(self.dls)
            
            # Prepare test dataloader
            items_test = self.data_block.get_items(self.data_path/"test")
            self.dl_test = self.dls.test_dl(items_test, rm_type_tfms=None, num_workers=0, with_labels=True, with_input=False)
            
            return self.dls, self.dl_test
        
        except Exception as e:
            logger.error(f"Failed to prepare data for dataset {self.dataset_name}: {str(e)}")
            raise

    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        with open(config_file) as file:
            config = json.load(file)
        return config

    def print_properties(self):
        logger.info("Printing all properties")
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                logger.info(f"{key}: {value}")

    @staticmethod
    def load_config_from_run(run):
        source_files = get_source_file_paths_from_run(run, f"source_code/files")

        training = {
            "batch_size": get_param_cond(run, "config/batch_size", convert_type=int),
            "learning_rate": get_param_cond(run, "config/optimizer/initial_hyperparameters/lr", convert_type=float),
            "use_wd": is_wd_config(run),
            "wd": get_param_cond(run, "config/optimizer/initial_hyperparameters/wd", convert_type=float),
            "num_epochs": get_param_cond(run, "metrics/fit_0/n_epoch", convert_type=int),
            "use_random_flip": get_param_if_exists(run, "config/augmentation/use_random_flip", na_val=True, convert_type=bool),
            "use_random_erasing": get_param_if_exists(run, "config/augmentation/use_random_erasing", na_val=False, convert_type=bool),
            "use_randaugment": get_param_if_exists(run, "config/augmentation/use_randaugment", na_val=False, convert_type=bool),
            "rand_flip_prob": get_param_if_exists(run, "config/augmentation/rand_flip_prob", na_val=0.5, convert_type=float),
            "random_erasing_prob": get_param_if_exists(run, "config/augmentation/random_erasing_prob", na_val=0.3, convert_type=float),
            "weights_seed": get_param_cond(run, "config/weights_seed", convert_type=int),
            "use_lr_scheduler": get_param_cond(run, "config/optimizer/use_lr_sched", convert_type=bool),
            "early_stopping": {
                "monitor": get_param_if_exists(run, "config/optimizer/early_stopping/monitor", na_val="valid_loss", convert_type=str),
                "min_delta": get_param_if_exists(run, "config/optimizer/early_stopping/min_delta", na_val=2e-3, convert_type=float),
                "patience": get_param_if_exists(run, "config/optimizer/early_stopping/patience", na_val=20, convert_type=int)
            }
        }

        network = {
            "base_network": get_param_cond(run, "config/model/net_name"),
            "net_params": parse_list_params(get_param_cond(run, "config/model/net_params", convert_type=dict)),
        }

        loss_params = parse_list_params(get_param_cond(run, "config/loss", convert_type=dict))
        del loss_params["name"]
        loss = {
            "loss_func": get_param_cond(run, "config/loss/name"),
            "loss_params": loss_params,
        }

        schedulers = {}
        if training["use_lr_scheduler"]:
            lr_sched_params = parse_list_params(get_param_cond(run, "config/optimizer/lr_sched", convert_type=dict))
            schedulers["lr_sched"] = lr_sched_params["name"]
            sched_params_path = "FitOneCycle_params" if lr_sched_params["name"] == "FitOneCycle" else "lr_sched_params"
            del lr_sched_params["name"]
            schedulers[sched_params_path] = lr_sched_params

        config = {
            "source_files": source_files,
            "training": training,
            "network": network,
            "loss": loss,
            "schedulers": schedulers,
            "evaluation": {}
        }

        return config
