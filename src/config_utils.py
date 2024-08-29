import torch
from src.net_utils import get_param_cond
from src.config_BE import BEConfig
from src.config_PN import PNConfig
from src.config_SNGP import SNGPConfig
from src.constants import REPULSIVE_ENSEMBLE, SNGP_CONSTANT, POSTERIOR_NETWORK

def get_config_type(method: str):
    """Get the appropriate config class based on the method."""
    if method == REPULSIVE_ENSEMBLE:
        return BEConfig
    elif method == SNGP_CONSTANT:
        return SNGPConfig
    elif method == POSTERIOR_NETWORK:
        return PNConfig
    else:
        raise ValueError(f"Unsupported method: {method}")

def get_method_str_from_run(run) -> str:
    model_name = run["config/model/model_name"].fetch()

    match model_name:
        case "VectorizedEnsemble":
            me_str = REPULSIVE_ENSEMBLE
        case "SNGP":
            me_str = SNGP_CONSTANT
        case "PosteriorNetwork":
            me_str = POSTERIOR_NETWORK
        case _:
            raise ValueError(f"Unexpected UQ method: {model_name}")
        
    return me_str

def create_config_from_run(run, device, mode):
    split_seed = get_param_cond(run, "config/train_valid_split_seed", convert_type=int)

    # recreate the original train and valid dataloaders
    dataset_name = get_param_cond(run, "io_files/resources/dataset/name")
    dataset_subdir = get_param_cond(run, "io_files/resources/dataset/sub_directory")
    include_classes = get_param_cond(run, "io_files/resources/dataset/include_classes")
    exclude_classes = get_param_cond(run, "io_files/resources/dataset/exclude_classes")

    # determine which type of model was trained in the run
    currentMethod = get_method_str_from_run(run)
    model_type = get_config_type(currentMethod)
        
    # load the required hyperparameters from the run (depending on the model type)
    model_config = model_type.load_config_from_run(run)

    tags = get_param_cond(run, "sys/tags", na_str='{}', na_val={}, convert_type=dict)
    config = {
        "currentMethod": currentMethod,
        "random_seed": None,  # only used for random param search, don't seed!
        "dataset": {
            "name": dataset_name,
            "sub_directory": dataset_subdir,
            "include_classes": include_classes,
            "exclude_classes": exclude_classes,
            "train_valid_split_seed": split_seed,
        },
        "neptune_mode": mode,
        "device": str(device),
        "tags": tags,
        "UQMethods": {
            currentMethod: {}
        }
    }
    config["UQMethods"][currentMethod] = model_config

    return config, model_type

def get_valid_folds(model_config):
    # re-create the original datablock 
    model_config.prepare_generators()
    _ = model_config.prepare_data()
    ds = model_config.data_block.datasets(model_config.data_path / "train")

    # get the validation indices from the train/valid dataset
    valid_idx = ds.splits[1]

    # divide the remaining indices into num_folds-1 many parts
    train_idx = ds.splits[0]

    rand_mask = torch.randperm(len(train_idx), generator=model_config.tgen)
    split_size = len(valid_idx)  # any different choice would change the size of the training set!
    valid_folds = [t for t in torch.tensor(train_idx)[rand_mask].split(split_size=split_size)]

    return valid_folds
