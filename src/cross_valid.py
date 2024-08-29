import sys
import secrets

sys.path.append("./")
import argparse
from fastai.vision.all import *
import neptune
from src.neptune_tracking import track_cross_valid
from src.constants import PROJECT_NAME
from src.net_utils import get_param_cond
from src.config_utils import create_config_from_run, get_valid_folds
from src.run_utils import set_missing_field, get_fold_metrics

def cross_validate(run, model_version, num_folds, torch_device:str, mode):
    device = torch.device(torch_device)

    config, model_type = create_config_from_run(run, device, mode)
    model_config = model_type(config)

    valid_folds = get_valid_folds(model_config)
    
    if len(valid_folds) < (num_folds - 1):
        run.stop()
        raise ValueError(f"Too many folds for cross validation! Requested {num_folds} but can do {len(valid_folds) + 1} at most!")

    # stop the original run (not needed anymore)
    run_id = get_param_cond(run, "sys/id")
    config["fold_0_id"] = run_id
    _id = run_id.split("-")[1]
    run.stop()

    model_version_id = get_param_cond(model_version, "sys/id")


    split_success = 0

    for i in range(1, num_folds):
        if model_version.exists(f"run/id_fold_{i}"):
            print(f"Skipping fold {i}: already exists (run_id: {model_version[f'run/id_fold_{i}'].fetch()})")
            split_success += 1
            continue

        fold_config = deepcopy(config)
        fold_config["tags"].update(["Hyper", f"cv_{_id}_f{i}"])
        # modify config to set the current validation split explictly 
        # and force the dataloader to use the `IndexSplitter`
        fold_config["dataset/valid_indices"] = valid_folds[i-1]
        fold_config[f"dataset/use_index_splitter"] = True

        # draw a new random weight seed and set in the config
        fold_config[f"UQMethods/{fold_config['currentMethod']}/training/weights_seed"] = secrets.randbelow(int(1e16))

        # train the model using the config
        # and log to the initial model version (validation and testing)
        print(f"Starting fold {i}/{num_folds-1}")
        model_type(fold_config).train_and_run_model(model_version=model_version_id, fold=i)

        split_success += 1

    # once we have all the splits
    if split_success == num_folds - 1:
        # assemble a cross validation section that has the mean and std for each numerical metric
        fold_metrics = get_fold_metrics(eval_default=True, eval_ood=False, eval_tta=False)
        track_cross_valid(model_version, fold_metrics, overwrite=False)
        return True
    else: 
        print(f"Unexpected number of folds completed: got {split_success}, expected {num_folds - 1}")
        return False
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform cross-validation based on an existing run. All folds will log into the same model version and the initial run will be used as fold 0.")

    parser.add_argument("-m", "--model_version_id", required=True, default=None, help="id of an existing model version to extend")
    parser.add_argument("-f", "--num_folds", required=False, default=5, type=int, help="final number of folds (including the existing run)")
    parser.add_argument("-c", "--cuda_device", required=False, default="cuda:0", choices=["cpu", "cuda", "cuda:0", "cuda:1"], help="cuda device or cpu")
    parser.add_argument("-n", "--neptune_mode", required=False, default="async", choices=["async", "debug", "read-only"], help="neptune mode used for the run and model version")

    args = parser.parse_args()

    try:
        _mode = "read-only" if args.neptune_mode == 'debug' else args.neptune_mode
        model_version = neptune.init_model_version(with_id=args.model_version_id, project=PROJECT_NAME, mode=_mode)
    except Exception as e:
        e.add_note(f"Failed to open model_version '{PROJECT_NAME}/{args.model_version_id}'.")
        raise

    try:
        run_id = get_param_cond(model_version, "run/id")
        run = neptune.init_run(project=PROJECT_NAME, with_id=run_id, mode="read-only")
    except Exception as e:
        e.add_note(f"Failed to open run '{PROJECT_NAME}/{run_id}'.")
        raise

    if cross_validate(run, model_version, args.num_folds, args.cuda_device, mode=args.neptune_mode):
        print(f"Completed {args.num_folds}-fold-cross-validation for run {run_id} and model version {args.model_version_id}")
    else:
        print(f"Failed to complete cross-validation for run {run_id} and model version {args.model_version_id}!")
