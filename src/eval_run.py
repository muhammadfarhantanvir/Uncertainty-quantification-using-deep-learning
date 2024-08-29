from fastai.vision.all import *

import sys
sys.path.append("./")
import argparse

from src.evaluation import eval_model, eval_model_ood
from src.net_utils import get_param_cond, get_param_if_exists
from src.config_utils import create_config_from_run, get_valid_folds
from src.run_utils import open_model_version, process_callbacks
from src.constants import DATASET_DIRECTORY, PROJECT_NAME, MODEL_PATH, DATASETS

import neptune

def eval_run(run_id: str=None, model_version: str=None, do_eval_default: bool = True, dataset_ood = None, apply_tta: bool = False, fold: int = 0, torch_device: str="cuda", mode: str="async", overwrite: bool = False, keep_alive: bool = False):
    
    device = torch.device(torch_device)

    run_mode = "read-only" if mode == "debug" else mode

    if run_id is not None:
        # resume the original run (id passed as parameter)
        run = neptune.init_run(project=PROJECT_NAME, with_id=run_id, mode=run_mode)

        if run.exists("config/fold_0_id") and run["config/fold_0_id"] != run_id:
            raise ValueError("The specified run appears to be a fold, not the initial run. Therefore, a model version should already exist!\n\
                             To evaluate a fold, specify the model version and the fold number instead.")

        neptune_model = model_config.get_neptune_model_name() if model_version is None else model_version
        mid = open_model_version(neptune_model, neptune_project=PROJECT_NAME, mode=mode, strict=0)
    elif model_version is not None:
        mid = open_model_version(model_version, neptune_project=PROJECT_NAME, mode=mode, strict=1)

        if fold == 0:
            run_id = get_param_cond(mid, "run/id")
        else:
            run_id = get_param_if_exists(mid, f"run/id_fold_{fold}", na_val=False)
            if not run_id:
                raise ValueError(f"Model version '{model_version}' does not have fold {fold}! Field 'run/id_fold_{fold}' does not exist.")

        run = neptune.init_run(project=PROJECT_NAME, with_id=run_id, mode=run_mode)

    # load the config from the run
    config, model_type = create_config_from_run(run, device, mode)
    model_config = model_type(config, use_random_search=False)
    model_config.run_id = run_id

    model_config.current_fold = fold
    if fold > 0:
        if do_eval_default and mid.exists(f"validation_fold_{fold}") and not overwrite:
            raise ValueError(f"Tried to overwrite the default evaluation for existing model version {model_version} (fold {fold})! Remove flag '-d' for an existing model version or set '-w' flag.")
        if dataset_ood and mid.exists(f"testing_fold_{fold}/ood/{dataset_ood}") and not overwrite:
            raise ValueError(f"Tried to overwrite the ood evaluation ({dataset_ood}) for existing model version {model_version} (fold {fold})! Remove arg '-o' or set '-w' flag.")

        valid_folds = get_valid_folds(model_config)
        model_config.valid_indices = valid_folds[fold-1]
        model_config.use_index_splitter = True
    elif do_eval_default and mid.exists("validation") and not overwrite:
        raise ValueError(f"Tried to overwrite the default evaluation for existing model version {model_version}! Remove flag '-d' for an existing model version  or set '-w' flag.")
    elif dataset_ood and mid.exists(f"testing/ood/{dataset_ood}") and not overwrite:
        raise ValueError(f"Tried to overwrite the ood evaluation ({dataset_ood}) for existing model version {model_version}! Remove arg '-o' or set '-w' flag.")
        
    # prepare the experiment context
    model_config.prepare_generators()

    dls, dl_test = model_config.prepare_data()

    if apply_tta:
        # tta loader
        raise NotImplementedError("TTA is not implemented yet!")
    else:
        dl_tta = None
    
    if dataset_ood:
        # ood loader
        ood_path = Path(DATASET_DIRECTORY)/dataset_ood
        items_ood = model_config.data_block.get_items(ood_path/"train")
        dl_ood = dls.test_dl(items_ood, rm_type_tfms=None, num_workers=0, with_labels=True, with_input=False)
    else:
        dl_ood = None

    model = model_config.create_model(dls)

    loss_func = model_config.get_loss_func()
    loss_func.is_logits = True  # we need logits to optimize the temp

    # note: we don't instantiate the optimizer since we're not going to update the weights
    #       we also skip the optimizer callback (e.g., f_WGD)


    # note: we skip the neptunecallback and the early stopping callback, as we are not training here
    callbacks = model_config.get_callbacks(train=False)

    # instantiate the learner
    learn = Learner(dls=dls, model=model, loss_func=loss_func, cbs=callbacks, path=MODEL_PATH)

    # load the model
    filename = model_config.get_filename()
    learn.load(filename, device=device, strict=True)

    # prepare additional fields to log in a new model_version
    # which the program may not be able to infer
    if model_version is None:
        log_fields = {
            "data/validation/name": model_config.dataset_name,
            "data/validation/size": learn.dls[1].n,
            "data/testing/name": model_config.dataset_test,
            "data/testing/size": dl_test.n,
        }
    else:
        log_fields = None

    _log_scale = torch.log2(torch.tensor(10, dtype=torch.float))

    _unct_cbs = process_callbacks(model_config.evaluation_callbacks, [], _log_scale)
    _unct_cbs.extend(model_config.get_additional_eval_callbacks())

    if do_eval_default:
        eval_model(
            learn=learn,
            neptune_model=mid,
            neptune_project=PROJECT_NAME,
            run=run,
            dl_valid=learn.dls[1],
            dl_test=dl_test,
            neptune_fit_idx=0,
            reduce_preds=model_config.should_reduce_preds(),
            do_validate=True,
            do_test=True,
            neptune_base_namespace="",
            log_additional=log_fields,
            uncertainty_callbacks=_unct_cbs,
            mode=mode,
            fold=model_config.current_fold,
            inner=False,
            keep_alive=True,
        )

    if dl_tta is not None:
        raise NotImplementedError("TTA is not implemented yet!")

    if dl_ood is not None:        
        eval_model_ood(
            learn=learn,
            model_version_id=mid,
            neptune_project=PROJECT_NAME,
            dl_ood=dl_ood,
            ood_name=dataset_ood,
            uncertainty_callbacks=_unct_cbs,
            reduce_preds=model_config.should_reduce_preds(),
            mode=mode,
            fold=model_config.current_fold,
            keep_alive=True
        )

    run.stop()
    if not keep_alive:
        mid.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a run from scratch by specifying a run_id or add evaluation to an existing model version.\n\
                                                If you want to evaluate a fold, specify the model version and the fold id!")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-r", "--run_id", required=False, default=None, help="string with the run id to evaluate")
    group.add_argument("-m", "--model_version_id", required=False, default=None, help="id of an existing model version to extend")
    parser.add_argument("-d", "--default", required=False, default=False, action="store_true", help="Enable default evaluation")
    parser.add_argument("-o", "--ood_set", required=False, default=None, choices=DATASETS.__args__, help="enable additional evaluation on the specified OOD")
    parser.add_argument("-t", "--apply_tta", required=False, default=False, action="store_true", help="Enable additional evaluation using test-time-augmentation")
    parser.add_argument("-f", "--fold_idx", required=False, default=0, type=int, help="The index of the fold to evaluate. Defaults to 0.")
    parser.add_argument("-c", "--cuda_device", required=False, default="cuda:0", choices=["cpu", "cuda", "cuda:0", "cuda:1"], help="cuda device or cpu")
    parser.add_argument("-n", "--neptune_mode", required=False, default="async", choices=["async", "debug", "read-only"], help="neptune mode used for the run and model version")
    parser.add_argument("-w", "--overwrite", required=False, default=False, action="store_true", help="Enable overwriting existing evaluations for the current fold")

    args = parser.parse_args()

    eval_run(
        run_id=args.run_id,
        model_version=args.model_version_id,
        do_eval_default=args.default,
        dataset_ood=args.ood_set,
        apply_tta=args.apply_tta,
        fold=args.fold_idx,
        torch_device=args.cuda_device,
        mode=args.neptune_mode,
        overwrite=args.overwrite,
        keep_alive=False,
    )
