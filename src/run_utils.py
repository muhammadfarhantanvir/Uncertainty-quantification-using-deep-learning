from fastai.vision.all import *
import neptune
import ast

import neptune.exceptions
from src.callback import calc_dirichlet_uncertainty, calc_sngp_uncertainty, calc_validation_total_entropy
    
def open_model_version(neptune_model, neptune_project, mode, strict:int=1):
    if isinstance(neptune_model, neptune.ModelVersion):
        return neptune_model
    
    try:
        model_version = neptune.init_model_version(with_id=neptune_model, project=neptune_project, mode=mode)
    except neptune.exceptions.ModelVersionNotFound:
        if strict > 0:
            raise ValueError(f"Existing model version '{neptune_model}' not found!")
        
        try:
            model_version = neptune.init_model_version(model=neptune_model, project=neptune_project, mode=mode)
        except neptune.exceptions.ModelNotFound:
            raise ValueError(f"'{neptune_model}' provided as 'neptune_model' is neither an existing model, nor an existing model version!")
        
    return model_version

def get_source_file_paths_from_run(run, field) -> list[str]:
    def fetch_fileset(sub_path=None):
        files = []
        sub_path_str = f"{sub_path}/" if sub_path is not None else ""
        for f in run[field].list_fileset_files(path=sub_path):
            if f.file_type == "file":
                files.append(f"{sub_path_str}{f.name}")
            elif f.file_type == "directory":
                files.extend(fetch_fileset(sub_path=f"{sub_path_str}{f.name}"))

        return files
            
    return fetch_fileset()

def is_wd_config(run) -> bool:
    return "AdamW" in run["sys/tags"].fetch() or run["config/optimizer/name"].fetch() == "AdamW"

def process_callbacks(callbacks: List[Dict[str, Any]], results: List[Any],log_scale: Optional[str] = "natural") -> List[Any]:
    for callback in callbacks:
        callback_name = callback['callback_name']
        callback_params = callback.get('callback_params', {})

        if callback_name == 'calc_validation_total_entropy':
            result = calc_validation_total_entropy(log_scale=log_scale, **callback_params)
        elif callback_name == 'calc_sngp_uncertainty':
            result = calc_sngp_uncertainty(**callback_params)
        elif callback_name == 'calc_dirichlet_uncertainty':
            result = calc_dirichlet_uncertainty(**callback_params)
        else:
            print(f"Unknown callback function: {callback_name}")
            result = None

        if result is not None:
            results.append(result)
        return results

def parse_list_params(param_dict):
    for k,v in param_dict.items():
        if isinstance(v, str):
            try:
                param_dict[k] = ast.literal_eval(v)
            except ValueError:
                pass

    return param_dict

def get_fold_values(model_version, prefix:str, metric_name: str, patience: int = 1):
    def inner(field_prefix, fold_idx):
        field = f"{field_prefix}/{metric_name}" if field_prefix is not None else metric_name
        if model_version.exists(field):
            return model_version[field].fetch()
        elif model_version.exists(f"{field}_fold_{fold_idx}"):
            return model_version[f"{field}_fold_{fold_idx}"].fetch()
        else:
            raise neptune.exceptions.MissingFieldException(f'The field "{field}", resp. "{field}_fold_{fold_idx}" was not found.')

    if model_version is None:
        raise ValueError("'model_version' cannot be None!")
    if metric_name == "" or metric_name is None:
        raise ValueError("'metric_name' must be specified!")
    values = []

    # get the value from the original fold
    values.append(inner(prefix, 0))

    # get the values from the consecutive folds, 
    # allowing `patience` many to be missing before aborting
    curr_fold = 1
    num_folds = 1 + patience
    while True:
        try:
            values.append(inner(f"{prefix}_fold_{curr_fold}", curr_fold))
            num_folds += 1
        except KeyError:
            pass

        if curr_fold >= num_folds:
            break

        curr_fold += 1
        
    return values

def get_fold_metrics(eval_default: bool, eval_additional: bool, eval_ood: bool, eval_tta: bool, ood_names: List[str] = []):
    fold_metrics = {}

    if eval_default:
        fold_metrics.update({
            "accuracy": {"allow_missing": False, "apply": ["validation", "testing"]},
            "calibrated_log_likelihood": {"allow_missing": False, "apply": ["validation", "testing"]},
            "ece": {"allow_missing": False, "apply": ["validation", "testing"]},
            "sbece": {"allow_missing": False, "apply": ["validation", "testing"]},
            "calibrated_log_likelihood_sbece": {"allow_missing": False, "apply": ["validation", "testing"]},
            "ece_sbece": {"allow_missing": False, "apply": ["validation", "testing"]},
            "sbece_sbece": {"allow_missing": True, "apply": ["validation", "testing"]},
            "total_unct/conditional_accuracy_20": {"allow_missing": False, "apply": ["validation", "testing"]},
            "aleatoric_unct/conditional_accuracy_20": {"allow_missing": True, "apply": ["validation", "testing"]},
            "epistemic_unct/conditional_accuracy_20": {"allow_missing": True, "apply": ["validation", "testing"]},
        })

    if eval_additional:
        fold_metrics.update({
            "f1_score": {"allow_missing": True, "apply": ["validation", "testing"]},
            "precision": {"allow_missing": True, "apply": ["validation", "testing"]},
            "recall": {"allow_missing": True, "apply": ["validation", "testing"]},
            "roc_auc": {"allow_missing": True, "apply": ["validation", "testing"]},
            "cohen_kappa": {"allow_missing": True, "apply": ["validation", "testing"]},
        })

    if eval_ood:
        if not ood_names or len(ood_names) == 0:
            raise ValueError("ood_names has to contain at least one name!")
        
        for _name in ood_names:
            fold_metrics.update({
                f"ood/{_name}/calibrated_log_likelihood": {"allow_missing": False, "apply": ["testing"]},
                f"ood/{_name}/sbece": {"allow_missing": False, "apply": ["testing"]},
                f"ood/{_name}/ece": {"allow_missing": False, "apply": ["testing"]},
                f"ood/{_name}/calibrated_log_likelihood_sbece": {"allow_missing": False, "apply": ["testing"]},
                f"ood/{_name}/sbece_sbece": {"allow_missing": False, "apply": ["testing"]},
                f"ood/{_name}/ece_sbece": {"allow_missing": False, "apply": ["testing"]},
                f"ood/{_name}/total_unct/conditional_accuracy_20": {"allow_missing": False, "apply": ["testing"]},
                f"ood/{_name}/aleatoric_unct/conditional_accuracy_20": {"allow_missing": True, "apply": ["testing"]},
                f"ood/{_name}/epistemic_unct/conditional_accuracy_20": {"allow_missing": True, "apply": ["testing"]},
            })

    if eval_tta:
        fold_metrics.update({
            "calibrated_log_likelihood_tta": {"allow_missing": False, "apply": ["testing"]},
            "ece_tta": {"allow_missing": False, "apply": ["testing"]},
            "sbece_tta": {"allow_missing": False, "apply": ["testing"]},
        })

    return fold_metrics

def set_missing_field(run, field, value) -> bool:
    if not run.exists(field):
        print(f"Setting field {field} = {value}")
        run[field] = value
        return True
    else:
        print(f"field already set! {field}: {run[field].fetch()}")
        return False

def get_ood_names_from_model_version(model_version) -> List[str]:
    if model_version.exists("data/testing/ood"):
        return [f for f in model_version["data/testing/ood"].fetch()]
    return []
