import argparse

from fastai.vision.all import *

import sys
sys.path.append("./")
import argparse

from src.run_utils import open_model_version, get_fold_metrics, get_ood_names_from_model_version
from src.neptune_tracking import track_cross_valid
from src.constants import PROJECT_NAME

def eval_folds(model_version: str=None, do_eval_default: bool = True, do_eval_ood: bool = False, do_eval_tta: bool = False, mode: str="async", overwrite: bool = False):

    mid = open_model_version(model_version, neptune_project=PROJECT_NAME, mode=mode, strict=1)

    if do_eval_ood:
        ood_names = get_ood_names_from_model_version(mid)
        if len(ood_names) == 0:
            raise ValueError(f"Did not find any OOD names in model version!")
    else:
        ood_names = []

    fold_metrics = get_fold_metrics(eval_default=do_eval_default, 
                                    eval_additional=do_eval_default, 
                                    eval_ood=do_eval_ood, 
                                    eval_tta=do_eval_tta,
                                    ood_names=ood_names)

    track_cross_valid(mid, fold_metrics, overwrite=overwrite)

    mid.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the mean and std for existing fold evaluations")

    parser.add_argument("-m", "--model_version_id", required=False, default=None, help="id of an existing model version to extend")
    parser.add_argument("-d", "--default", required=False, default=False, action="store_true", help="Enable fold statistics on default evaluation")
    parser.add_argument("-o", "--apply_ood", required=False, default=False, action="store_true", help="Enable fold statistics on OOD evaluation")
    parser.add_argument("-t", "--apply_tta", required=False, default=False, action="store_true", help="Enable fold statistics on test-time-augmentation evaluation")
    parser.add_argument("-n", "--neptune_mode", required=False, default="async", choices=["async", "debug", "read-only"], help="neptune mode used for the run and model version")
    parser.add_argument("-w", "--overwrite", required=False, default=False, action="store_true", help="Enable overwriting existing fold statistics")

    args = parser.parse_args()

    eval_folds(
        model_version=args.model_version_id,
        do_eval_default=args.default,
        do_eval_ood=args.apply_ood,
        do_eval_tta=args.apply_tta,
        mode=args.neptune_mode,
        overwrite=args.overwrite,
    )
