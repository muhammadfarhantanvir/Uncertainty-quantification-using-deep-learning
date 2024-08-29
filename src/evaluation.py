from fastai.vision.all import *

from src.uncertainty_utils import get_unct_bins
from src.net_utils import get_param_cond
from src.uncertainty_utils import get_cond_acc_at_rejection_rate
from src.run_utils import open_model_version
from src.neptune_tracking import track_model_version_from_run, track_accuracy_rejection_curves, track_confusion_matrix, \
                                    track_epoch_accuracy, track_accuracy_rejection_curve_base
from neptune.metadata_containers import ModelVersion, Run

import torch
import torch.nn.functional as F
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score
from sklearn.calibration import calibration_curve
from fastai.learner import Learner
from neptune.types import File
from typing import Optional
from sklearn.preprocessing import label_binarize

EPSILON = 1e-15

def calculate_additional_metrics(y_true, y_pred, y_proba):
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    
    kappa = cohen_kappa_score(y_true, y_pred)
        
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'cohen_kappa': kappa,
    }

def plot_calibration_curve(y_true, y_prob, n_bins=10, class_names=None):
    n_classes = y_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_prob[:, i], n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker='o', label=class_names[i])
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curves (One-vs-Rest)')
    ax.legend(loc='best')
    
    return fig

def find_optimal_temperature(logits, labels):
    """ finds the optimal temperature by minimizing the -ve log likelihood over a temperature value. 
    The temperature default value is set to 1.0 with the upper and lower limit being 0.01 and 10.0
    """
    def nll_loss(temp):
        temp = torch.tensor(temp, requires_grad=True, device=logits.device)
        scaled_logits = logits / temp
        loss = F.cross_entropy(scaled_logits, labels)
        return loss.item()
    
    # maximize the log-likelihood <> minimize the negative log-likelihood
    optimal_temp = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
    return optimal_temp.x[0]

def apply_temperature_scaling(logits, temperature):
    return F.softmax(logits / temperature, dim=-1)

def calibrated_log_likelihood(scaled_preds, labels, is_logits: bool=False):
    """Calculates the log-likelihood of the scaled predictions from temperature scaling."""
    if is_logits:
        log_likelihood = -F.cross_entropy(scaled_preds, labels, reduction='mean').item()
    else:
        log_likelihood = -F.nll_loss(torch.log(scaled_preds+EPSILON), labels, reduction='mean').item()
    return log_likelihood

def soft_binning(confidences, num_bins, temperature):
    bin_centers = np.linspace(0, 1, num_bins)
    bin_memberships = np.exp(-((confidences[:, None] - bin_centers) ** 2) / temperature)
    bin_memberships /= bin_memberships.sum(axis=1, keepdims=True)
    return bin_memberships

def compute_sbece(confidences, accuracies, num_bins=15, temperature=1.0, p=2):
    """
    Compute the Soft-Binned Expected Calibration Error (SB-ECE).

    Args:
        confidences (numpy.ndarray): Array of predicted confidences.
        accuracies (numpy.ndarray): Array of accuracies (0 or 1).
        num_bins (int): Number of bins to use.
        p (int): Power to which the differences are raised (default is 2 for squared error).

    Returns:
        float: The computed SB-ECE.
    """
    bin_memberships = soft_binning(confidences, num_bins, temperature)
    bin_acc = np.zeros(num_bins)
    bin_conf = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for i in range(num_bins):
        bin_sizes[i] = np.sum(bin_memberships[:, i])
        if bin_sizes[i] > 0:
            bin_acc[i] = np.sum(bin_memberships[:, i] * accuracies) / bin_sizes[i]
            bin_conf[i] = np.sum(bin_memberships[:, i] * confidences) / bin_sizes[i]

    sbece = np.sum(bin_sizes * np.abs(bin_acc - bin_conf) ** p) / np.sum(bin_sizes)
    return sbece ** (1 / p)

def compute_ece(confidences, accuracies, num_bins=15):
    """
    Compute the Expected Calibration Error (ECE).

    Args:
        confidences (numpy.ndarray): Array of predicted confidences.
        accuracies (numpy.ndarray): Array of accuracies (0 or 1).
        num_bins (int): Number of bins to use.

    Returns:
        float: The computed ECE.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure indices are within valid range

    bin_acc = np.zeros(num_bins)
    bin_conf = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for b in range(num_bins):
        bin_mask = bin_indices == b
        bin_sizes[b] = np.sum(bin_mask)
        if bin_sizes[b] > 0:
            bin_acc[b] = np.mean(accuracies[bin_mask])
            bin_conf[b] = np.mean(confidences[bin_mask])

    # Compute ECE
    ece = 0
    for b in range(num_bins):
        if bin_sizes[b] > 0:
            ece += (bin_sizes[b] * np.abs(bin_acc[b] - bin_conf[b]))

    ece /= np.sum(bin_sizes)  # Mean of the values

    return ece

def optimize_temperature_sbece(logits, labels, dec_preds, num_bins=15, initial_temp=1.0):
    accuracies = (dec_preds == labels).float()

    def sbece_loss(temp):
        scaled_preds = apply_temperature_scaling(logits, temp)
        confidences = torch.max(scaled_preds, dim=1)[0]
        return compute_sbece(confidences.cpu().numpy(), accuracies.cpu().numpy())

    result = minimize(sbece_loss, initial_temp, bounds=[(0.01, 10.0)])
    return result.x[0]

def eval_uncertainty(model_version: ModelVersion, prefix: str, uncts: torch.Tensor, targs: torch.Tensor, preds: torch.Tensor, rejection_rate: float=0.2, unct_name: str="uncertainty", vocab: list = None, **kwargs):

    # track the conditional accuracy (for all classes) 
    # for a given rejection rate
    cond_acc = get_cond_acc_at_rejection_rate(rejection_rate, uncts, preds, targs)
    model_version[f"{prefix}/conditional_accuracy_{int(rejection_rate*100)}"] = cond_acc
    
    x_num = len(get_unct_bins(uncts))

    _ = track_accuracy_rejection_curves(model_version,
                                        prefix,
                                        x_num,
                                        uncts,
                                        targs,
                                        preds,
                                        title_suffix=f"({unct_name}; all classes)",
                                        vocab=vocab,
                                        **kwargs)

    _ = track_accuracy_rejection_curve_base(model_version,
                                            prefix,
                                            x_num,
                                            uncts,
                                            targs,
                                            preds,
                                            title_suffix=f"({unct_name}; all classes)",
                                            **kwargs)                               

def eval_model(learn: Learner, neptune_model: str, neptune_project: str, run: Run, dl_valid, dl_test, neptune_fit_idx: int, uncertainty_callbacks: list, reduce_preds: bool = True, do_validate: bool = True, do_test: bool = True, neptune_base_namespace: str = "", log_additional: dict = None, mode: Optional[str] = None, fold: int = 0, inner=True, keep_alive: bool=False):
    """
    Evaluates a model (single or ensemble).

    Args:
        learn: The FastAI Learner object.
        neptune_model: Name of the model or an existing model version for logging.
        neptune_project: Neptune project name.
        run: Neptune run object.
        dl_valid: Validation DataLoader.
        dl_test: Test DataLoader.
        uncertainty_callbacks: List of callbacks to calculate uncertainty metrics.
        reduce_preds: Whether to reduce predictions.
        do_validate: Whether to perform validation.
        do_test: Whether to perform testing.
        neptune_base_namespace: Base namespace for Neptune logging.
        log_additional: Additional fields to log.
    """

    # log the model version
    model_version = open_model_version(neptune_model=neptune_model, neptune_project=neptune_project, mode=mode, strict=fold)
    
    if run is not None:
        if fold == 0:
            track_model_version_from_run(model_version, run, track_ensemble=reduce_preds)
        else:
            model_version[f"run/id_fold_{fold}"] = run["sys/id"].fetch()
            model_version[f"run/url_fold_{fold}"] = run.get_url()

    if log_additional is not None:
        for k, v in log_additional.items():
            model_version[k] = v

    if uncertainty_callbacks is None or not len(uncertainty_callbacks):
        raise ValueError(f"List of uncertainty callbacks cannot be empty!")

    if do_validate:
        if fold == 0:
            model_version["data/validation/vocab"] = str(dl_valid.vocab)
            model_version["data/validation/num_classes"] = len(dl_valid.vocab)
        
        val_prefix = f"validation_fold_{fold}" if fold > 0 else "validation"

        logits_val, targs_val, dec_preds_val, prefix = track_epoch_accuracy(
            learn,
            dl_valid,
            val_prefix,
            neptune_fit_idx,
            run=run,
            model_version=model_version,
            reduce_preds=reduce_preds,
            base_namespace=neptune_base_namespace,
            cbs=uncertainty_callbacks,
            inner=inner,
        )

        # Find the optimal temperature
        temperature = find_optimal_temperature(logits_val, targs_val)
        model_version[f"{val_prefix}/temperature"] = temperature

        # Apply temperature scaling to validation predictions
        scaled_preds_val = apply_temperature_scaling(logits_val, temperature)

        # Calculate the calibrated log-likelihood for val
        log_likelihood = calibrated_log_likelihood(scaled_preds_val, targs_val)
        model_version[f"{val_prefix}/calibrated_log_likelihood"] = log_likelihood

        # Compute accuracies using decoded predictions
        accuracies_val = (dec_preds_val == targs_val).float()

        # Compute ECE for validation set
        confidences_val = torch.max(scaled_preds_val, dim=1)[0]
        ece_val = compute_ece(confidences_val.cpu().numpy(), accuracies_val.cpu().numpy())
        model_version[f"{val_prefix}/ece"] = ece_val

        # Compute SB-ECE for validation set
        sbece_val = compute_sbece(confidences_val.cpu().numpy(), accuracies_val.cpu().numpy())
        model_version[f"{val_prefix}/sbece"] = ece_val

        # Optimize temperature for SB-ECE using accuracies
        temperature_sbece = optimize_temperature_sbece(logits_val, targs_val, dec_preds_val)
        model_version[f"{val_prefix}/temperature_sbece"] = temperature_sbece

        # Apply TS-SB-ECE to validation predictions
        scaled_preds_val_sbece = apply_temperature_scaling(logits_val, temperature_sbece)

        # Compute calibrated log-likelihood with TS-SB-ECE
        log_likelihood_sbece = calibrated_log_likelihood(scaled_preds_val_sbece, targs_val)
        model_version[f"{val_prefix}/calibrated_log_likelihood_sbece"] = log_likelihood_sbece

        # Compute SB-ECE with TS-SB-ECE
        confidences_val_sbece = torch.max(scaled_preds_val_sbece, dim=1)[0]
        sbece_val_sbece = compute_sbece(confidences_val_sbece.cpu().numpy(), accuracies_val.cpu().numpy())
        model_version[f"{val_prefix}/sbece_sbece"] = sbece_val_sbece

        # Compute ECE with TS-SB-ECE
        ece_val_sbece = compute_ece(confidences_val_sbece.cpu().numpy(), accuracies_val.cpu().numpy())
        model_version[f"{val_prefix}/ece_sbece"] = ece_val_sbece

        # Calculate additional metrics for this fold
        additional_metrics = calculate_additional_metrics(targs_val.numpy(), dec_preds_val.numpy(), scaled_preds_val.numpy())
        
        for metric_name, metric_value in additional_metrics.items():
            if metric_name != 'confusion_matrix':
                model_version[f"{val_prefix}/{metric_name}"] = metric_value

        # Continue with evaluations using scaled predictions
        if fold == 0:
            _ = track_confusion_matrix(model_version,
                                    targs_val,
                                    dec_preds_val,
                                    val_prefix, 
                                    title="class confusions",
                                    name="class confusions (validation)",
                                    description="unmodified classes on validation set confusions.",
                                    vocab=dl_valid.vocab,
                                    normalize="true")

        # evaluate the uncertainty values
        has_unct = False
        unct_attrs = {"total_unct": "total uncertainty",
                      "aleatoric_unct": "aleatoric uncertainty",
                      "epistemic_unct": "epistemic uncertainty"}

        for attr, name in unct_attrs.items():
            if hasattr(learn, attr):
                uncts = getattr(learn, attr)
                has_unct = True

                eval_uncertainty(
                    model_version=model_version,
                    prefix=f"{val_prefix}/{attr}",
                    uncts=uncts,
                    targs=targs_val,
                    preds=scaled_preds_val,
                    unct_name=name,
                    vocab=dl_valid.vocab,
                    track_arc_values=True,
                )

        cal_curve_fig = plot_calibration_curve(targs_val.numpy(), scaled_preds_val.numpy(), class_names=dl_valid.vocab)
        model_version[f"{val_prefix}/calibration_curve"].upload(File.as_image(cal_curve_fig))
        plt.close(cal_curve_fig)

        if not has_unct:
            raise AttributeError(f"No uncertainty values found in the learner! At least one of the following needs to be defined: {[k for k in unct_attrs.keys()]}")

    if do_test:
        if fold == 0:
            model_version["data/testing/vocab"] = str(dl_test.vocab)
            model_version["data/testing/num_classes"] = len(dl_test.vocab)

        tst_prefix = f"testing_fold_{fold}" if fold > 0 else "testing"

        logits_tst, targs_tst, dec_preds_tst, prefix = track_epoch_accuracy(
            learn,
            dl_test,
            tst_prefix,
            neptune_fit_idx + 1 if do_validate else neptune_fit_idx,
            run=run,
            model_version=model_version,
            reduce_preds=reduce_preds,
            base_namespace=neptune_base_namespace,
            cbs=uncertainty_callbacks,
            inner=inner,
        )

        # Apply the same temperature scaling to test predictions
        if not do_validate:
            temperature = find_optimal_temperature(logits_tst, targs_tst)
        model_version[f"{tst_prefix}/temperature"] = temperature

        scaled_preds_tst = apply_temperature_scaling(logits_tst, temperature)

        # Calculate the calibrated log-likelihood for test
        log_likelihood = calibrated_log_likelihood(scaled_preds_tst, targs_tst)
        model_version[f"{tst_prefix}/calibrated_log_likelihood"] = log_likelihood

        # Now that we have scaled_preds_tst, we can calculate additional metrics
        additional_metrics = calculate_additional_metrics(targs_tst.numpy(), dec_preds_tst.numpy(), scaled_preds_tst.numpy())
        
        for metric_name, metric_value in additional_metrics.items():
            if metric_name != 'confusion_matrix':
                model_version[f"{tst_prefix}/{metric_name}"] = metric_value
        
        if fold == 0:
            _ = track_confusion_matrix(model_version,
                                        targs_tst,
                                        dec_preds_tst,
                                        tst_prefix, 
                                        title="class confusions",
                                        name="class confusions (testing)",
                                        description="unmodified classes on testing set confusions.",
                                        vocab=dl_test.vocab,
                                        normalize="true")

        confidences_tst = torch.max(scaled_preds_tst, dim=1)[0]
        accuracies_tst = (dec_preds_tst == targs_tst).float()

        # Compute ECE for test set
        ece_tst = compute_ece(confidences_tst.cpu().numpy(), accuracies_tst.cpu().numpy())
        model_version[f"{tst_prefix}/ece"] = ece_tst

        # Compute SB-ECE for test set
        ece_tst = compute_sbece(confidences_tst.cpu().numpy(), accuracies_tst.cpu().numpy())
        model_version[f"{tst_prefix}/sbece"] = ece_tst

        # Optimize temperature for SB-ECE if not already optimized
        if not do_validate:
            temperature_sbece = optimize_temperature_sbece(logits_tst, targs_tst, dec_preds_tst)
        model_version[f"{tst_prefix}/temperature_sbece"] = temperature_sbece

        # Apply TS-SB-ECE to test predictions
        scaled_preds_tst_sbece = apply_temperature_scaling(logits_tst, temperature_sbece)

        # Compute calibrated log-likelihood with TS-SB-ECE
        log_likelihood_sbece = calibrated_log_likelihood(scaled_preds_tst_sbece, targs_tst)
        model_version[f"{tst_prefix}/calibrated_log_likelihood_sbece"] = log_likelihood_sbece

        # Compute SB-ECE with TS-SB-ECE
        confidences_tst_sbece = torch.max(scaled_preds_tst_sbece, dim=1)[0]
        sbece_tst_sbece = compute_sbece(confidences_tst_sbece.cpu().numpy(), accuracies_tst.cpu().numpy())
        model_version[f"{tst_prefix}/sbece_sbece"] = sbece_tst_sbece

        # Compute ECE with TS-SB-ECE
        ece_tst_sbece = compute_ece(confidences_tst_sbece.cpu().numpy(), accuracies_tst.cpu().numpy())
        model_version[f"{tst_prefix}/ece_sbece"] = ece_tst_sbece
       
        # evaluate the uncertainty values
        has_unct = False
        unct_attrs = {"total_unct": "total uncertainty",
                      "aleatoric_unct": "aleatoric uncertainty",
                      "epistemic_unct": "epistemic uncertainty"}

        for attr, name in unct_attrs.items():
            if hasattr(learn, attr):
                uncts = getattr(learn, attr)
                has_unct = True

                eval_uncertainty(
                    model_version=model_version,
                    prefix=f"{tst_prefix}/{attr}",
                    uncts=uncts,
                    targs=targs_tst,
                    preds=scaled_preds_tst,
                    unct_name=name,
                    vocab=dl_test.vocab,
                    cached_arc_prefix=f"{val_prefix}/{attr}",
                    plot_cached_arc=True,
                    reject_on_cached=True,
                )

        cal_curve_fig = plot_calibration_curve(targs_tst.numpy(), scaled_preds_tst.numpy(), class_names=dl_test.vocab)
        model_version[f"{tst_prefix}/calibration_curve"].upload(File.as_image(cal_curve_fig))
        plt.close(cal_curve_fig)

        if not has_unct:
            raise AttributeError(f"No uncertainty values found in the learner! At least one of the following needs to be defined: {[k for k in unct_attrs.keys()]}")

    # Fetch the model version ID
    model_version_id = model_version["sys/id"].fetch()

    # stop the model version
    if not keep_alive:
        model_version.stop()

    return model_version_id


def eval_model_ood(learn: Learner, model_version_id: str, neptune_project: str, dl_ood, ood_name: str, uncertainty_callbacks: list, reduce_preds: bool = True, mode: Optional[str] = None, fold: int=0, keep_alive: bool=False):
    model_version = open_model_version(neptune_model=model_version_id, neptune_project=neptune_project, mode=mode, strict=fold)

    if uncertainty_callbacks is None or not len(uncertainty_callbacks):
        raise ValueError(f"List of uncertainty callbacks cannot be empty!")

    # log the details of the ood set
    data_prefix = f"data/testing/ood/{ood_name}"
    if not model_version.exists(data_prefix):
        model_version[f"{data_prefix}/name"] = ood_name
        model_version[f"{data_prefix}/size"] = dl_ood.n
        model_version[f"{data_prefix}/vocab"] = str(dl_ood.vocab)
        model_version[f"{data_prefix}/num_classes"] = len(dl_ood.vocab)

    # predict on all samples of the OOD set
    ood_prefix = f"testing/ood/{ood_name}" if fold == 0 else f"testing_fold_{fold}/ood/{ood_name}"
    valid_prefix = "validation" if fold == 0 else f"validation_fold_{fold}"

    logits, targs, dec_preds, _ = track_epoch_accuracy(
        learn,
        dl_ood,
        target=ood_prefix,
        fit_index=0,
        run=None,
        model_version=model_version,
        reduce_preds=reduce_preds,
        cbs=uncertainty_callbacks,
    )

    # Apply temperature scaling to validation predictions
    valid_temp = get_param_cond(model_version, f"{valid_prefix}/temperature", convert_type=float)
    scaled_preds = apply_temperature_scaling(logits, valid_temp)

    # Calculate the calibrated log-likelihood for val
    log_likelihood = calibrated_log_likelihood(scaled_preds, targs)
    model_version[f"{ood_prefix}/calibrated_log_likelihood"] = log_likelihood

    # Compute SB-ECE for OOD set
    confidences_ood = torch.max(scaled_preds, dim=1)[0]
    accuracies_ood = (dec_preds == targs).float()
    sbece_ood = compute_sbece(confidences_ood.cpu().numpy(), accuracies_ood.cpu().numpy())
    model_version[f"{ood_prefix}/sbece"] = sbece_ood

    # Compute ECE for OOD set
    ece_ood = compute_ece(confidences_ood.cpu().numpy(), accuracies_ood.cpu().numpy())
    model_version[f"{ood_prefix}/ece"] = ece_ood

    # Optimize temperature for SB-ECE
    temperature_sbece_ood = get_param_cond(model_version, f"{valid_prefix}/temperature_sbece", convert_type=float)
    model_version[f"{ood_prefix}/temperature_sbece"] = temperature_sbece_ood

    # Apply TS-SB-ECE to OOD predictions
    scaled_preds_sbece = apply_temperature_scaling(logits, temperature_sbece_ood)

    # Compute calibrated log-likelihood with TS-SB-ECE
    log_likelihood_sbece = calibrated_log_likelihood(scaled_preds_sbece, targs)
    model_version[f"{ood_prefix}/calibrated_log_likelihood_sbece"] = log_likelihood_sbece

    # Compute SB-ECE with TS-SB-ECE
    confidences_ood_sbece = torch.max(scaled_preds_sbece, dim=1)[0]
    accuracies_ood_sbece = (dec_preds == targs).float()
    sbece_ood_sbece = compute_sbece(confidences_ood_sbece.cpu().numpy(), accuracies_ood_sbece.cpu().numpy())
    model_version[f"{ood_prefix}/sbece_sbece"] = sbece_ood_sbece

    # Compute ECE with TS-SB-ECE
    ece_ood_sbece = compute_ece(confidences_ood_sbece.cpu().numpy(), accuracies_ood_sbece.cpu().numpy())
    model_version[f"{ood_prefix}/ece_sbece"] = ece_ood_sbece

    # Continue with evaluations using scaled predictions
    if fold == 0:
        _ = track_confusion_matrix(model_version,
                                    targs,
                                    dec_preds,
                                    prefix=ood_prefix, 
                                    title="class confusions",
                                    name="class confusions (OOD)",
                                    description="unmodified classes on OOD set confusions.",
                                    vocab=dl_ood.vocab,
                                    normalize="true")

    # evaluate the uncertainty values
    has_unct = False
    unct_attrs = {"total_unct": "total uncertainty",
                    "aleatoric_unct": "aleatoric uncertainty",
                    "epistemic_unct": "epistemic uncertainty"}

    for attr, name in unct_attrs.items():
        if hasattr(learn, attr):
            uncts = getattr(learn, attr)
            has_unct = True

            # add kwargs for plotting the OOD ARC with respect to the validation ARC
            eval_uncertainty(
                model_version=model_version,
                prefix=f"{ood_prefix}/{attr}",
                uncts=uncts,
                targs=targs,
                preds=scaled_preds,
                unct_name=name,
                vocab=dl_ood.vocab,
                cached_arc_prefix=f"{valid_prefix}/{attr}",
                plot_cached_arc=True,
                reject_on_cached=True,
            )

    if not has_unct:
        raise AttributeError(f"No uncertainty values found in the learner! At least one of the following needs to be defined: {[k for k in unct_attrs.keys()]}")

    if not keep_alive:
        model_version.stop()

    return model_version_id

