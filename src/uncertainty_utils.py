import numpy as np
import torch


def get_auc_score(y, x, normalize=True):
    map_no_nan = ~y.isnan()
    if map_no_nan.sum() == 0:
        return 0.0
    auc = np.trapz(y=y[map_no_nan], x=x[map_no_nan])
    if normalize:
        return auc / (x[map_no_nan][-1] - x[map_no_nan][0])
    return auc


def get_unct_bins(unct):
    return np.histogram_bin_edges(unct.cpu(), bins="stone", range=(0.0, 1.01*unct.cpu().max().item()))


def get_unct_tresholds(target_class, unct_bins, pred_uncertainty, targets):
    """
    Get uncertainty threshold values for various analysis functions.
    Sorts the predictions in ascending order by their uncertainties
    and bins them with an equal number of samples per bin. 

    If the number of samples is not divisible by `bin_num` without a remainder, 
    the first bins will contain one extra sample until the remainder has been
    distributed fully. 

    params:
        target_class: is used to obtain the target class map
        unct_bins: tensor with bin thresholds or number of equal-mass bins to use on the uncertainty values. If None, an appropriate number of bins is attempted to be determined.
        pred_uncertainty: predictive uncertainies for the samples
        targets: target values to compare `targets` against
    """
    class_map = targets == target_class

    unct_masked = pred_uncertainty[class_map]
    class_num = unct_masked.shape[0]

    if isinstance(unct_bins, (torch.Tensor, np.ndarray, list, tuple)):
        if isinstance(unct_bins, torch.Tensor) and unct_bins.dim() == 0:
            unct_bins = unct_bins.item()
        elif len(unct_bins) == 1:
            unct_bins = unct_bins[0].item() if isinstance(unct_bins[0], torch.Tensor) else unct_bins[0]
        else:
            uncty_vals = unct_bins
            bin_num = len(uncty_vals)

    if isinstance(unct_bins, (int, np.int32, np.int64)):
        bin_num = unct_bins
        _, idxs = torch.sort(unct_masked)
        bin_mass, rem = np.divmod(class_num, bin_num-1)
        bin_end = 0

        uncty_vals = torch.zeros(bin_num)
        
        for i in range(1,bin_num):
            bin_end += bin_mass
            if i <= rem and rem > 0:
                bin_end += 1

            bin_idxs = idxs[:bin_end]

            uncty_vals[i] = unct_masked[bin_idxs[-1]] + 1e-4

    elif unct_bins is None:
        uncty_vals = get_unct_bins(unct_masked)
        bin_num = len(uncty_vals)

    return uncty_vals, bin_num


def accuracy_rejection_curve(target_class, pred_class, unct_bins, pred_uncertainty, targets, preds, verbose=False, min_bin_size: int=10):
    """
    Sorts the predictions in ascending order by their uncertainties
    and bins them with an equal number of samples per bin. 

    If the number of samples is not divisible by `bin_num` without a remainder, 
    the first bins will contain one extra sample until the remainder has been
    distributed fully. 

    The conditional accuracy for a rejection rate of 100% is 100% by definition. 
    The conditional accuracy for a rejection rate of 0% is the unconditional class accuracy. 

    params:
        target_class: is used to obtain the target class map
        pred_class: prediction values are compared with this value
        unct_bins: tensor with bin thresholds or number of equal-mass bins to use on the uncertainty values
        pred_uncertainty: predictive uncertainies for the samples
        targets: target values to compare `targets` against
        preds: predicted values, are compared with `pred_class`
        verbose: if true, returns uncertainty threshold values for the obtained binning
    """

    class_map = targets == target_class

    unct_masked = pred_uncertainty[class_map]
    class_num = unct_masked.shape[0]

    uncty_vals, bin_num = get_unct_tresholds(target_class, unct_bins, pred_uncertainty, targets)

    acc = torch.zeros(bin_num)
    rej = torch.zeros(bin_num)

    for i in range(0,bin_num):
        bin_idxs = torch.argwhere(unct_masked < uncty_vals[i]).to(preds.device)

        pred_bin = torch.argmax(preds[class_map,:][bin_idxs,:], -1)

        pred_map = pred_class if isinstance(pred_class, (int, np.intc)) else pred_class[bin_idxs]
        ntr = (pred_bin.float() == pred_map).sum().float()

        acc[i] = ntr / len(bin_idxs) if len(bin_idxs) > min_bin_size else np.nan
        rej[i] = 1 - (len(bin_idxs) / class_num) if class_num > 0 else 1.0


    if verbose:
        return rej, acc, uncty_vals

    return rej, acc

def get_cond_acc_at_rejection_rate(rejection_rate, uncts, preds, targs):
    _ut = torch.quantile(uncts, torch.tensor(1-rejection_rate, device=uncts.device))
    bin_idxs = torch.argwhere(uncts < _ut).to(preds.device)
    pred_accepted = torch.argmax(preds[bin_idxs,:], -1)
    
    pred_map = targs if isinstance(targs, (int, np.intc)) else targs[bin_idxs]
    ntr = (pred_accepted.float() == pred_map).sum().float()
    
    return ntr / len(bin_idxs) if len(bin_idxs) > 0 else np.nan
