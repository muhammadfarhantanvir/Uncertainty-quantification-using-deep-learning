from fastai.vision.all import *
import numpy as np
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from neptune.types import File
from sklearn.metrics import ConfusionMatrixDisplay
from src.callback import average_sampled_preds
from src.uncertainty_utils import accuracy_rejection_curve, get_auc_score
from src.run_utils import get_fold_values

def track_model_version_from_run(model_version, run, fine_map=None, excl_dict=None, epochs_from_fit_idx:int=0, track_attr_loss:bool=False, track_sasghmc:bool=False, track_ensemble:bool=True):
    # run
    model_version["run/id"] = run["sys/id"].fetch()
    model_version["run/url"] = run.get_url()
    # model
    if track_ensemble:
        model_version["model/ensemble_size"] = run["config/ensemble_size"].fetch()

    model_version["model/name"] = run["config/model/model_name"].fetch() 
    model_version["model/net"] = run["config/model/net_name"].fetch() 
    model_version["model/num_epochs"] = run[f"metrics/fit_{epochs_from_fit_idx}/n_epoch"].fetch()
    model_version["model/batch_size"] = run["config/batch_size"].fetch()
    if track_sasghmc:
        model_version["model/sim_steps"] = run["config/optimizer/initial_hyperparameters/sim_steps"].fetch()
    # loss
    model_version["loss/func"] = run["config/criterion"].fetch()
    if track_attr_loss:
        model_version["loss/attr_scale"] = run["config/optimizer/initial_hyperparameters/attr_scale"].fetch()

    # optimizer
    model_version["optimizer/func"] = run["config/optimizer/name"].fetch()
    model_version["optimizer/lr"] = run["config/optimizer/initial_hyperparameters/lr"].fetch()
    model_version["optimizer/lr_sched"] = run["config/optimizer/lr_sched/name"].fetch() if run.exists("config/optimizer/lr_sched/name") else "None"
    if track_sasghmc:
        model_version["optimizer/base_C"] = run["config/optimizer/initial_hyperparameters/base_C"].fetch()
        model_version["optimizer/burn_in_epochs"] = run["config/optimizer/initial_hyperparameters/burn_in_epochs"].fetch()
        model_version["optimizer/resample_momentum_batches"] = run["config/optimizer/initial_hyperparameters/resample_momentum_its"].fetch()
        model_version["optimizer/resample_prior_batches"] = run["config/optimizer/initial_hyperparameters/resample_prior_its"].fetch()
    # data
    model_version["data/training/name"] = run["io_files/resources/dataset/name"].fetch()
    model_version["data/training/size"] = run["io_files/resources/dataset/size"].fetch()
    model_version["data/training/vocab"] = run["config/model/vocab/details"].fetch()
    model_version["data/training/num_classes"] = run["config/model/vocab/total"].fetch()
    model_version["data/training/fine_map"] = str(fine_map if fine_map is not None else [])
    model_version["data/training/excl_dict"] = excl_dict if excl_dict is not None else {}



def track_epoch_accuracy(learn, dl, target:str, fit_index:int, run=None, model_version=None, reduce_preds:bool=True, base_namespace:str="", cbs:list=[], inner:bool=False):
    """Tracks the ensembled or single accuracy by performing one validation epoch on the provided data (change reduce_preds to False for single accuracy).
    Can be used to run additional callbacks during this validation epoch, e.g., for computing uncertainty
    The `fit_index` will be incremented by 1, as the internal call to get_preds() in this function
    will increment the counter in the neptune tracking, unless `stop_neptune_tracking`==`True`."""
    
    if reduce_preds:
        has_reduce_preds_cb = len([cb for cb in cbs if isinstance(cb, average_sampled_preds)]) > 0
        if not has_reduce_preds_cb:
            cbs.append(average_sampled_preds())

    with learn.no_bar():
        logits, targs = learn.get_preds(dl=dl, with_input=False, with_decoded=False, act=noop, inner=inner, cbs=cbs, concat_dim=0)

    dec_preds = getcallable(learn.loss_func, 'decodes')(logits)
    acc = (targs.eq(dec_preds).sum() / targs.shape[0]).item()

    prefix = f"{base_namespace}/metrics/fit_{fit_index+1}/{target}/model"

    if run is not None:
        run[f"{prefix}/accuracy"] = acc
    if model_version is not None:
        model_version[f"{target}/accuracy"] = acc

    return logits, targs, dec_preds, prefix


def retrieve_and_plot_cached(model_version, fig, cached_arc_prefix:str, plot_cached_arc:bool, reject_on_cached:bool, bin_num:int, line_kwargs, write_auc_score):
    # plot a previously computed validation ARC as a reference
    if plot_cached_arc and cached_arc_prefix != "":
        if not model_version.exists(f"{cached_arc_prefix}/arc_rej"):
            print(f"Missing field: {cached_arc_prefix}/arc_rej. Unable to plot cached ARC!")
            cached_rej = None
        else:
            cached_rej = model_version[f"{cached_arc_prefix}/arc_rej"].fetch_values()['value'].values
            cached_acc = model_version[f"{cached_arc_prefix}/arc_acc"].fetch_values()['value'].values
            _auc_str = f": {get_auc_score(y=cached_acc, x=cached_rej):.3f}" if write_auc_score else ""
            dotted = dict(dash="dot")
            fig.add_trace(go.Scatter(x=cached_rej, y=cached_acc, **line_kwargs, line=dotted, name=f"{cached_arc_prefix} (ref.){_auc_str}"))
    else:
        cached_rej = None

    # compute the ARC over all classes
    # optionally use the uncertainty thresholds obtained for the cached ARC
    if reject_on_cached and cached_arc_prefix != "":
        if not model_version.exists(f"{cached_arc_prefix}/arc_unc"):
            print(f"Missing field: {cached_arc_prefix}/arc_unc. Unable to reject on cached uncertainty thresholds!")
            unct_thrsh = bin_num
        else:
            unct_thrsh = model_version[f"{cached_arc_prefix}/arc_unc"].fetch_values()
            unct_thrsh = torch.tensor(unct_thrsh['value'].values)
    else:
        unct_thrsh = bin_num

    return cached_rej, unct_thrsh

def track_accuracy_rejection_curves(model_version, prefix:str, bin_num:int, unct_vals, targets, preds, title_suffix:str="", write_auc_score:bool=False, vocab=None, 
                                    track_arc_values: bool=False, cached_arc_prefix: str="", plot_cached_arc: bool=False, reject_on_cached: bool=False):
    """generates accuracy-rejection curves for all  classes in `targets` and records the plot in the provided `model_version`
    under `prefix`/accuracy_rejection_curves"""
    scatter_opts = dict(mode="lines", opacity=0.5, hovertemplate="(%{x:.2f},%{y:.2f})")
    sc_opts = dict(dash="dot")

    fig = go.Figure()
    
    # plot a previously computed validation ARC as a reference
    # optionally use the uncertainty thresholds obtained for the cached ARC from now
    cached_rej, unct_thrsh = retrieve_and_plot_cached(
        model_version=model_version,
        fig=fig,
        cached_arc_prefix=cached_arc_prefix,
        plot_cached_arc=plot_cached_arc,
        reject_on_cached=reject_on_cached,
        bin_num=bin_num,
        line_kwargs=scatter_opts,
        write_auc_score=write_auc_score,
    )
    
    # compute the ARC over all classes
    ret = accuracy_rejection_curve(targets, targets, unct_thrsh, unct_vals, targets, preds, verbose=cached_rej is None)
    if cached_rej is not None:
        arc_rej = cached_rej[-len(ret[1]):][~ret[1].isnan()]
    else:
        arc_rej = ret[0][~ret[1].isnan()]
        unct_thrsh = ret[2]
    arc_acc = ret[1][~ret[1].isnan()]
    _auc_str = f": {get_auc_score(y=arc_acc, x=arc_rej):.3f}" if write_auc_score else ""
    fig.add_trace(go.Scatter(x=arc_rej, y=arc_acc, **scatter_opts, line=sc_opts, name=f"all classes{_auc_str}"))
    
    # compute the ARCs for each class
    for i in targets.unique().numpy().astype(np.intc):
        _, arc_acc_sc = accuracy_rejection_curve(i, i, unct_thrsh, unct_vals, targets, preds)
        _arc_acc = arc_acc_sc[~arc_acc_sc.isnan()]
        _arc_rej = cached_rej[-len(arc_acc_sc):][~arc_acc_sc.isnan()] if cached_rej is not None else ret[0][-len(arc_acc_sc):][~arc_acc_sc.isnan()]

        _sc_str = vocab[i] if vocab is not None else f"class {i}"
        _auc_str = f": {get_auc_score(y=_arc_acc, x=_arc_rej):.3f}" if write_auc_score else ""
        fig.add_trace(go.Scatter(x=_arc_rej, y=_arc_acc, **scatter_opts, name=f"{_sc_str} {_auc_str}"))

    title = "Accuracy-Rejection Curve"
    if title_suffix:
        title = f"{title} {title_suffix}"

    fig.update_layout(title=title,
                        xaxis_title="Rejection Rate (wrt. ref)" if reject_on_cached else "Rejection Rate (wrt. all classes)",
                        yaxis_title="Conditional Accuracy")

    fig.update_yaxes(range=[-0.05, 1.05])

    if model_version is not None:
        model_version[f"{prefix}/accuracy_rejection_curves_all"].upload(fig)

    return fig

def track_confusion_matrix(model_version, targs, dec_preds, prefix:str, title:str="class confusions", name:str="", description:str="", vocab:list=[], **kwargs):
    """Returns the figure with the confusion matrix. If vocab is not empty, adds a list of class descriptions."""
    
    # TODO refactor to use plotly instead
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_axes((1,1,1,1))
    ax.set_title(f'class confusions', fontsize=20)
    ConfusionMatrixDisplay.from_predictions(targs, dec_preds, ax=ax)

    if name == "":
        name = title

    if len(vocab) > 0:
        for i, sc in enumerate(vocab):
            description += f" ({i}: {sc})"

    if model_version is not None:
        model_version[f"{prefix}/confusion_matrix"].upload(fig)
    return fig

def track_accuracy_rejection_curve_base(model_version, prefix:str, bin_num:int, unct_vals, targets, preds, title_suffix:str="", write_auc_score:bool=False, 
                                        track_arc_values: bool=False, cached_arc_prefix: str="", plot_cached_arc: bool=False, reject_on_cached: bool=False):
    """generates accuracy-rejection curves for all  classes in `targets` and records the plot in the provided `model_version`
    under `prefix`/accuracy_rejection_curves.
    If `track_arc_values`==True, writes the values of the ARC to series in the model version.
    If `plot_cached_arc`==True and `cached_arc_prefix` is provided plots an ARC tracked in a previous call to this function.
    If `reject_on_cached` is set additionally, also aligns the new ARC with the cached one by re-using the rejection rates
    to emphasize the same uncertainty tresholds being used"""
    scatter_opts = dict(mode="lines", opacity=0.5, hovertemplate="(%{x:.2f},%{y:.2f})")

    fig = go.Figure()

    # plot a previously computed validation ARC as a reference
    # optionally use the uncertainty thresholds obtained for the cached ARC from now
    cached_rej, unct_thrsh = retrieve_and_plot_cached(
        model_version=model_version,
        fig=fig,
        cached_arc_prefix=cached_arc_prefix,
        plot_cached_arc=plot_cached_arc,
        reject_on_cached=reject_on_cached,
        bin_num=bin_num,
        line_kwargs=scatter_opts,
        write_auc_score=write_auc_score,
    )

    # compute the ARC over all classes
    ret = accuracy_rejection_curve(targets, targets, unct_thrsh, unct_vals, targets, preds, verbose=track_arc_values)
    # Align the new ARC with the one it's uncertainty thresholds are based on 
    # by re-using the rejection rates (same x-values)
    # only used when the reference ARC is plotted as well
    arc_rej = cached_rej[-len(ret[1]):][~ret[1].isnan()] if cached_rej is not None else ret[0][~ret[1].isnan()]
    arc_acc = ret[1][~ret[1].isnan()]
    _auc_str = f": {get_auc_score(y=arc_acc, x=arc_rej):.3f}" if write_auc_score else ""
    fig.add_trace(go.Scatter(x=arc_rej, y=arc_acc, **scatter_opts, name=f"all classes{_auc_str}"))

    # track uncertainty values (this would use the uncertainty thresholds over all classes)
    # for use in other ARC functions (avoid having to re-compute this one) - use sparingly
    if track_arc_values:
        if reject_on_cached:
            print("Caution: both `track_arc_values` and `reject_on_cached` were set to True. \
                  These are mutually exclusive! Skipping caching new ARC values.")
        else:
            model_version[f"{prefix}/arc_rej"].extend(list(ret[0][~ret[1].isnan()]))
            model_version[f"{prefix}/arc_acc"].extend(list(ret[1][~ret[1].isnan()]))
            # we need the ability to reject OOD samples based on the validation uncertainty thresholds
            model_version[f"{prefix}/arc_unc"].extend(list(ret[2][~ret[1].isnan()]))


    # TODO compute the random baseline and plot it as a dotted line


    # TODO compute the "rolling spread" of the ARC and plot it as a hull over both the ARC
    # and the random baseline



    title = "Accuracy-Rejection Curve"
    if title_suffix:
        title = f"{title} {title_suffix}"

    fig.update_layout(title=title,
                        xaxis_title="Rejection Rate" if cached_rej is None else "Rejection Rate (wrt. ref.)",
                        yaxis_title="Conditional Accuracy")

    fig.update_yaxes(range=[-0.05, 1.05])

    if model_version is not None:
        model_version[f"{prefix}/accuracy_rejection_curve_base"].upload(fig)

    return fig

def track_cross_valid(model_version, all_folds_metrics: dict, overwrite: bool = False):
    for metric_name, params in all_folds_metrics.items():
        for dataset in params["apply"]:
            field = f"{dataset}/{metric_name}"
            if model_version.exists(f"{field}_mean") and not overwrite:
                print(f"Skipping evaluating metric '{field}'. Already exists!")
                continue
            values = get_fold_values(model_version, dataset, metric_name)
            if len(values) == 0:
                if params["allow_missing"]:
                    continue
                else:
                    raise KeyError(f"Model version has no metric '{metric_name}' in {dataset} and 'allow_missing' is False!")
            mean_value = np.mean(values)
            std_value = np.std(values)
            model_version[f"{field}_mean"] = mean_value
            model_version[f"{field}_std"] = std_value
            
            # Create a box plot for each metric
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(values)
            ax.set_title(f'{metric_name} Distribution Across Folds ({dataset.capitalize()})')
            ax.set_ylabel(metric_name)
            model_version[f"{field}_distribution"].upload(File.as_image(fig))
            plt.close(fig)
