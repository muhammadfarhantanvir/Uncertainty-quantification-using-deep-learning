from fastai.vision.all import *
import torch.autograd as autograd
from torch.func import stack_module_state
import copy
from torch.nn.init import xavier_normal_
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, Categorical
from fastai.callback.core import Callback


class aleatoric_entropy(Callback):
    def __init__(self, reduction_dim: int = 1, eps=1e-10):
        self.reduction_dim = reduction_dim
        self.eps = eps

    def before_validate(self):
        self._aleatoric_unct = []

    def after_pred(self):
        # perform activation
        act = getcallable(self.loss_func, 'activation')

        # preds has shape (batch_size, Nsamples, classes)
        with torch.no_grad():
            activated = act(self.learn.pred.clone())
            sample_preds_entropy = -(activated * torch.log(activated + self.eps)).sum(dim=2, keepdim=False)
        self._aleatoric_unct.append(sample_preds_entropy.mean(dim=self.reduction_dim, keepdim=False))

    def after_validate(self):
        self.learn.aleatoric_unct = torch.concat(self._aleatoric_unct)


class f_WGD(Callback):
    order = 8
    run_valid = False

    def __init__(self, kernel, grad_estim, gamma: float, annealing_steps: int, density_method: str = "kde", verbose: bool = False,
                 **kwargs):
        """
        Args:
            :param:`kernel`: kernel instance (callable)
            :param:`grad_estim`: instance of a gradient estimator which exposes a `compute_score_gradients` method
            :param:`gamma`: multiplying factor for the driving force
            :param:`annealing_steps`: number of annealing steps
            :param:`density_method`: Can take the values 'kde', 'ssge' or 'sge'
        """
        self.grad_estim = grad_estim
        self.kernel = kernel
        self.gamma = gamma
        self.annealing_steps = annealing_steps
        self.ann_schedule = [gamma]
        self.density_method = density_method
        self.verbose = verbose

    def compute_gradient_density(self):
        """Computes the gradient density using the density estimator specified by :attr:`density_method`.
        The negative of this value is the repulsive force in the update rule.
        """

        if self.density_method == 'kde':
            _K_f = self.kernel(self.learn.pred_k, self.learn.pred_k.detach())
            _grad_K = autograd.grad(_K_f.sum(), self.learn.pred_k)[0]
            _grad_K = _grad_K.view(self.learn.model.ensemble_size, -1)

            self.learn.grad_density = _grad_K / _K_f.sum(1, keepdim=True)

        elif self.density_method == 'ssge':
            self.learn.grad_density = self.grad_estim.compute_score_gradients(self.learn.pred, self.learn.pred)

        elif self.density_method == 'sge':
            _eta = 0.01
            _K_f = self.kernel(self.learn.pred_k, self.learn.pred_k.detach())
            _grad_K = autograd.grad(_K_f.sum(), self.learn.pred_k)[0]
            _grad_K = _grad_K.view(self.learn.model.ensemble_size, -1)
            K_ = _K_f + _eta * torch.eye(_K_f.shape[0], device=_K_f.device)
            self.learn.grad_density = torch.linalg.solve(K_, _grad_K)

        else:
            print(f"Unexpected density method, expected one of [kde, sge, ssge], got {self.density_method}!")

    def before_fit(self):
        "Set selected parameters as hyperparameters in the optimizer, so they will be logged automatically."
        self.learn.opt.set_hyper('grad_estim', self.grad_estim.name)
        self.learn.opt.set_hyper('grad_estim_eta', self.grad_estim.eta)
        self.learn.opt.set_hyper('kernel', self.kernel.name)
        self.learn.opt.set_hyper('gamma', self.gamma)
        self.learn.opt.set_hyper('annealing_steps', self.annealing_steps)
        self.learn.opt.set_hyper('density_method', self.density_method)
        self.learn.opt.set_hyper('ensemble_opt_func', type(self).__name__)

        if hasattr(self.grad_estim, "kernel"):
            self.learn.opt.set_hyper('grad_estim_kernel', self.grad_estim.kernel.name)

        self.ann_schedule = torch.cat([torch.linspace(self.gamma / max(self.annealing_steps, 1), self.gamma, self.annealing_steps),
                                       self.gamma * torch.ones(self.learn.n_epoch - self.annealing_steps)])

    def before_backward(self):
        _score_func = autograd.grad(self.learn.log_prob.sum(), self.learn.pred, retain_graph=True)[0]

        self.learn.score_func = _score_func.view(self.learn.model.ensemble_size, -1)

        # do not call .backward() on the loss, move to "before_step" directly
        # Caution: "after_backward" is never reached!
        # the gradients are computed explicitly instead in 2-3 locations using autograd.grad
        raise CancelBackwardException

    def before_step(self):
        ### gradient functional prior ###
        self.learn.pred = self.learn.pred.view(self.learn.model.ensemble_size, -1)

        self.learn.grad_prior = self.grad_estim.compute_score_gradients(self.learn.pred, self.learn.prior_pred)

        ### update rule ###
        self.learn.drive = self.learn.score_func + self.learn.grad_prior

        self.compute_gradient_density()

        _driv_mult = self.ann_schedule[self.learn.epoch] if (self.learn.epoch < len(self.ann_schedule)) else \
        self.ann_schedule[-1]
        self.learn.drive *= _driv_mult

        _f_phi = (self.learn.drive - self.learn.grad_density)

        # TODO vectorize this call with vmap
        _w_phi_elems = [autograd.grad(self.learn.pred, _wt, grad_outputs=_f_phi, retain_graph=True)[0]
                        for _wt in self.learn.model.mparams.values()]

        if self.verbose:
            print(f"params.grad before: {[p.grad for p in self.learn.model.parameters() if p.requires_grad][0]}")

        _trainable_params = [p for p, *_ in self.learn.opt.all_params() if p.requires_grad]
        if self.verbose:
            print(f"trainable_params: {[p.shape for p in _trainable_params]}")
            print(f"_w_phi_elems: {[p.shape for p in _w_phi_elems]}")

        for j in range(_w_phi_elems[0].shape[0]):
            for i, param in enumerate(_trainable_params):
                _i = i % len(_w_phi_elems)
                param.grad = -_w_phi_elems[_i][j]

        if self.verbose:
            print(f"params.grad after: {[p.grad for p in self.learn.model.parameters() if p.requires_grad][0]}")

        self.learn.w_phi = torch.cat([wp.flatten(start_dim=1) for wp in _w_phi_elems], dim=1)

        if self.verbose:
            _mparams = [v for v in self.learn.model.mparams.values()]
            print(f"mparam before step:\n{_mparams[0][0][0]}")

    def after_step(self):
        # sanity check: do the gradients transfer to the model.mparams?
        if self.verbose:
            _mparams = [v for v in self.learn.model.mparams.values()]
            print(f"mparam after step:\n{_mparams[0][0][0]}")


class activate_logits(Callback):
    """Applies an activation function to `learn.pred`
    If no activation function is provided, defaults to the loss's activation."""

    def __init__(self, act=None):
        self.act = act

    def after_pred(self):
        _act = self.act if self.act is not None else getcallable(self.learn.loss_func, 'activation')
        self.learn.pred = _act(self.learn.pred)


class auto_repulse(Callback):
    order = f_WGD.order - 1

    def before_fit(self):
        "Set selected parameters as hyperparameters in the optimizer, so they will be logged automatically."
        self.learn.opt.set_hyper('is_auto_repulsive', True)

    def before_step(self):
        self.learn.pred_k = self.learn.pred.clone().view(self.learn.model.ensemble_size,
                                                         -1)  # [num_particles, classes * batch_size]


class repulse_additional_points(Callback):
    """
    Computes the repulsion on additional points. 
    The additional points need to be provided to the learner's attribute 'X_add'.
    """
    order = auto_repulse.order

    def before_fit(self):
        "Set selected parameters as hyperparameters in the optimizer, so they will be logged automatically."
        self.learn.opt.set_hyper('is_auto_repulsive', False)

    def before_step(self):
        if self.learn.X_add is not None:
            _logits = self.learn.model.forward(*self.learn.X_add)
            _act = getcallable(self.learn.loss_func, 'activation')

            self.learn.pred_k = (_act(_logits)).view(self.learn.model.ensemble_size, -1)


class stack_module_states(Callback):
    """Stacks the model states for use with vectorized call to forward."""

    def before_batch(self):
        self.learn.model.mparams, self.learn.model.buffers = stack_module_state(self.learn.model.models)


class predict_on_prior_weights(Callback):
    order = f_WGD.order - 1
    run_valid = False

    def before_fit(self):
        "Set selected parameters as hyperparameters in the optimizer, so they will be logged automatically."
        self.learn.opt.set_hyper('predict_on_prior_weights', True)

    def before_step(self):
        # sample weights of shape (ensemble_size, num_params)
        _w_prior = self.learn.prior.sample(torch.Size([self.learn.model.ensemble_size]))

        _device = self.learn.xb[0].device
        self.learn.model.set_tmp_weights_from_flattened(_w_prior, _device)

        self.learn.model.use_temp = True

        # the resulting tensor will be detached from the current graph inside the `compute_score_function`
        # so we might get away with performing the following line without gradients
        with torch.no_grad():
            _act = getcallable(self.learn.loss_func, 'activation')
            self.learn.prior_pred = _act(self.learn.model.forward(*self.learn.xb)).reshape(
                self.learn.model.ensemble_size, -1)

        # we have to use above line in a context, such that we can restore the old params and buffers afterwards
        self.learn.model.use_temp = False


class Unorm_post(Callback):
    order = f_WGD.order - 2
    run_valid = False

    def __init__(self, prior, pred_dist_std: float = 1.0, prior_var: float = 1.0, add_prior: bool = True):
        """Implementation of an unnormalized posterior for a neural network model.
        
        Args:
            :param:`prior`: prior instance from torch.distributions or custom, .log_prob() method is required
            :param:`pred_dist_std`:  The standard deviation of the predictive distribution. 
                            Note, this value should be fixed and reasonable for a given dataset.
        """
        self.prior = prior
        self.pred_dist_std = pred_dist_std
        self.prior_var = prior_var
        self.add_prior = add_prior

    def after_create(self):
        self.learn.prior = self.prior
        self.learn.log_prob = None

    def before_fit(self):
        "Set selected parameters as hyperparameters in the optimizer, so they will be logged automatically."
        self.learn.opt.set_hyper('pred_dist_std', self.pred_dist_std)
        self.learn.opt.set_hyper('prior', str(self.prior))
        self.learn.opt.set_hyper('prior_var', self.prior_var)
        self.learn.opt.set_hyper('prior_method', type(self).__name__)

    def after_loss(self):
        _ll = -self.learn.loss_grad * self.learn.dl.n / self.pred_dist_std ** 2

        if self.add_prior:
            _particles = self.learn.model.particles
            _log_prob = self.learn.prior.log_prob(_particles)

            self.learn.log_prob = torch.add(_log_prob.sum(1), _ll)
        else:
            self.learn.log_prob = _ll


class calculate_norm_stats(Callback):
    """
    Use this callback for one full epoch on the training data without
    a `Normalize.from_stats()` `batch_tfms` applied to calculate the
    running mean and standard deviations.
    """

    def __init__(self, channel_dim):
        """All dimensions are reduced over, except `channel_dim`."""
        self.channel_dim = channel_dim
        self.dim = None
        self.data = []

    def before_batch(self):
        _shape = self.learn.x.shape
        if self.dim is None:
            self.dim = [i for i in range(len(_shape)) if i != self.channel_dim]
        self.data.append(self.learn.x)

    def after_epoch(self):
        _data = torch.cat(self.data)
        self.mean = torch.mean(_data, dtype=float, dim=self.dim)
        self.std = torch.std(_data, dim=self.dim)
        print(f"based on {_data.shape[0]} samples: means: ({self.mean}), stds: ({self.std})")


class unpack_model_output_dict(Callback):
    def after_pred(self):
        if isinstance(self.learn.pred, dict):
            self.learn.cov = self.learn.pred.get('covariance', None)
            self.learn.phi = self.learn.pred.get('random_features', None)
            self.learn.pred = self.learn.pred.get('logits', self.learn.pred)
        else:
            warn(f"Unable to unpack prediction, got {type(self.learn.pred)}, expected 'dict'!")


class gp_classifier_helper(Callback):
    run_valid = False

    def before_batch(self):
        if self.epoch < self.n_epoch - 1:
            self.learn.model.gp_classifier.reset_precision()

    def after_loss(self):
        self.learn.loss_grad += 0.5 * self.learn.model.gp_classifier.random_feature_layer.bias.square().sum()  #.sqrt()


class DataModCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_batch(self):
        Y_train_hot = torch.zeros(self.learn.yb[0].shape[0], self.learn.model.output_dim,
                                  device=self.learn.yb[0].device)
        self.learn.yb = (Y_train_hot.scatter_(1, self.learn.yb[0].reshape(-1, 1), 1),)

        self.learn.xb = (self.xb[0].double(),)  #post net makes all weights as doubles
        if not self.training:
            self.learn.xb = (self.xb[0],)

    def after_loss(self):
        self.learn.yb = (self.yb[0].argmax(dim=-1),)
        if not self.training:
            self.learn.loss = self.learn.loss / self.yb[0].shape[0]

    def after_step(self):
        self.learn.loss = self.learn.loss.float()  #fastai has hardcoded float type for Recorder AvgSmoothLoss-Logging
        self.learn.loss = self.learn.loss / self.yb[0].shape[
            0]  #avg loss per batch for logging to be consistent with non-fastai version


class PosteriorDensity(Callback):
    def __init__(self, output_dim: int, num_class_samples: list):
        """`num_class_samples`: Count of data from each class in training set. list of ints"""
        store_attr()

    def after_pred(self):
        _device = self.learn.pred.device
        _batch_size = self.learn.pred.shape[-2]

        log_q_zk = torch.zeros((_batch_size, self.output_dim)).to(_device)
        self.learn.alpha = torch.zeros((_batch_size, self.output_dim)).to(_device)

        if isinstance(self.learn.model.density_estimation, nn.ModuleList):
            for c in range(self.output_dim):
                log_p = self.learn.model.density_estimation[c].log_prob(self.learn.pred)
                log_q_zk[:, c] = log_p
                self.learn.alpha[:, c] = 1. + (self.num_class_samples[c] * torch.exp(log_q_zk[:, c]))
        else:
            log_q_zk = self.learn.model.density_estimation.log_prob(self.learn.pred)
            self.learn.alpha = 1. + (self.num_class_samples[:, None] * torch.exp(log_q_zk)).permute(1, 0)


class NormalizeAlphaPred(Callback):
    order = PosteriorDensity.order + 1

    def __init__(self, normalize_before_loss: bool = False):
        store_attr()

    def after_pred(self):
        if self.normalize_before_loss:
            self.learn.pred = torch.nn.functional.normalize(self.learn.alpha, p=1)
        else:
            self.learn.pred = self.learn.alpha

    def after_loss(self):
        if not self.normalize_before_loss:
            self.learn.pred = torch.nn.functional.normalize(self.learn.alpha, p=1)


class calc_validation_total_entropy(Callback):
    order = activate_logits.order + 1

    def __init__(self, reduction_dim: int = 0, eps=1e-20, do_activate: bool = True, log_base_str: str = "natural",
                 log_scale: float = None):
        self.reduction_dim = reduction_dim
        self.eps = eps
        self.do_activate = do_activate
        if log_base_str == "natural":
            self.log_func = torch.log
        elif log_base_str == "two":
            self.log_func = torch.log2
        else:
            print(f"In total_entropy: Unexpected log_base_str: {log_base_str}. Defaulting to natural logarithm.")
            self.log_func = torch.log

        self.log_scale = log_scale

    def before_validate(self):
        self._total_unct = []

    def after_loss(self):
        # preds has shape (Nsamples, batch_size, classes)
        with torch.no_grad():
            if self.do_activate:
                # perform activation
                act = getcallable(self.loss_func, 'activation')
                activated = act(self.learn.pred.clone())
            else:
                activated = self.learn.pred
            posterior_preds = activated.mean(dim=self.reduction_dim, keepdim=False)
            log_preds = self.log_func(posterior_preds + self.eps)
            if self.log_scale is not None:
                log_preds /= self.log_scale
            _total_unct = -(posterior_preds * log_preds).sum(dim=1, keepdim=False)
        self._total_unct.append(_total_unct)

    def after_validate(self):
        self.learn.total_unct = torch.concat(self._total_unct)


class calc_sngp_uncertainty(Callback):
    order = activate_logits.order + 1

    def __init__(self, reduction_dim: int = 0, eps=1e-20, log_base_str: str = "natural", log_scale: float = None):
        self.reduction_dim = reduction_dim
        self.eps = eps

        if log_base_str == "natural":
            self.log_func = torch.log
        elif log_base_str == "two":
            self.log_func = torch.log2
        else:
            print(f"Unexpected log_base_str: {log_base_str}. Defaulting to natural logarithm.")
            self.log_func = torch.log
        self.log_scale = log_scale

    def before_validate(self):
        self._total_unct = []

    def after_loss(self):
        try:
            with torch.no_grad():
                logits = self.learn.pred
                if logits is None:
                    return

                covariances = self.learn.cov
                if covariances is not None:
                    diag_cov = torch.diagonal(covariances, dim1=-2, dim2=-1) + self.eps
                    uncertainty = diag_cov
                else:
                    raise ValueError("Covariance matrix is missing.")

                self._total_unct.append(uncertainty)

        except ValueError as e:
            print(f"ValueError encountered: {e}")
            self._total_unct.append(torch.tensor([]))
            raise e

    def after_validate(self):
        # Concatenate all valid uncertainties
        valid_uncertainties = [unc for unc in self._total_unct if unc.numel() > 0]
        if valid_uncertainties:
            self.learn.total_unct = torch.cat(valid_uncertainties, dim=0)
        else:
            self.learn.total_unct = torch.tensor([])

        # Ensure the total number of uncertainties matches the number of samples processed
        if self.learn.total_unct.shape[0] != self.learn.dl.n:
            raise ValueError(
                f"Total uncertainties shape {self.learn.total_unct.shape} does not match the number of samples processed {self.learn.dl.n}")


class average_sampled_preds(Callback):
    order = calc_validation_total_entropy.order + 2

    def __init__(self, reduction_dim: int = 0):
        self.reduction_dim = reduction_dim

    def after_loss(self):
        self.learn.pred = self.learn.pred.mean(self.reduction_dim)

class calc_dirichlet_uncertainty(Callback):
    order = activate_logits.order + 1

    def __init__(self, eps: float = 1e-20, normalize_uncertainty: bool = False):
        self.eps = eps
        self.normalize_uncertainty = normalize_uncertainty
        self._aleatoric_unct = []
        self._epistemic_unct = []

    def before_validate(self):
        self._aleatoric_unct = []
        self._epistemic_unct = []

    def after_loss(self):
        with torch.no_grad():
            logits = self.learn.alpha

            # Calculate aleatoric uncertainty using Categorical distribution entropy
            p = F.normalize(logits, p=1, dim=-1)
            aleatoric_entropy = Categorical(p).entropy()

            # Ensure aleatoric_entropy is at least 1-dimensional
           # if aleatoric_entropy.dim() == 0:
            #    aleatoric_entropy = aleatoric_entropy.unsqueeze(0)

            self._aleatoric_unct.append(aleatoric_entropy)

            # Calculate epistemic uncertainty using Dirichlet distribution entropy
            epistemic_entropy = Dirichlet(logits).entropy()

            # Ensure epistemic_entropy is at least 1-dimensional
            #if epistemic_entropy.dim() == 0:
            #    epistemic_entropy = epistemic_entropy.unsqueeze(0)

            self._epistemic_unct.append(epistemic_entropy)

    def after_validate(self):
        aleatoric_unct = torch.cat(self._aleatoric_unct)
        epistemic_unct = torch.cat(self._epistemic_unct)

        if self.normalize_uncertainty:
            # Normalize uncertainties to [0, 1] range
            aleatoric_unct = (aleatoric_unct - aleatoric_unct.min()) / (aleatoric_unct.max() - aleatoric_unct.min() + self.eps)
            epistemic_unct = (epistemic_unct - epistemic_unct.min()) / (epistemic_unct.max() - epistemic_unct.min() + self.eps)
        
        # Calculate total uncertainty
        total_unct = aleatoric_unct + (-epistemic_unct)

        # Aggregate uncertainties by computing the mean and rounding to three decimal points
        self.learn.aleatoric_unct = aleatoric_unct
        self.learn.epistemic_unct = -(epistemic_unct)
        self.learn.total_unct = total_unct


class MixUpCallback(Callback):
    "Callback that creates the mixed-up input and target"
    learner: Learner
    alpha: float = 0.4
    stack_x: bool = False
    stack_y: bool = True
    run_valid: bool = False

    def before_batch(self, last_input, last_target):
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else:
            new_input = (
                        last_input * lambd.view(lambd.size(0), 1, 1, 1) + x1 * (1 - lambd).view(lambd.size(0), 1, 1, 1))
        if self.stack_y:
            new_target = torch.cat([last_target[:, None].float(), y1[:, None].float(), lambd[:, None].float()], 1)
        else:
            new_target = last_target * lambd + y1 * (1 - lambd)
        return (new_input, new_target)
