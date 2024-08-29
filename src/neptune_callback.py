from fastai.vision.all import *
from neptune.integrations.fastai import NeptuneCallback


class NeptuneCallbackBayesianEnsemble(NeptuneCallback):
    def __init__(self, batch_subsampling:int=20, model_name:str=None, model_params:dict=None, net_name:str=None, net_params:dict=None, *args, **kwargs):
        super(NeptuneCallbackBayesianEnsemble, self).__init__(*args, **kwargs)

        self.batch_subsampling = batch_subsampling
        self.model_name = model_name
        self.model_params = model_params
        self.net_name = net_name
        self.net_params = net_params

    @property
    def _optimizer_states(self) -> Optional[dict]:
        if hasattr(self, "opt") and hasattr(self.opt, "state"):
            state = [self.opt.state[p] for p,*_ in self.opt.all_params()]
            
            if len(state) == 1:
                return dict(state[0])

            return {
                f"group_layer_{layer}": {hyper: value for hyper, value in opts.items()}
                for layer, opts in enumerate(state)
            }

    def _log_learner_training_loop(self):
        s = io.StringIO()
        with redirect_stdout(s): self.learn.show_training_loop()

        self.neptune_run[f"{self.base_namespace}/config/learner"] = s.getvalue()

    def _log_ensemble_size(self):
        if hasattr(self.learn.model, 'ensemble_size'):
            self.neptune_run[f"{self.base_namespace}/config/ensemble_size"] = self.learn.model.ensemble_size

    def _log_model_name(self):
        if hasattr(self.learn.model, 'name'):
            self.neptune_run[f"{self.base_namespace}/config/model_str"] = self.learn.model.name

    def _log_attr_loss_func(self):
        if hasattr(self.learn, 'attr_loss_func'):
            attr_name = f"_{self.learn.attr_loss_func}-{self.learn.attr_reduction}"
            self.neptune_run[f"{self.base_namespace}/config/criterion"] = self._optimizer_criterion + attr_name

    def _log_model_configuration(self):
        super(NeptuneCallbackBayesianEnsemble, self)._log_model_configuration()

        self._log_learner_training_loop()
        self._log_model_name()
        self._log_ensemble_size()
        self._log_attr_loss_func()

    def _filter_states_batch(self, states, prefix, step):
        raise NotImplementedError

    def _log_optimizer_states(self, prefix, func, **kwargs):
        if torch.tensor(["group_layer" in key for key in self._optimizer_states.keys()]).all():
            for param, value in self._optimizer_states.items():
                metric = f"{prefix}/{param}"
                func(value, metric, **kwargs)
        else:
                func(self._optimizer_states, prefix, **kwargs)

    def after_create(self):
        super(NeptuneCallbackBayesianEnsemble, self).after_create()

        if self.model_name is not None:
             self.neptune_run[f"{self.base_namespace}/config/model/model_name"] = self.model_name

        if self.model_params is not None:
            for k,v in self.model_params.items():
                if isinstance(v, list):
                    _value = str(v)
                elif isinstance(v, type):
                    _value = v.__name__
                elif v is None:
                    _value = "None"
                else:
                    _value = v
                self.neptune_run[f"{self.base_namespace}/config/model/model_params/{k}"] = _value

        if self.net_name is not None:
            self.neptune_run[f"{self.base_namespace}/config/model/net_name"] = self.net_name

        if self.net_params is not None:
            for k,v in self.net_params.items():
                if isinstance(v, list):
                    _value = str(v)
                elif isinstance(v, type):
                    _value = v.__name__
                elif v is None:
                    _value = "None"
                else:
                    _value = v
                self.neptune_run[f"{self.base_namespace}/config/model/net_params/{k}"] = _value
        
    def after_batch(self):
        _step = self.learn.train_iter-1
        if _step % self.batch_subsampling:
            return

        prefix = f"{self.base_namespace}/metrics/fit_{self.fit_index}/{self._target}/batch"

        self.neptune_run[f"{prefix}/loss"].append(value=torch.mean(self.learn.loss.clone()), step=_step)
            
        if self._target != "validation":
            self.neptune_run[f"{prefix}/estimated_loss"].append(value=torch.mean(self.learn.loss_grad.clone()), step=_step)

            if hasattr(self.learn, 'grad_density'):
                self.neptune_run[f"{prefix}/repulsive_force"].append(torch.quantile(-self.learn.grad_density, q=0.5), step=_step)

            if hasattr(self.learn, 'drive'):
                self.neptune_run[f"{prefix}/driving_force"].append(torch.quantile(self.learn.drive, q=0.5), step=_step)
                if torch.ge(self.learn.drive, 1e-10).any():
                    _forces_ratio = torch.quantile(torch.mean(-self.learn.grad_density/self.learn.drive, dim=1), q=0.5).abs()
                else:
                    _forces_ratio = 0.0
                self.neptune_run[f"{prefix}/forces_ratio"].append(_forces_ratio, step=_step)

            #self._log_optimizer_states(f"{prefix}/optimizer_states", self._filter_states_batch, step=_step)

            if hasattr(self, "smooth_loss"):  # smooth_loss is computed for training only
                self.neptune_run[f"{prefix}/smooth_loss"].append(value=self.learn.smooth_loss.clone(), step=_step)
