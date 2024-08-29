from fastai.optimizer import *
from fastai.torch_basics import *
from fastai.callback.schedule import ParamScheduler
from functools import partial, update_wrapper


def nesterov_step(p, mom, grad_avg, **kwargs):
    "Step for SGD with nesterov momentum with `lr`"
    grad_avg.mul_(mom).add_(p.grad.data, alpha=1.)
    return {'grad_avg': grad_avg}

def SGD_with_nesterov(
    params:Tensor|Iterable, # Model parameters
    lr:float|slice, # Default learning rate
    mom:float=0., # Gradient moving average (β1) coefficient
    wd:Real=0., # Optional weight decay (true or L2)
    decouple_wd:bool=True, # Apply true weight decay or L2 regularization (SGD)
    nesterov:bool=True # Apply nesterov momentum 
) -> Optimizer:
    "A SGD `Optimizer`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    if mom != 0: 
        cbs.append(average_grad)
        cbs += ([nesterov_step, momentum_step] if nesterov else [momentum_step])  
    else: 
        cbs.append(sgd_step)
    return Optimizer(params, cbs, lr=lr, mom=mom, wd=wd)

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

class ParamSchedulerEpochs(ParamScheduler):
    def before_batch(self): self._update_val(self.epoch)

    def _update_val(self, epoch):
        for n,f in self.scheds.items(): self.opt.set_hyper(n, f(epoch))

def SchedStep(start, step, drop):
    """Reduces the `start` value by `step` every `drop` `epoch`s"""
    def _inner(epoch): return start * (step ** (epoch // drop))
    return _inner


class CosineDecayWithWarmup:
    def __init__(self, start, lr_warmup_epochs, num_epochs, min_lr=0.0):
        self.start = start
        self.lr_warmup_epochs = lr_warmup_epochs
        self.num_epochs = num_epochs
        self.min_lr = min_lr

    def __call__(self, pct):
        if pct < self.lr_warmup_epochs / self.num_epochs:
            return self.start * (pct / (self.lr_warmup_epochs / self.num_epochs))
        else:
            progress = (pct - self.lr_warmup_epochs / self.num_epochs) / (1 - self.lr_warmup_epochs / self.num_epochs)
            return self.min_lr + (self.start - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

class WarmUpPieceWiseConstantSchedStep:
    def __init__(self, start, lr_warmup_epochs, lr_decay_epochs, lr_decay_ratio, num_epochs, pct_offs):
        self.start = start
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_decay_epochs = lr_decay_epochs
        self.lr_decay_ratio = lr_decay_ratio
        self.num_epochs = num_epochs
        self.pct_offs = pct_offs

    def __call__(self, pct):
        learning_rate = self.start
        if self.lr_warmup_epochs >= 1:
            learning_rate *= (pct + self.pct_offs) * (float(self.num_epochs) / self.lr_warmup_epochs)

        _decay_epochs = [float(e) / self.num_epochs for e in [self.lr_warmup_epochs] + self.lr_decay_epochs]
        for index, start_epoch in enumerate(_decay_epochs):
            if pct < start_epoch:
                break
            learning_rate = self.start * self.lr_decay_ratio**index

        return learning_rate
