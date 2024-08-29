from fastai.vision.all import *

import sys



sys.path.append("./")

from src.architectures.selunet import ConvNet
from src.loss import EnsembleClassificationLoss
from src.callback import Unorm_post, f_WGD, auto_repulse, repulse_additional_points, stack_module_states, predict_on_prior_weights, activate_logits
from src.densities.kernel import RBF
from src.architectures.gradient_estim import SpectralSteinEstimator
from src.optimizer import wrapped_partial
from src.architectures.ensemble import VectorizedEnsemble
from src.densities.distributions import Normal
from src.neptune_callback import NeptuneCallbackBayesianEnsemble as NeptuneCallback
from src.datasets import get_data_block
from src.evaluation import eval_model
from src.callback import calc_validation_total_entropy

from src.constants import DATASET_DIRECTORY, PROJECT_NAME, MODEL_PATH
import neptune
from neptune.types.mode import Mode as RunMode


num_classes = 10
batch_size = 128
lr = 0.00025
use_wd = False # False for L2 reg (Adam), true for weight decay (AdamW)
num_models = 20
num_epochs = 3
rand_flip_prob = 0.5
train_valid_split_seed = 67280421310721
weights_seed = None
dataset = "cifar10"

end_lr = lr * 0.01  # Setting the end learning rate to 1% of the starting learning rate
lr_schedule = {'lr': SchedCos(start=lr, end=end_lr)}

# Comment when using learn fit_one_cycle 
#  ~~~ Start ~~~~ 
# learning rate schedule with warmup
start_lr = 1e-6  # Starting learning rate for warmup
warmup_epochs = 3  # epoch no. for warmup

# Linear warmup followed by cosine decay
lr_schedule = {
    'lr': combine_scheds(
        [warmup_epochs/num_epochs, (num_epochs-warmup_epochs)/num_epochs],
        [SchedLin(start=start_lr, end=lr), SchedCos(start=lr, end=lr * 0.01)]
    )
}

# ~~~ End ~~~~

net_params = {
    'channels_in': 3,
    'num_channels_out': [6,16],
    'kernel_size': 5,
    'conv_stride': 1,
    'max_pool': 2,
    'pool_stride': 2,
    'num_conv': 2, 
    'num_hidden': [120, 84],
    'num_linear': 3,
    'num_classes': num_classes,
    'activation': nn.ReLU
}

device = torch.device('cuda:0')
# not pretty, but it works
torch.cuda.set_device(device)

# seed the generator for the train/valid split
tgen = torch.Generator()
if train_valid_split_seed is not None:
    tgen = tgen.manual_seed(train_valid_split_seed)
else:
    train_valid_split_seed = tgen.seed()

# seed the generator for the weigths
wgen = torch.Generator(device=device)
if weights_seed is not None:
    wgen = tgen.manual_seed(weights_seed)
else:
    weights_seed = wgen.seed()

cifar10 = get_data_block(dataset, rand_flip_prob=rand_flip_prob, generator=tgen)

data_path = Path(DATASET_DIRECTORY)/dataset
dls = cifar10.dataloaders(data_path/"train", bs=batch_size, device=device)

tags = [
    "AdamW" if use_wd else "Adam",
    net_params['activation'].__name__
]

run = neptune.init_run(
    source_files=["./src/train_BE_cifar.py",
                    "./src/architectures/ensemble.py",
                    "./src/processing.py",
                    "./src/densities/kernel.py",
                    "./src/architectures/gradient_estim.py",
                    "./src/optimizer.py",
                    "./src/loss.py",
                    "./src/architectures/selunet.py",
                    "./src/callback.py", 
                    "./src/densities/distributions.py",
                    "./src/neptune_callback.py",
                    "./src/neptune_tracking.py",
                    "./src/uncertainty_utils.py",
                    "./src/evaluation.py",
                    "./src/net_utils.py",
                    "./src/datasets.py"],
    project=PROJECT_NAME,
    tags=tags,
    mode="debug"
)

neptune_model = "CS24-CF10RE"

base_net = [ConvNet(**net_params).to(dls.device) for _ in range(num_models)]
model = VectorizedEnsemble(models=base_net, init_weights=True, generator=wgen)
loss_func = EnsembleClassificationLoss(is_logits=False, num_classes=num_classes)
opt_func = wrapped_partial(Adam, decouple_wd=use_wd)  # authors implementation used Adam (L2 reg), not AdamW (weight decay)

neptune_callback = NeptuneCallback(run=run, 
                                   model_name=type(model).__name__, 
                                   model_params=None,
                                   net_name=type(base_net[0]).__name__, 
                                   net_params=net_params,
                                   upload_saved_models=None)

# log the dataset name
run[f"{neptune_callback.base_namespace}/io_files/resources/dataset/name"] = dataset

prior_variance = 0.1
prior = Normal(torch.zeros(model.num_params, device=device),
                torch.ones(model.num_params, device=device) * prior_variance,
                generator=wgen)
pred_dist_std = 1.

kernel = RBF()
grad_estim_kernel = RBF()
grad_estim = SpectralSteinEstimator(eta=0.01, kernel=grad_estim_kernel)
density_method = "ssge"
do_auto_repulse = True  # Only disable if predictions to compute the repulsive term on
                        # are provided through another callback instead

# sched = {'lr': SchedStep(lr, lr_sched_step, lr_sched_drop)}
# lr_scheduler = ParamSchedulerEpochs(sched)
gamma = 1.
annealing_steps = 0
# TODO refactor to use a ParamScheduler instead

filename = f"{dataset}_{model.name}_{run['sys/id'].fetch()}"
learner_cbs = [
                stack_module_states(),
                activate_logits(),
                Unorm_post(prior, pred_dist_std),
                predict_on_prior_weights(),
                f_WGD(kernel, grad_estim, gamma, annealing_steps, density_method),
                GradientClip(),
                SaveModelCallback(monitor='valid_loss', min_delta=2e-3, fname=filename, every_epoch=False, at_end=False, with_opt=True),
                neptune_callback,
                ParamScheduler(lr_schedule),
                ]

if density_method == "sge" or density_method == "kde":
    learner_cbs.append(auto_repulse()) if do_auto_repulse else learner_cbs.append(repulse_additional_points())

# store debug runs under the default path
path = MODEL_PATH if run._mode != RunMode.DEBUG else None

learn = Learner(dls=dls, model=model, lr=lr, loss_func=loss_func, opt_func=opt_func, cbs=learner_cbs, path=path)

# train the model
with learn.no_bar():
    learn.fit(num_epochs) # Comment line for fit_one_cycle
    # learn.fit_one_cycle(num_epochs, lr_max=lr)  # unComment for fit_one_cycle


# log the split seed
run[f"{neptune_callback.base_namespace}/config/train_valid_split_seed"] = str(train_valid_split_seed)
run[f"{neptune_callback.base_namespace}/config/weights_seed"] = str(weights_seed)

# log the specific Adam optimizer version
if run[f"{neptune_callback.base_namespace}/config/optimizer/name"].fetch() == "Adam":
    run[f"{neptune_callback.base_namespace}/config/optimizer/name"] = "AdamW" if use_wd else "Adam"


### evaluate the trained model
# prepare the test loader
items_test = cifar10.get_items(data_path/"test")
dl_test = dls.test_dl(items_test, rm_type_tfms=None, num_workers=0, with_labels=True, with_input=False)
dataset_test = dataset  # make sure to update when testing on a separate dataset, e.g., OOD data

# prepare additional fields to log in the model_version
# which the program may not be able to infer
log_fields = {
    "data/validation/name": dataset,
    "data/validation/size": learn.dls[1].n,
    "data/testing/name": dataset_test,
    "data/testing/size": dl_test.n,
}

_unct_cbs = []
_log_scale = torch.log2(torch.tensor(10, dtype=torch.float))
_unct_cbs.append(calc_validation_total_entropy(
    do_activate=False,  # set to True if `activate_logits` is not set on the learner
    log_base_str="two",  # the authors of repulsive ensembles use this
    log_scale=_log_scale))  # the authors of repulsive ensembles use this

eval_model(
    learn=learn,
    neptune_model=neptune_model,
    neptune_project=PROJECT_NAME,
    run=run,
    dl_valid=learn.dls[1],
    dl_test=dl_test,
    neptune_fit_idx=neptune_callback.fit_index,
    do_validate=True,
    do_test=True,
    neptune_base_namespace=neptune_callback.base_namespace,
    log_additional=log_fields,
    reduce_preds=True,
    uncertainty_callbacks=_unct_cbs
)


run.stop()
