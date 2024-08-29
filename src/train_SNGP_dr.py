from fastai.vision.all import *

import sys
sys.path.append("./")

from src.architectures.wide_resnet import WideResNet
from src.architectures.sngp_classification_layer import SNGP
from src.optimizer import wrapped_partial, WarmUpPieceWiseConstantSchedStep, ParamScheduler, SGD_with_nesterov
from src.loss import CrossEntropyClassificationLoss
from src.callback import unpack_model_output_dict, gp_classifier_helper
from src.neptune_callback import NeptuneCallbackBayesianEnsemble as NeptuneCallback
from src.datasets import get_data_block
from src.evaluation import eval_ensemble
from src.constants import DATASET_DIRECTORY, PROJECT_NAME, MODEL_PATH

import neptune
from neptune.types.mode import Mode as RunMode

num_classes = 5  # DR512 has 5 classes
batch_size = 2  # Adjusted for larger image size otherwise CUDA out of memory for this device.
lr = 0.08
use_wd = False  # False for L2 reg (Adam), true for weight decay (AdamW)
wd = 3e-4
use_nesterov = True
one_minus_momentum = 0.1
num_models = 20
num_epochs = 250
rand_flip_prob = 0.5
train_valid_split_seed = 67280421310721
weights_seed = None
dataset = "dr512"

net_params = {
    'channels_in': 3,
    'num_conv': 28,
    'widen_factor': 10,
    'dropout_p': 0.1,
    'use_spec_norm': True,
    'spec_norm_iteration': 1,
    'num_linear': 0,
    'num_classes': num_classes,
    'activation': nn.ReLU
}

model_params = {
    "pre_clas_features": 640*16*16,
    "num_classes": num_classes,
    "custom_random_features_initializer": None,
}


device = torch.device('cuda:0')
torch.cuda.set_device(device)

# Seed generators
tgen = torch.Generator()
if train_valid_split_seed is not None:
    tgen = tgen.manual_seed(train_valid_split_seed)
else:
    train_valid_split_seed = tgen.seed()

wgen = torch.Generator(device=device)
if weights_seed is not None:
    wgen = tgen.manual_seed(weights_seed)
else:
    weights_seed = wgen.seed()

# Data block for dr512
data_path = Path(DATASET_DIRECTORY)/dataset
dr512 = get_data_block(dataset, train_labels_path=data_path/"trainLabels.csv", generator=tgen,rand_flip_prob=rand_flip_prob)
dls = dr512.dataloaders(data_path/"train", batch_size = batch_size, device=device)


tags = [
    "SGD_nesterov" if use_nesterov else "SGD",
    net_params['activation'].__name__
]

run = neptune.init_run(
    source_files=["./src/train_SNGP_dr.py",
                    "./src/train_SNGP_cifar.py",
                    "./src/processing.py",
                    "./src/optimizer.py",
                    "./src/loss.py",
                    "./src/architectures/wide_resnet.py",
                    "./src/callback.py", 
                    "./src/densities/gaussian_process.py",
                    "./src/architectures/sngp_classification_layer.py",
                    "./src/densities/random_fourier_features.py",
                    "./src/neptune_callback.py",
                    "./src/neptune_tracking.py",
                    "./src/uncertainty_utils.py",
                    "./src/evaluation.py",
                    "./src/net_utils.py",
                    "./src/datasets.py"],
    project=PROJECT_NAME,
    tags=tags,
    #mode="debug"
)

neptune_model = "CS24-DR512SG"



base_net = WideResNet(**net_params, generator=wgen).to(dls.device)
model = SNGP(
            **model_params,
            pre_classifier=base_net,
            generator=wgen
)
loss_func = CrossEntropyClassificationLoss(is_logits=True)
opt_func = wrapped_partial(SGD_with_nesterov, mom=1.0-one_minus_momentum, wd=wd, decouple_wd=use_wd, nesterov=use_nesterov)

neptune_callback = NeptuneCallback(run=run, 
                                   model_name=type(model).__name__, 
                                   model_params=model_params,
                                   net_name=type(base_net).__name__, 
                                   net_params=net_params,
                                   upload_saved_models=None)

# log the dataset name
run[f"{neptune_callback.base_namespace}/io_files/resources/dataset/name"] = dataset

filename = f"{dataset}_{model.name}_{run['sys/id'].fetch()}"
learner_cbs = [
                unpack_model_output_dict(),
                gp_classifier_helper(),
                GradientClip(),
                SaveModelCallback(monitor='valid_loss', min_delta=2e-3, fname=filename, every_epoch=False, at_end=False, with_opt=True),
                neptune_callback,
                ]


base_learning_rate = 0.08
base_lr = base_learning_rate * batch_size / 128
lr_decay_epochs = [60, 120, 160]

lr_decay_epochs = [(start_epoch * num_epochs) // 200 for start_epoch in lr_decay_epochs]
lr_decay_ratio = 0.2
lr_warmup_epochs = 1

_pct_offs = 1. / (len(dls.train) * num_epochs)
sched = {'lr': WarmUpPieceWiseConstantSchedStep(base_lr, lr_warmup_epochs, lr_decay_epochs, lr_decay_ratio, num_epochs, _pct_offs)}
lr_scheduler = ParamScheduler(sched)

# store debug runs under the default path
path = MODEL_PATH if run._mode != RunMode.DEBUG else None

learn = Learner(dls=dls, model=model, lr=base_lr, loss_func=loss_func, opt_func=opt_func, cbs=learner_cbs)

with learn.no_bar():
    learn.fit(num_epochs, cbs=lr_scheduler)


    
run[f"{neptune_callback.base_namespace}/config/train_valid_split_seed"] = str(train_valid_split_seed)
run[f"{neptune_callback.base_namespace}/config/weights_seed"] = str(weights_seed)


### evaluate the trained model

"""



"""



run.stop()
