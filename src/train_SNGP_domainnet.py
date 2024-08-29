from fastai.vision.all import *
import sys

sys.path.append("./")

from src.architectures.wide_resnet import WideResNet
from src.architectures.sngp_classification_layer import SNGP
from src.optimizer import wrapped_partial, WarmUpPieceWiseConstantSchedStep, ParamScheduler, SGD_with_nesterov
from src.loss import CrossEntropyClassificationLoss
from src.callback import unpack_model_output_dict, gp_classifier_helper, calculate_norm_stats
from src.neptune_callback import NeptuneCallbackBayesianEnsemble as NeptuneCallback
from src.datasets import get_data_block
from src.constants import PROJECT_NAME, DOMAINNET_DIRECTORY

import neptune

classes = ['lighthouse', 'eye', 'sink', 'dolphin', 'cow', 'banana', 'axe', 'hat', 'fish', 'face']
num_classes = len(classes)
batch_size = 256
lr = 0.08
epochs = 50
use_wd = False
wd = 3e-4
use_nesterov = True
one_minus_momentum = 0.1
num_models = 20
num_epochs = 250
rand_flip_prob = 0.5
train_valid_split_seed = 67280421310725
weights_seed = None
dataset = "domainnet"
sub_directory = 'quickdraw'
data_dir = f"{DOMAINNET_DIRECTORY}/{sub_directory}"
neptune_model = "CS24-DMNTSG"

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

device = torch.device('cuda:0')
torch.cuda.set_device(device)

# Seed the generator for the train/valid split
tgen = torch.Generator()
if train_valid_split_seed is not None:
    tgen = tgen.manual_seed(train_valid_split_seed)
else:
    train_valid_split_seed = tgen.seed()

# Seed the generator for the weights
wgen = torch.Generator(device=device)
if weights_seed is not None:
    wgen = tgen.manual_seed(weights_seed)
else:
    weights_seed = wgen.seed()

domainnet = get_data_block(dataset, do_normalize=True, include_classes=None)
dls = domainnet.dataloaders(data_dir, bs=batch_size, device=device)

tags = [
    "resnet",
    net_params['activation'].__name__
]

run = neptune.init_run(
    source_files=["./src/train_SNGP_domainnet.py",
                  "./src/processing.py",
                  "./src/optimizer.py",
                  "./src/loss.py",
                  "./src/architectures/wide_resnet.py",
                  "./src/callback.py",
                  "./src/densities/gaussian_process.py",
                  "./src/architectures/sngp_classification_layer.py",
                  "./src/densities/random_fourier_features.py",
                  "./src/neptune_callback.py",
                  "./src/net_utils.py",
                  "./src/datasets.py"],
    project=PROJECT_NAME,
    tags=tags,
    # mode="debug"
)

neptune_callback = NeptuneCallback(run=run, upload_saved_models=None)

# Log the dataset name
run[f"{neptune_callback.base_namespace}/io_files/resources/dataset/name"] = dataset

# base_net = WideResNet(**net_params, generator=wgen).to(dls.device)
# model = SNGP(
#     pre_clas_features=10*64,
#     num_classes=num_classes,
#     pre_classifier=base_net,
#     custom_random_features_initializer=None,
#     generator=wgen
# )

# Define loss function and optimizer
# loss_func = CrossEntropyClassificationLoss(is_logits=True)
# opt_func = wrapped_partial(SGD_with_nesterov, mom=1.0-one_minus_momentum, wd=wd, decouple_wd=use_wd, nesterov=use_nesterov)

# Initialize the learner with the Neptune callback
# calc_norm_callback = calculate_norm_stats(channel_dim=1) 
learn = vision_learner(dls=dls, arch=resnet18, metrics=accuracy, normalize=False)
learn.fit_one_cycle(epochs, cbs=[neptune_callback])

# Retrieve the calculated normalization stats
# norm_stats = learn.norm_stats
# print(f"Calculated normalization stats: {norm_stats}")

# learn.fit(epochs, lr, cbs=[neptune_callback])
# learn = Learner(dls, model, loss_func=loss_func, opt_func=opt_func, metrics=accuracy, cbs=[neptune_callback])


# Stop Neptune experiment
run.stop()






# run = neptune.init_run(
#     source_files=["./src/train_SNGP_domainnet.py",
#                     "./src/processing.py",
#                     "./src/optimizer.py",
#                     "./src/loss.py",
#                     "./src/architectures/wide_resnet.py",
#                     "./src/callback.py",
#                     "./src/densities/gaussian_process.py",
#                     "./src/architectures/sngp_classification_layer.py",
#                     "./src/densities/random_fourier_features.py",
#                     "./src/neptune_callback.py",
#                     #"./src/neptune_tracking.py",
#                     #"./src/uncertainty_utils.py",
#                     "./src/net_utils.py",
#                     "./src/datasets.py"],
#     project=PROJECT_NAME,
#     tags=tags,
#     mode="debug"
# )
# neptune_callback = NeptuneCallback(run=run, upload_saved_models=None)


# # log the dataset name
# run[f"{neptune_callback.base_namespace}/io_files/resources/dataset/name"] = dataset

# base_net = WideResNet(**net_params, generator=wgen).to(dls.device)
# model = SNGP(
#             pre_clas_features=10*64,
#             num_classes=num_classes,
#             pre_classifier=base_net,
#             custom_random_features_initializer=None,
#             generator=wgen
#         )
# loss_func = CrossEntropyClassificationLoss(is_logits=True)
# opt_func = wrapped_partial(SGD_with_nesterov, mom=1.0-one_minus_momentum, wd=wd, decouple_wd=use_wd, nesterov=use_nesterov)  # authors implementation used Adam (L2 reg), not AdamW (weight decay)

# filename = f"{dataset}_{model.name}_{run['sys/id'].fetch()}"
# learner_cbs = [
#                 unpack_model_output_dict(),
#                 gp_classifier_helper(),
#                 GradientClip(),
#                 SaveModelCallback(monitor='valid_loss', min_delta=2e-3, fname=filename, every_epoch=False, at_end=True, with_opt=True)
#                 #neptune_callback,
#                 ]

# authors implementation: Linearly scale learning rate and the decay epochs
# base_learning_rate = 0.08
# base_lr = base_learning_rate * batch_size / 128
# lr_decay_epochs = [60, 120, 160]

# lr_decay_epochs = [(start_epoch * num_epochs) // 200
#                        for start_epoch in lr_decay_epochs]
# lr_decay_ratio = 0.2
# lr_warmup_epochs = 1

# _pct_offs = 1. / (len(dls.train) * num_epochs)
# sched = {'lr': WarmUpPieceWiseConstantSchedStep(base_lr, lr_warmup_epochs, lr_decay_epochs, lr_decay_ratio, num_epochs, _pct_offs)}
# lr_scheduler = ParamScheduler(sched)
#learn = Learner(dls=dls, model=model, lr=base_lr, loss_func=loss_func, opt_func=opt_func, cbs=learner_cbs)
#print("Learning model")


#Normalising
#print("Normalising")
# means = [x.mean(dim=(0, 2, 3)) for x, y in dls.train]
# stds = [x.std(dim=(0, 2, 3)) for x, y in dls.train]
# mean = torch.stack(means).mean(dim=0)
# std = torch.stack(stds).mean(dim=0)
# print(mean, std)
#TensorImage([2.0458, 2.2209, 2.4332], device='cuda:0') TensorImage([0.8522, 0.8712, 0.8674], device='cuda:0') Quickdraw



# train the model
# with learn.no_bar():
#     learn.fit(num_epochs, cbs=lr_scheduler)

# log the split seed
# run[f"{neptune_callback.base_namespace}/config/train_valid_split_seed"] = str(train_valid_split_seed)
# run[f"{neptune_callback.base_namespace}/config/weights_seed"] = str(weights_seed)

# run.stop()