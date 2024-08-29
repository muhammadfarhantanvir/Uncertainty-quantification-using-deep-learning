from fastai.vision.all import *

import sys
sys.path.append("./")

from src.architectures.convnext import ConvNeXt
from src.architectures.posteriornet import PosteriorNetwork
from src.optimizer import wrapped_partial
from src.loss import CrossEntropyClassificationLoss, UCELoss
from src.callback import PosteriorDensity, NormalizeAlphaPred, DataModCallback
from src.neptune_callback import NeptuneCallbackBayesianEnsemble as NeptuneCallback
from src.datasets import get_data_block
from src.net_utils import count_class_instances
from src.evaluation import eval_model
from src.callback import calc_dirichlet_uncertainty
from src.constants import DATASET_DIRECTORY, PROJECT_NAME, MODEL_PATH

import neptune
from neptune.types.mode import Mode as RunMode

num_classes = 10
batch_size = 64
lr = 5e-4
use_wd = False
wd = 3e-4
one_minus_momentum = 0.1
num_epochs = 200
rand_flip_prob = 0.5
train_valid_split_seed = 67280421310721
weights_seed = None
no_density = False
dataset = "cifar10"

# ConvNeXt parameters
net_params = {
    'in_chans': 3,
    'num_classes': 6,  # This is the output_dim for PosteriorNetwork
    'depths': [3, 3, 9, 3],
    'dims': [96, 192, 384, 768],
    'drop_path_rate': 0.1,
    'layer_scale_init_value': 1e-6,
    'head_init_scale': 1.,
    'kernel_size': 4,
    'stride': 2,  # Reduced from 4 to 2 for CIFAR-10
    'downsample_layers': 3,
    'stages': 4,
    'activation': nn.GELU,
    'include_head': True,
    'use_spec_norm': False,
    'spec_norm_iteration': 1
}

post_net_params = {
    'output_dim': num_classes,
    'latent_dim': net_params['output_dim'],
    'no_density': False,
    'density_type': 'radial_flow',
    'n_density': 6,
    'hidden_dim_class': net_params['dims'][-1],
    'k_lipschitz_class': None, 
    'init_weights': True,
}

device = torch.device('cuda:0')
torch.cuda.set_device(device)

torch.set_default_dtype(torch.float32)  # Changed to float32 for ConvNeXt

tgen = torch.Generator()
if train_valid_split_seed is not None:
    tgen = tgen.manual_seed(train_valid_split_seed)
else:
    train_valid_split_seed = tgen.seed()

wgen = torch.Generator(device=device)
if weights_seed is not None:
    wgen = wgen.manual_seed(weights_seed)
else:
    weights_seed = wgen.seed()

cifar10 = get_data_block(dataset, rand_flip_prob=rand_flip_prob, generator=tgen)

data_path = Path(DATASET_DIRECTORY)/dataset
dls = cifar10.dataloaders(data_path/"train", bs=batch_size, device=device)

num_class_samples = count_class_instances(dls.train_ds)

tags = [
    "AdamW" if use_wd else "Adam",
    "ConvNeXt"
]

run = neptune.init_run(
    source_files=["./src/train_convnext_PN_cifar.py",
                  "./src/processing.py",
                  "./src/optimizer.py",
                  "./src/loss.py",
                  "./src/callback.py", 
                  "./src/architectures/posteriornet.py",
                  "./src/architectures/convnext.py",
                  "./src/densities/MixtureDensity.py",
                  "./src/densities/NormalizingFlowDensity.py",
                  "./src/densities/BatchedNormalizingFlowDensity.py",
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

neptune_model = "CS24-CF10CNPN"

base_net = ConvNeXt(**net_params).to(dls.device)
model = PosteriorNetwork(encoder=base_net, generator=wgen, **post_net_params)
loss_func = CrossEntropyClassificationLoss(is_logits=True) if no_density else UCELoss(output_dim=num_classes)
opt_func = wrapped_partial(Adam, mom=1.0-one_minus_momentum, wd=wd, decouple_wd=use_wd)

neptune_callback = NeptuneCallback(run=run, 
                                   model_name=type(model).__name__, 
                                   model_params=post_net_params,
                                   net_name=type(base_net).__name__, 
                                   net_params=net_params,
                                   upload_saved_models=None)

run[f"{neptune_callback.base_namespace}/io_files/resources/dataset/name"] = dataset

filename = f"{dataset}_{model.name}_{run['sys/id'].fetch()}"
learner_cbs = [
                DataModCallback(),
                GradientClip(),
                SaveModelCallback(monitor='valid_loss', min_delta=2e-3, fname=filename, every_epoch=False, at_end=False, with_opt=True),
                neptune_callback,
                ]

if not no_density:
    learner_cbs.extend([PosteriorDensity(output_dim=post_net_params['output_dim'],
                                         num_class_samples=num_class_samples),
                        NormalizeAlphaPred(normalize_before_loss=False),])

path = MODEL_PATH if run._mode != RunMode.DEBUG else None

learn = Learner(dls=dls, model=model, lr=lr, loss_func=loss_func, opt_func=opt_func, cbs=learner_cbs, path=path)

with learn.no_bar():
    learn.fit(num_epochs)

run[f"{neptune_callback.base_namespace}/config/train_valid_split_seed"] = str(train_valid_split_seed)
run[f"{neptune_callback.base_namespace}/config/weights_seed"] = str(weights_seed)

# Evaluation code remains the same
items_test = cifar10.get_items(data_path/"test")
dl_test = dls.test_dl(items_test, rm_type_tfms=None, num_workers=0, with_labels=True, with_input=False)
dataset_test = dataset

log_fields = {
    "data/validation/name": dataset,
    "data/validation/size": learn.dls[1].n,
    "data/testing/name": dataset_test,
    "data/testing/size": dl_test.n,
}

uncertainty_callbacks = [calc_dirichlet_uncertainty()]

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
    reduce_preds=False,
    uncertainty_callbacks=uncertainty_callbacks,
)

run.stop()