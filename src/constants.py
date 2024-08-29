from typing_extensions import Literal

PROJECT_NAME = 'tud-ls9/case-study-SS24'
DATASET_DIRECTORY = '/home/pgshare/shared/datasets'
DOMAINNET_DIRECTORY = '/home/pgshare/shared/datasets/domainnet'
MODEL_PATH = "/home/pgshare/shared/CS24"

######### uncertainty methods
REPULSIVE_ENSEMBLE = "BE"
SNGP_CONSTANT = "SNGP"
POSTERIOR_NETWORK = "PN"

######### activation methods
ACTIVATION_METHODS = Literal["RELU", "SELU", "GELU"]

ALLOWED_NEPTUNE_MODES = Literal["debug", "async"]

DATASETS = Literal[
    "domainnet", "cifar10", "cifar10_2", "dr512", "dr250", "bgraham_dr", "dr_original", "HAM10000", "HAM10000_2", "bgraham_dr_as_OOD",
    "domainnet/real","domainnet/infograph","domainnet/clipart","domainnet/sketch","domainnet/painting","domainnet/quickdraw"]

LOSSES = Literal["EnsembleClassificationLoss",
                 "CrossEntropyClassificationLoss",
                 "UCELoss",
                 "WeightedEnsembleClassificationLoss",
                 "WeightedCrossEntropyClassificationLoss",
                 "WeightedUCELoss",
                 "FocalEnsembleClassificationLoss",
                 "FocalCrossEntropyClassificationLoss",
                 "FocalUCELoss"
                 ]

BASE_NETWORKS = Literal["ConvNet", "WideResNet", "ConvLinSeq", "ConvNeXt", "ConvNeXtSNGP"]

EVALUATION_CALLBACKS = Literal["calc_validation_total_entropy","calc_sngp_uncertainty","calc_dirichlet_uncertainty"]

LR_SCHEDULERS = Literal["CosineDecayWithWarmup","WarmUpPieceWiseConstantSchedStep","FitOneCycle"]

AUGMENTATIONS = Literal["mixup"]
