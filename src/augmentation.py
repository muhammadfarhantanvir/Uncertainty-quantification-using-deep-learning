from src.callback import  MixUpCallback
from src.loss import MixUpLoss, EnsembleClassificationLoss
from fastai.vision.all import *

class Augmentation:

    """Class used for data augmentations"""
    @staticmethod
    def process_augmentations(augmentations: List[Dict[str, Any]], learn: Learner) -> Learner:
        if augmentations is not None:
            for augment in augmentations:
                augment_name = augment['augmentation_name']
                augment_params = augment.get('augmentation_params', {})

                if augment_name == 'mixup':
                    print('Applying Mixup augmentation, this will use MixUpLoss as loss_func')
                    if isinstance(learn.loss_func, EnsembleClassificationLoss):
                        learn.loss_func = MixUpLoss(learn.loss_func)
                    else:
                        learn.loss_func = MixUpLoss(CrossEntropyLossFlat())
                    learn.cbs.append(MixUpCallback())
                else:
                    print(f"Unknown augmentation function: {augment_name}")
            return learn