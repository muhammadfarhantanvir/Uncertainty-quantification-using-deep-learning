import warnings
from fastai.vision.all import *
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms import transforms
import pandas as pd
import os
from src.processing import RandomSplitter
from typing import Optional, List, Tuple, Dict
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import LabelEncoder
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter, OrderedDict


def get_image_data_block(
        item_tfms: List[Transform],
        batch_tfms: List[Transform],
        valid_pct: float = 0.2,
        useIndexSplitter: bool = False,
        include_classes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
        generator: Optional[torch.Generator] = None,
        valid_indices: Optional[torch.Tensor] = None
) -> DataBlock:
    def get_filtered_image_files(path: str) -> List[Path]:
        try:
            items = get_image_files(path)
            if include_classes:
                items = [item for item in items if parent_label(item) in include_classes]
            if exclude_classes:
                items = [item for item in items if parent_label(item) not in exclude_classes]
            return items
        except Exception as e:
            warnings.warn(f"Error in get_filtered_image_files: {str(e)}")
            return []

    splitter = RandomSplitter(valid_pct=valid_pct, generator=generator)
    if useIndexSplitter:
        if valid_indices is None:
            raise ValueError("`valid_indices` cannot be None while `useIndexSplitter==True`!")
        splitter = IndexSplitter(valid_idx=valid_indices)
    
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_filtered_image_files,
        splitter=splitter,
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )

def get_labels_from_csv(*csv_paths: str) -> Dict[str, str]:
    label_dict = {}
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            df['image'] = df['image'].str.strip().apply(lambda x: x if x.endswith('.jpeg') else f"{x}.jpeg")
            label_dict.update(df.set_index('image')['level'].to_dict())
        except Exception as e:
            warnings.warn(f"Error reading CSV file {csv_path}: {str(e)}")
    return label_dict

def get_dr_image_data_block(batch_tfms:list, valid_pct:float=0.2, useIndexSplitter:bool=False, generator: Optional[Generator] = None, label_dict=None, valid_indices: Optional[torch.Tensor] = None):
    def label_func(filepath): return label_dict.get(os.path.basename(filepath))

    splitter=RandomSplitter(valid_pct=valid_pct, generator=generator)
    if useIndexSplitter:
        if valid_indices is None:
            raise ValueError("`valid_indices` cannot be None while `useIndexSplitter==True`!")
        splitter = IndexSplitter(valid_idx=valid_indices)
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=splitter,
        get_y=label_func,
        batch_tfms=batch_tfms
    )

def get_data_block(ds_name: str,
                   valid_pct: float = 0.2,
                   norm_stats: tuple | None = None,
                   do_normalize: bool = True,
                   use_random_flip: bool = False,
                   use_random_erasing: bool = False,
                   use_randaugment: bool = False,
                   rand_flip_prob: float = 0.5,
                   random_erasing_prob: float = 0.1,
                   include_classes: Optional[List[str]] = None,
                   exclude_classes: Optional[List[str]] = None,
                   **kwargs):
    """Factory function returning data blocks for a number of datasets."""
    _item_tfms = []
    _batch_tfms = []

    if use_random_flip:
        _item_tfms.append(FlipItem(p=rand_flip_prob))

    if use_random_erasing:
        random_erasing = RandomErasing(p=random_erasing_prob, sl=0.03, sh=0.08,max_count=2)
        _batch_tfms.append(random_erasing)

    if use_randaugment:
        aug_tfms = aug_transforms(
            do_flip=True,
            flip_vert=False,
            max_rotate=15.0,
            min_scale=0.75,
            max_zoom=1.1,
            pad_mode='reflection',
            p_affine=0.7,
            p_lighting=0.6
        )
        _batch_tfms.extend(aug_tfms)

    if ds_name == "cifar10":
        _stats = norm_stats if norm_stats is not None else ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        if do_normalize:
            _batch_tfms.append(Normalize.from_stats(*_stats))

        return get_image_data_block(item_tfms=_item_tfms,
                                    batch_tfms=_batch_tfms,
                                    valid_pct=valid_pct,
                                    **kwargs)

    elif ds_name == "cifar10_2":
        _stats = norm_stats if norm_stats is not None else ((0.5133, 0.4973, 0.4619), (0.2650, 0.2616, 0.2748))
        if do_normalize:
            _batch_tfms.append(Normalize.from_stats(*_stats))

        return get_image_data_block(item_tfms=_item_tfms,
                                    batch_tfms=_batch_tfms,
                                    valid_pct=valid_pct,
                                    **kwargs)

    elif ds_name == "dr512":
        train_labels_path = kwargs.pop('train_labels_path')
        test_labels_path = kwargs.pop('test_labels_path', None)
        _stats = norm_stats if norm_stats is not None else ((0.3724, 0.2589, 0.1853), (0.2423, 0.1687, 0.1208))
        if do_normalize:
            _batch_tfms.append(Normalize.from_stats(*_stats))
        label_dict = get_labels_from_csv(train_labels_path, test_labels_path) if test_labels_path else get_labels_from_csv(train_labels_path)

        return get_dr_image_data_block(batch_tfms=_batch_tfms,
                                       valid_pct=valid_pct,
                                       label_dict=label_dict,
                                       **kwargs)

    elif ds_name == "dr250":
        train_labels_path = kwargs.pop('train_labels_path')
        test_labels_path = kwargs.pop('test_labels_path', None)
        _stats = norm_stats if norm_stats is not None else ((0.3205, 0.2246, 0.1616), (0.2609, 0.1823, 0.1320))
        if do_normalize:
            _batch_tfms.append(Normalize.from_stats(*_stats))
        label_dict = get_labels_from_csv(train_labels_path, test_labels_path) if test_labels_path else get_labels_from_csv(train_labels_path)

        return get_dr_image_data_block(batch_tfms=_batch_tfms,
                                    valid_pct=valid_pct,
                                    label_dict=label_dict,
                                    **kwargs)

    elif "domainnet" in ds_name:
        _item_tfms.append(Resize((256, 256)))
        mean, std = norm_stats if norm_stats is not None else ((2.0460, 2.2211, 2.4334), (0.8526, 0.8716, 0.8677))  # Quickdraw
        if do_normalize:
            _batch_tfms.append(Normalize.from_stats(mean, std))

        return get_image_data_block(item_tfms=_item_tfms,
                                    batch_tfms=_batch_tfms,
                                    valid_pct=valid_pct,
                                    include_classes=include_classes,
                                    exclude_classes=exclude_classes,
                                    **kwargs)

    elif ds_name == "HAM10000":
        normalization_stats = ([0.7622, 0.5465, 0.5707], [0.1413, 0.1520, 0.1688])
        if do_normalize:
            _batch_tfms.append(Normalize.from_stats(*normalization_stats))

        return get_image_data_block(item_tfms=_item_tfms,
                                    batch_tfms=_batch_tfms,
                                    valid_pct=valid_pct,
                                    **kwargs)

    elif ds_name == "HAM10000_2":
        mean, std = norm_stats if norm_stats is not None else ([0.7636, 0.5462, 0.5701], [0.0890, 0.1163, 0.1295])
        if do_normalize:
            _batch_tfms.append(Normalize.from_stats(mean, std))

        return get_image_data_block(item_tfms=_item_tfms,
                                    batch_tfms=_batch_tfms,
                                    valid_pct=valid_pct,
                                    **kwargs)

    elif ds_name == "bgraham_dr":
        train_labels_path = kwargs.pop('train_labels_path')
        test_labels_path = kwargs.pop('test_labels_path', None)
        _stats = norm_stats if norm_stats is not None else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if do_normalize:
            _batch_tfms.append(Normalize.from_stats(*_stats))
        label_dict = get_labels_from_csv(train_labels_path, test_labels_path) if test_labels_path else get_labels_from_csv(train_labels_path)

        return get_dr_image_data_block(batch_tfms=_batch_tfms,
                                        valid_pct=valid_pct,
                                        label_dict=label_dict,
                                        **kwargs)

    else:
        warn(f"Unknown dataset name: {ds_name}")

# please note that this is only designed for ham10000_2 and of course include ham10000, 
# only for imbalanced datasets, but also you do not have to do this
# cannot promise the best version, need do more experiments 
def mixed_sampling(dls, label_from_folder=True, sampling_strategy='auto'):
    print("Mixed sampling called")
    items = dls.train_ds.items
    
    if label_from_folder:
        labels = [item.parent.name for item in items]
    else:
        labels = dls.train_ds.y
   
    class_distribution = Counter(labels)

    # please note that the following is alao not necessary
    # I just want to have more samples for more important category
    # the ratio and values may also not proper, but it is also not that important for final result, 
    severity_factor = {
        'akiec': 1.3,
        'bcc': 1.2,
        'bkl': 1.0,
        'df': 1.0,
        'mel': 1.5,
        'nv': 1.0,
        'vasc': 1.0
    }
    severity_factor = {cls: severity_factor.get(cls, 1.0) for cls in class_distribution.keys()}
    
    def get_target_count(current_count, cls):
        base_count = (
            min(current_count * 2, 200) if current_count < 100
            else min(int(current_count * 1.5), 800) if current_count < 500
            else min(int(current_count * 1.3), 2000)
        )
        return int(base_count * severity_factor[cls])
    
    if sampling_strategy == 'auto':
        sampling_strategy = {cls: get_target_count(count, cls)
                             for cls, count in class_distribution.items()}
    
    item_array = np.arange(len(items)).reshape(-1, 1)
    
    majority_class = max(class_distribution, key=class_distribution.get)
    under_sampling_strategy = {majority_class: min(sampling_strategy[majority_class], 
                                                   int(class_distribution[majority_class] * 0.8))}
    under_sampler = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=42)
    item_array, labels = under_sampler.fit_resample(item_array, labels)
    
    current_distribution = Counter(labels)
    oversampling_strategy = {cls: max(count, sampling_strategy[cls])
                             for cls, count in current_distribution.items()
                             if count < sampling_strategy[cls]}
    
    if oversampling_strategy:
        k_neighbors = max(5, min(10, min(current_distribution.values()) - 1))
        oversampler = SMOTE(sampling_strategy=oversampling_strategy, k_neighbors=k_neighbors, random_state=42)
        resampled_indices, resampled_labels = oversampler.fit_resample(item_array, labels)
    else:
        resampled_indices, resampled_labels = item_array, labels
    
    new_items = [items[i[0]] for i in resampled_indices]
    
    sorted_vocab = sorted(dls.vocab)
    label_mapping = {label: idx for idx, label in enumerate(sorted_vocab)}
    resampled_labels = [label_mapping[label] for label in resampled_labels]
    
    dls.train_ds.items = new_items
    dls.train_ds.y = resampled_labels
    dls.vocab = sorted_vocab
    dls.train.n = len(new_items)
    dls.train.create_batches(dls.train_ds)
    dls.c = len(dls.vocab)
    
    final_distribution = Counter(resampled_labels)
    print("Sample distribution after mixed sampling:")
    for label, count in sorted(final_distribution.items()):
        print(f"{dls.vocab[label]}: {count}")

    return dls