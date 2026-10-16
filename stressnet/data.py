'''Dataset and dataloader construction shared by the training drivers.

Every site-specific path (dataset roots, label filenames and split directory
names) is resolved through :mod:`stressnet.config` at *call* time, so importing
this module never requires a ``site_config.yaml``::

    from stressnet import config
    from stressnet.data import build_datasets, build_loaders

    cfg = config.dataset_config('datav5')     # {'root': Path, 'ch_names': ...}
    datasets = build_datasets(cfg, train_transform, val_frac=0.2)
    loaders = build_loaders(datasets, batch_size=16)

The internal train/val split is stratified by a column of the labels table
(``'Group'`` for age, ``'Category'`` for the stiffness categories). Rows are
tiles of larger samples, so a random row-level split would leak sibling tiles;
callers that need group-aware splitting should pass ``stratify_col`` or split
by sample prefix themselves. Prediction_Data is returned as ``'heldout'`` and
is never trained on.
'''

from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from dataset import AgeDataset, StiffnessDataset
from stressnet import config
from transforms import (
    data_transforms,
    data_transforms_inference,
    data_transforms_v4,
    stiffness_transform,
)

# Training-transform registry used by the classifier drivers (`--transform`
# strategies). Xv1 = centre-crop 256; Xv4 = strong augmentation for small sets.
TRAIN_TRANSFORMS = {'Xv1': data_transforms['train'], 'Xv4': data_transforms_v4['train']}


def stratified_split(groups: pd.Series, val_frac: float, seed: int = 42):
    '''Stratified split of row indices by `groups` (e.g. the Group column A/Y).'''
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for g in pd.unique(groups):
        idx = np.flatnonzero((groups == g).to_numpy())
        rng.shuffle(idx)
        n_val = round(len(idx)*val_frac)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return sorted(train_idx), sorted(val_idx)


def build_datasets(cfg: dict, train_transform, val_frac: float,
                   dataset_cls=AgeDataset, stratify_col: str = 'Group') -> dict:
    '''Internal stratified train/val split of Training_Data (+ held-out dataset).

    The val phase uses the internal val split (inference transforms);
    Prediction_Data is returned separately as 'heldout' and never trained on.
    dataset_cls allows swapping in an AgeDataset subclass (e.g.
    AgeDatasetNoNucleus, AgeStiffnessDataset) for all three splits.
    stratify_col selects which img_labels column the internal split is
    stratified by (default 'Group' = age; AgeStiffnessDataset exposes
    'Category' for its stiffness categories).

    `cfg` is a :func:`stressnet.config.dataset_config` mapping; the label file
    names and split directory names come from
    :func:`stressnet.config.train_split` / :func:`stressnet.config.prediction_split`.
    '''
    sub_dir, suffix, labels = config.train_split()
    common = {
        'ch_names': cfg['ch_names'],
        'ch_name_prefix': cfg['ch_name_prefix'],
        'ch_dir_suffix': suffix,
    }
    ds_train_tf = dataset_cls(labels, cfg['root']/sub_dir, transform=train_transform, **common)
    ds_infer = dataset_cls(labels, cfg['root']/sub_dir,
                           transform=data_transforms_inference, **common)

    train_idx, val_idx = stratified_split(ds_infer.img_labels[stratify_col], val_frac)
    print(f'Internal split ({stratify_col}-stratified): {len(train_idx)} train / '
          f'{len(val_idx)} val (of {len(ds_infer)} Training_Data samples)')
    print(repr(ds_train_tf))

    h_sub, h_suffix, h_labels = config.prediction_split()
    ds_heldout = dataset_cls(h_labels, cfg['root']/h_sub,
                             transform=data_transforms_inference,
                             ch_names=cfg['ch_names'],
                             ch_name_prefix=cfg['ch_name_prefix'],
                             ch_dir_suffix=h_suffix)
    print(repr(ds_heldout))

    return {
        'train': Subset(ds_train_tf, train_idx),
        'val': Subset(ds_infer, val_idx),
        'heldout': ds_heldout,
        # inference-transform versions of both splits, for final predictions
        'train_infer': Subset(ds_infer, train_idx),
        'val_infer': Subset(ds_infer, val_idx),
    }


def build_loaders(datasets: dict, batch_size: int) -> dict:
    return {
        phase: DataLoader(datasets[phase], batch_size=batch_size,
                          shuffle=(phase == 'train'), num_workers=4)
        for phase in ('train', 'val')
    }


def build_datasets_regression(cfg: dict, transforms_dict: dict) -> dict:
    '''Regression datasets: Training_Data -> 'train', Prediction_Data -> 'val'.

    `cfg` is a :func:`stressnet.config.dataset_config` mapping (``root`` and
    ``stiffness_col`` are used); `transforms_dict` supplies the image
    transforms for the 'train' and 'val' phases. Targets always go through the
    Yv1 ``stiffness_transform`` (log then (x-1.85)/3.0). Unlike the classifier
    datasets there is no internal split: the val phase *is* Prediction_Data.
    '''
    t_sub, t_suffix, t_labels = config.train_split()
    v_sub, v_suffix, v_labels = config.prediction_split()
    train_ds = StiffnessDataset(t_labels, cfg['root']/t_sub,
                                ch_dir_suffix=t_suffix,
                                stiffness_col=cfg['stiffness_col'],
                                transform=transforms_dict['train'],
                                target_transform=stiffness_transform)
    val_ds = StiffnessDataset(v_labels, cfg['root']/v_sub,
                              ch_dir_suffix=v_suffix,
                              stiffness_col=cfg['stiffness_col'],
                              transform=transforms_dict['val'],
                              target_transform=stiffness_transform)
    return {'train': train_ds, 'val': val_ds}
