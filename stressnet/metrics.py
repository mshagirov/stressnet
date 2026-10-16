'''Prediction and metric helpers shared by the training drivers.

``predict_classifier`` generalizes the per-driver ``predict_phase*`` helpers
(age, categorised stiffness, 4-way and 3-way stiffness): it writes one row per
sample with the label string, the class indices, whether the prediction is
correct and one softmax probability column ``p_<class name>`` per class. For
the 2-class age task this keeps the historical ``p_adult`` column (class order
is ``['young', 'adult']``).

``predict_regression`` returns the raw target/prediction arrays for the scalar
stiffness model (Yv1 normalised units).
'''

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def predict_classifier(model: torch.nn.Module, loader, device: torch.device,
                       class_names, desc: str = '') -> pd.DataFrame:
    '''Run a classifier over `loader`; one row per sample.

    `class_names` lists the class labels in model output order; every row gets
    ``Image`` (filename), ``label`` (label string), ``y_true``, ``pred``,
    ``correct`` and ``p_<name>`` for each class.
    '''
    class_names = list(class_names)
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, labels, fnames in tqdm(loader, desc=desc):
            probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            for fname, label, p in zip(fnames, labels, probs, strict=True):
                label = str(label)
                y_true = class_names.index(label)
                pred = int(p.argmax())
                row = {
                    'Image': fname, 'label': label, 'y_true': y_true,
                    'pred': pred, 'correct': int(y_true == pred),
                }
                row.update({f'p_{name}': p[i] for i, name in enumerate(class_names)})
                rows.append(row)
    return pd.DataFrame(rows)


def predict_regression(model: torch.nn.Module, loader, device: torch.device,
                       desc: str = '') -> dict:
    '''Run the scalar regression model over `loader`; returns y_tgt / y_pred arrays.'''
    model.eval()
    y_tgt, y_pred = [], []
    with torch.inference_mode():
        for x, y in tqdm(loader, desc=desc):
            y_pred.append(model(x.to(device)).cpu().numpy())
            y_tgt.append(y.numpy())
    return {'y_tgt': np.concatenate(y_tgt), 'y_pred': np.concatenate(y_pred)}


def accuracy(df: pd.DataFrame) -> float:
    '''Overall accuracy of a :func:`predict_classifier` frame.'''
    return float(df['correct'].mean())


def macro_accuracy(df: pd.DataFrame, n_classes: int) -> float:
    '''Mean per-class accuracy (classes absent from `df` are skipped).

    Matches the historical ``heldout.groupby('category')['correct'].mean().mean()``
    computation while making the expected class count explicit.
    '''
    per_class = df.groupby('y_true')['correct'].mean().reindex(range(n_classes))
    return float(per_class.mean())
