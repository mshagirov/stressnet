---
title: stressnet
---

# stressnet documentation

Fine-tuning CNNs (ResNet18/34) on 16-bit multi-channel microscopy TIFFs to
predict mechanical stiffness (regression) and to classify age or
stiffness categories. It is **not tied to any sample type** — it works with any
2D multi-channel microscopy dataset that follows the layout in
[data.md](data.md). The repository is split into a reusable library
(`stressnet/`), core model/data modules at the root, and thin CLI drivers.

## Reading order

1. [configuration.md](configuration.md) — point the project at *your* data. **Start here.**
2. [data.md](data.md) — the dataset layout the code expects.
3. [training.md](training.md) — the available training drivers and their artefacts.
4. [models.md](models.md) — checkpoints, loading, and fine-tuning from a run.
5. [reproduction.md](reproduction.md) — recipes for reproducing each experiment family.
6. [architecture.md](architecture.md) — module map and why the code is organised this way.

Example agent skills for this project (documentation, not auto-loaded) live in
[skills/](skills/README.md).

## Entrypoints

### Commands

| Command | Purpose |
| --- | --- |
| `predict.py` | Config-driven prediction CLI; `--task auto\|regression\|classification`. |
| `main.py` | Deprecated alias for `predict.py`; kept for the old positional flags. |
| `train_age_classifier_v2.py` | Age classifier, 4 pretrained strategies (the exploration driver). |
| `train_age_classifier_v3.py` | Age classifier, staged gradual-unfreeze ladder (the winning recipe). |
| `train_categorised_classifier.py` | 2-way stiffness-category classifier (low/high). |
| `train_stiffness_classifier.py` | 4-way stiffness-category classifier. |
| `train_stiffness_classifier_v2.py` | 3-way stiffness-category classifier. |
| `train_stiffness_regression.py` | Scalar stiffness regression. |
| `check_cuda.py` | Print whether torch sees a CUDA device. |

### Library modules

| Module | Contents |
| --- | --- |
| `stressnet/config.py` | Site config: `dataset_config()`, `dataset_configs()`, `data_root()`, `save_root()`, `train_split()`, `prediction_split()`, `DATASET_KEYS`. |
| `stressnet/data.py` | `stratified_split`, `build_datasets`, `build_datasets_regression`, `build_loaders`, `TRAIN_TRANSFORMS`. |
| `stressnet/engine.py` | Label maps (`AGE_TO_IDX`, `CATEGORISED_TO_IDX`, `STIFFNESS_TO_IDX`, `STIFFNESS_V2_TO_IDX`), `set_backbone_trainable`, `build_optimizer`, `train_classifier`. |
| `stressnet/metrics.py` | `predict_classifier`, `predict_regression`, `accuracy`, `macro_accuracy`. |
| `stressnet/plots.py` | `plot_scatter`, `plot_history`, `plot_classifier_predictions`, `plot_regression_predictions`. |
| `stressnet/reporting.py` | `append_report` — append a run row to a `training_report*.csv`. |
| `dataset.py` | Every `Dataset` class (regression, age, and stiffness-category tasks). |
| `models.py` | `resnet18`, `resnet18_seq`, `ResNet18WithAgeLoc`, `TORCH_DEVICE`. |
| `classifiers.py` | `resnet18_classifier`, `resnet34_classifier`. |
| `transforms.py` | Input transforms Xv1–Xv4, target transforms Yv1–Yv3. |
| `trainer.py` | `train_model` — the regression training/validation loop. |
