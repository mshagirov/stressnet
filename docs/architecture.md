---
title: Architecture
subtitle: "Module map and rationale"
---

# Architecture

## Module map

```text
stressnet/                shared, importable library
  config.py    site config resolution (no hardcoded paths)
  data.py      splits + dataset/loader builders + TRAIN_TRANSFORMS
  engine.py    label maps, optimizer/freezing, the unified classifier trainer
  metrics.py   prediction + accuracy helpers
  plots.py     plot_scatter / plot_history / plot_*_predictions
  reporting.py append rows to training_report*.csv

root modules              public, directly importable
  dataset.py   every Dataset class
  models.py    resnet18, resnet18_seq, ResNet18WithAgeLoc, TORCH_DEVICE
  classifiers.py  resnet18_classifier, resnet34_classifier
  transforms.py   Xv1..Xv4 input transforms, Yv1..Yv3 target transforms
  trainer.py      train_model — regression training loop

CLI drivers (thin)        train_age_classifier_v2.py, train_age_classifier_v3.py,
                          train_categorised_classifier.py,
                          train_stiffness_classifier.py,
                          train_stiffness_classifier_v2.py,
                          train_stiffness_regression.py
predict.py                config-driven prediction CLI
main.py                   deprecated alias for predict.py
```

## Data flow

```text
site_config.yaml ─┐
STRESSNET_* env ──┴─> stressnet.config.dataset_config(key)
                          │  root, ch_names, ch_name_prefix, stiffness_col
                          v
   dataset.py (Dataset classes) ──> 16-bit TIFF, /65535, p99-normalised
                          │
              stressnet.data.build_datasets / build_datasets_regression
                          │  train / val / heldout splits, loaders
                          v
      transforms.py (Xv input) ─> models.py | classifiers.py
                          │
      stressnet.engine.train_classifier  (classification)
      trainer.train_model                (regression)
                          │
      stressnet.metrics ─> stressnet.plots + stressnet.reporting
                          v
      save_root()/<run>/ (weights, plots, preds CSVs) + training_report*.csv
```

A driver only parses flags, picks the recipe, and wires these pieces together.
`--debug` substitutes small subsets; everything else is identical to a real run.

## Why this structure

**Drivers used to import other drivers.** Helpers such as `build_optimizer` and
`set_backbone_trainable` lived in `train_age_classifier_v2.py` and were imported
by the other training scripts. That coupled unrelated experiments to one
script's module-level configuration. Reusable logic now lives in `stressnet/`,
and the drivers import from the library instead of from each other.

**Four near-identical trainer copies collapsed into one.** The classification
stack had `classifier_trainer.py`, `classifier_trainer_v2.py` and
`classifier_trainer_categorised.py` / `classifier_trainer_stiffness*` that
differed only in how they mapped labels to class indices. There is now a single
`stressnet.engine.train_classifier`, parameterized by a label map
(`AGE_TO_IDX`, `CATEGORISED_TO_IDX`, `STIFFNESS_TO_IDX`, `STIFFNESS_V2_TO_IDX`).
The v1 trainers (`classifier_trainer.py`, `train_age_classifier.py`) were
deleted as superseded.

**No hardcoded paths.** Dataset roots, output roots and channel conventions are
resolved at runtime through `stressnet.config`, so the public repository carries
no site-specific values and the same code runs on any machine. See
[configuration.md](configuration.md).

**One prediction entrypoint.** `predict.py` is config-driven and dispatches on
`--task auto|regression|classification`; the old positional-flag CLI is retained
as the deprecated `main.py` alias. The `plot_finetuned.py` script was deleted
(it referenced a sibling checkout that does not exist here).

**Thin drivers keep the experiment surface unchanged.** Each driver's flags,
run-directory names and report CSVs are unchanged by the refactor, so existing
recipes and result files remain valid — see [training.md](training.md) and
[reproduction.md](reproduction.md).

See also: [index.md](index.md) · [data.md](data.md) · [models.md](models.md)
