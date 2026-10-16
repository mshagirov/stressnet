---
title: Reproduction
---

# Reproduction

Step-by-step recipes for each experiment family, using the **current** drivers.
Run everything from the repository root after `uv sync` and after
`site_config.yaml` is set up (see [configuration.md](configuration.md)).

Common notes:

- Every driver accepts `--debug` for a fast smoke test (tiny subset, few
  epochs). Use it to check wiring, then remove the `*_debug` run directories.
- Always pass a report path if you want a separate result file; the defaults are
  appended to in the repository root and are gitignored.
- Scores quoted below are held-out accuracy on `Prediction_Data`, scored once.
  Re-running the same config in a new process varies by roughly ±2–4%.

## 0. Configure

```bash
cp site_config.example.yaml site_config.yaml   # edit your paths
uv run python -m stressnet.config --show
```

## 1. Age classification (young vs adult)

### 1a. Strategy exploration — `train_age_classifier_v2.py`

```bash
# Smoke test first
uv run python train_age_classifier_v2.py --debug

# All 8 runs (2 datasets × 4 strategies)
uv run python train_age_classifier_v2.py

# A subset
uv run python train_age_classifier_v2.py --datasets datav5 \
    --strategies regularized mixup_augmented
```

Compare rows in `training_report_v2.csv` (column `heldout_acc`).

### 1b. Winning recipe — `train_age_classifier_v3.py`

The winner is **plain gradual unfreeze** with no label smoothing, no MixUp and
no dropout, so drive the ladder to that point explicitly:

```bash
uv run python train_age_classifier_v3.py --debug                 # smoke test

# Reproduce the winner
uv run python train_age_classifier_v3.py --ls 0.0 --mixup 0.0 --dropouts 0.0
# -> resnet18_classifier_datav5_Xv1_gradual_unfreeze_nolblsmooth_mx00_do00

# Longer schedule on the same config (no improvement observed)
uv run python train_age_classifier_v3.py --ls 0.0 --mixup 0.0 --dropouts 0.0 \
    --freeze-epochs 20 --epochs 100
```

The staged ladder itself (only needed to re-derive the winner):

```bash
uv run python train_age_classifier_v3.py                       # stage 1: ls 0.1 / 0.0
uv run python train_age_classifier_v3.py --ls <best> \
    --mixup 0 0.1 0.2 --epochs 40                              # stage 2
uv run python train_age_classifier_v3.py --ls <best> \
    --mixup <best> --dropouts 0 0.2 0.5                        # stage 3
```

### 1c. Nucleus ablation

```bash
uv run python train_age_classifier_v3.py --ls 0.0 --mixup 0.0 --dropouts 0.0 \
    --no-nucleus
# -> _nonuc run; on the reference dataset this collapsed to chance, i.e. the
#    Nucleus channel was required (recorded numbers are in HELP.md)
```

All v3 rows land in `training_report_v3.csv`.

## 2. Stiffness regression

```bash
uv run python train_stiffness_regression.py --debug                 # smoke test
uv run python train_stiffness_regression.py --dataset datav5        # tiled images, Xv1
uv run python train_stiffness_regression.py --dataset datav5.2      # grid-averaged variant, Xv4
```

Results append to `training_report_regression.csv`; checkpoints go to
`save_root()/stiffness/<run>/`. Compare `best_val_loss` across datasets.

## 3. Stiffness categories

All three use the best age recipe with a category-stratified internal split.

### 3a. 2-way (low/high) — `train_categorised_classifier.py`

```bash
uv run python train_categorised_classifier.py --debug
uv run python train_categorised_classifier.py                       # unbal, default
uv run python train_categorised_classifier.py --balanced            # 50/50 train split
uv run python train_categorised_classifier.py --balanced \
    --freeze-epochs 20 --epochs 100                                 # longer schedule
```

Outcome: ≈ majority baseline; macro accuracy ≈ chance regardless of balancing
or schedule. `training_report_categorised.csv` (`heldout_macro_acc`).

### 3b. 4-way — `train_stiffness_classifier.py`

```bash
uv run python train_stiffness_classifier.py --debug
uv run python train_stiffness_classifier.py
```

Outcome: ≈ majority baseline; `adult_high` never predicted.
`training_report_stiffness.csv`.

### 3c. 3-way — `train_stiffness_classifier_v2.py`

```bash
uv run python train_stiffness_classifier_v2.py --debug
uv run python train_stiffness_classifier_v2.py
```

Outcome: collapses to the majority `medium` class.
`training_report_stiffness_v2.csv`.

## 4. Predict with a produced run

```bash
uv run python predict.py --help
# logical dataset (resolved via site_config.yaml):
uv run python predict.py --task auto --dataset datav5 \
    --phase prediction --weights <run>/<weights>.pt
# or raw paths (the old main.py flags):
uv run python predict.py --weights <run>/<weights>.pt \
    -i <split_root> -l <labels.xlsx> -c _Prediction
```

`--task` selects `auto`, `regression` or `classification`. For each run the
driver writes `<phase>_preds.csv`; the regression drivers also write a
predicted-vs-true scatter image, and the legacy `main.py` writes a plotly
`<phase>_scatter.html`.

## Removed v1 stack

The superseded `train_age_classifier.py` (v1 age trainer) and
`plot_finetuned.py` have been **deleted**. Use `train_age_classifier_v2.py` /
`train_age_classifier_v3.py` above; results from the v1 era are not
reproducible from this repository.

See also: [training.md](training.md) · [models.md](models.md)
