---
title: Training
subtitle: "Training drivers and artefacts"
---

# Training

Every driver is a thin CLI on top of the `stressnet/` library. They share the
same recipe, so the same flags (`--datasets`, `--epochs`, `--freeze-epochs`,
`--batch-size`, `--val-frac`, `--report`, `--debug`) behave consistently.

## The common recipe

Unless noted otherwise, a run is a two-stage **gradual unfreeze**:

| Stage | Backbone | Head LR | Backbone LR | Epochs |
| --- | --- | --- | --- | --- |
| 1 | frozen | `1e-3` | — | `freeze_epochs` |
| 2 | trainable | `1e-4` (= head LR / 10) | `1e-5` | rest |

AdamW (`weight_decay=1e-4`), cosine annealing per stage, batch size 16, seed 42,
early stopping with patience 10. Classification uses plain `CrossEntropyLoss`
(no label smoothing), no MixUp and no dropout head; the winner found by
`train_age_classifier_v3.py` is exactly this plain recipe.

**Splits.** Training drivers split `Training_Data` internally (default 80/20,
`--val-frac`) for validation/early stopping, then score `Prediction_Data` once,
after training, as a `heldout` set that is never trained or checkpointed on. The
internal split is group- or category-stratified — see [data.md](data.md) for why
a random row-level split would leak sibling tiles.

**Artefacts** land under `save_root()` in a run directory named after the run.
Report rows are appended to a `training_report*.csv` in the current working
directory (the repository root). Report CSVs, `*.log`, `*.pt` and `out/` are
gitignored.

**Smoke tests.** Every driver has `--debug`: a tiny subset, one dataset and a
couple of epochs. Use it to validate wiring, then delete the `*_debug` run
directories.

> Seeds are set once per process, so re-running the same configuration in a later
> invocation can differ by roughly ±2–4% on the held-out accuracy.

## Driver reference

### `train_age_classifier_v2.py` — age, strategy exploration

Sweeps four ImageNet-pretrained strategies, including the anti-overfitting
regularisation (dropout head, label smoothing, MixUp, cosine schedule):

| Strategy | Transform | Recipe |
| --- | --- | --- |
| `frozen_pretrained` | Xv1 | Backbone frozen for the whole run; head only. |
| `gradual_unfreeze` | Xv1 | The common recipe (10 frozen epochs). |
| `regularized` | Xv1 | Full fine-tune with differential LR from epoch 0. |
| `mixup_augmented` | Xv4 | `regularized` + Xv4 strong augmentation + MixUp. |

- Flags: `--datasets`, `--strategies`, `--epochs`, `--batch-size`,
  `--val-frac`, `--report`, `--debug`.
- Run dir: `resnet18_classifier_<data>_Xv<1|4>_<strategy>` (`_debug` suffix).
- Artefacts: `<run>_<epochs>epoch_AdamW.pt` (best-val), `<run>_loss.png`,
  `<run>_preds.png`, `<run>_{train,val,heldout}_preds.csv`.
- Report: `training_report_v2.csv`.

### `train_age_classifier_v3.py` — age, staged ladder (best recipe)

The exploration ladder run one stage per CLI call; swept knobs are encoded in
the run name so stages never collide:

1. `--ls 0.1 0.0` — label smoothing on/off.
2. `--ls <best> --mixup 0 0.1 0.2 --epochs 40` — MixUp sweep.
3. `--ls <best> --mixup <best> --dropouts 0 0.2 0.5` — dropout sweep.

- Extra flags: `--ls`, `--mixup`, `--dropouts`, `--freeze-epochs`,
  `--no-nucleus`. Note `--epochs` defaults to **30**; the winning recipe used
  **40** (`--epochs 40`, 10 frozen + 30 unfrozen).
- Run dir:
  `resnet18_classifier_<data>_Xv1_gradual_unfreeze_<nolblsmooth|wlblsmooth>`
  `[_mx<NN>][_do<NN>][_fz<NN>][_nonuc][_debug]`.
- Artefacts: same weights/loss/preds plots, but only
  `<run>_{train,heldout}_preds.csv`.
- Report: `training_report_v3.csv`.

### `train_categorised_classifier.py` — stiffness category, 2-way

`low` vs `high` from `CategorisedStiffnessDataset` (single threshold). Uses the
best age recipe; the internal split is stratified by age `Group`.

- Flags: `--datasets`, `--arch {resnet18,resnet34}`, `--epochs`,
  `--freeze-epochs`, `--batch-size`, `--val-frac`, `--report`,
  `--stress-threshold`, `--balanced`, `--early-stopping`, `--save-best`,
  `--debug`.
- `--balanced` undersamples the majority category in the *train* partition to a
  50/50 split (val/heldout untouched).
- Run dir: `<arch>_<data>_CategorisedStiffness_<ddmmyy-HHMMSS>`.
- Artefacts: `train_params.yaml` (full config), `<run>_<epochs>epoch_AdamW.pt`
  (last weights, or best-val with `--save-best`), `<run>_loss.png`,
  `<run>_preds.png`, `<run>_{train,heldout}_preds.csv`.
- Report: `training_report_categorised.csv` (adds `heldout_macro_acc`).

### `train_stiffness_classifier.py` — stiffness category, 4-way

`adult_low`/`adult_high`/`young_low`/`young_high` from `AgeStiffnessDataset`.
Internal split stratified by `Category`.

- Flags: `--datasets`, `--epochs` (40), `--freeze-epochs` (10),
  `--batch-size`, `--val-frac`, `--report`, `--debug`.
- Run dir:
  `resnet18_stiffness_classifier_<data>_Xv1_gradual_unfreeze_nolblsmooth_mx00_do00_catstrat`
  (`_debug` suffix).
- Artefacts: weights + loss/preds plots + `<run>_{train,heldout}_preds.csv`.
- Report: `training_report_stiffness.csv`.

### `train_stiffness_classifier_v2.py` — stiffness category, 3-way

`young_low`/`medium`/`adult_high` from `AgeStiffnessDataset_v2`. Same flags and
artefacts as the 4-way driver.

- Run dir:
  `resnet18_stiffness_classifier_v2_<data>_Xv1_..._catstrat` (`_debug` suffix).
- Report: `training_report_stiffness_v2.csv`.

### `train_stiffness_regression.py` — scalar stiffness

ResNet18 scalar head, MSE loss, Yv1 target transform (log, then `(x-1.85)/3.0`).
Stage 1 is 15 frozen epochs, stage 2 adds 30 unfrozen epochs (45 total); the best
run on the *Prediction_Data* phase is kept via best-val checkpointing.

- Flags: `--dataset` (**single**, `datav5` default or `datav5.2`), `--epochs`,
  `--freeze-epochs`, `--batch-size`, `--report`, `--debug`.
- Run dir: `save_root()/stiffness/resnet18_<data>_Xv<1|4>_Yv1_gradual_unfreeze_fz<NN>_<ddmmyy-HHMMSS>`.
- Artefacts: `<run>_<epochs>epoch_AdamW.pt`, `<run>_loss.png`,
  `<run>_preds.png` (predicted-vs-true scatter with a dashed 1-to-1 line),
  `<run>_{train,val}_preds.csv`.
- Report: `training_report_regression.csv`.

## Reference results

Task-level outcomes depend on the specific dataset and are therefore not part
of the generic library. The recorded held-out numbers for the reference dataset
are kept in the gitignored `HELP.md`. Summary: the **age** task was learnable,
the **Nucleus** channel was required for it, and the **stiffness-category**
tasks stayed near the majority-class baseline regardless of balancing or
schedule length.

See also: [reproduction.md](reproduction.md) · [models.md](models.md) · [data.md](data.md)
