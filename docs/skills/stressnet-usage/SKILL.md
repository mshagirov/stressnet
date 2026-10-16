---
name: stressnet-usage
description: >
  Train, fine-tune, run experiments with, and report on the stressnet CNN
  models for cardiac stiffness. Use when asked to train or fine-tune a model
  (including fine-tuning on top of an existing checkpoint), run an experiment,
  find where trained models are saved, choose a model or run naming convention,
  or summarise results from the experiment reports. Triggers: train, fine-tune,
  fine-tune an existing model, run an experiment, where are my models, save
  models, model naming convention, report on experiments.
---

# Using stressnet: train, fine-tune, report

The repo fine-tunes ResNet18/34 models to (a) predict cardiac stiffness as a
scalar (regression) and (b) classify age (`young`/`adult`) or stiffness
categories from 16-bit TIFF microscopy channels (Nucleus/Collagen, optionally
Actin).

Training drivers live at the repository root (`train_*.py`); shared code lives in
`stressnet/` (`config.py`, `data.py`, `engine.py`, `metrics.py`, `plots.py`,
`reporting.py`). Run everything from the repository root with `uv run python ...`.

**Never hardcode dataset or save paths in code or scripts.** Every location is
resolved through `stressnet.config` (`dataset_config`, `save_root`), which reads
`site_config.yaml`. The real values for the current machine live in the
gitignored `HELP.md`.

Work through the steps below **in order**. Ask the user the questions marked
**Ask the user** — do not guess their answers.

## Step 1 — Ensure the site config is present

Run:

```bash
uv run python -m stressnet.config --show
```

This prints the resolved root of each logical dataset (`datav5`, `datav6`,
`datav5.2`) and `save_root`, or an unconfigured marker / a guided error.

If it is **unconfigured**:

1. **Ask the user** (interactively): "Which directory holds your datasets?" and
   "Where should trained models and run artefacts be written?"
2. Initialise the config with their answers:

   ```bash
   uv run python -m stressnet.config --init \
       --data-root <datasets-dir> --save-root <saved-models-dir>
   ```

   (Equivalently: copy `site_config.example.yaml` to `site_config.yaml` and edit
   it. Both files are at the repository root; `site_config.yaml` is gitignored.)
3. Edit the `datasets:` mapping so each logical key points at the user's actual
   folder (see the `stressnet-datasets` skill for validating the layout).
4. Re-run `--show` to confirm.
5. Tell the user that the machine-specific values belong in the gitignored
   `HELP.md`, and must never be committed to this public repository.

Environment variables `STRESSNET_DATA_ROOT` / `STRESSNET_SAVE_ROOT` (and
`STRESSNET_CONFIG` to point at another config file) override the YAML file.

## Step 2 — Detect existing models

**Ask the user**: "Do you already have trained checkpoints you want to reuse or
fine-tune?" If yes, **ask for the directory** they live in (the default is
`save_root()`).

List what is there and report the run names so they can be reused:

```bash
uv run python -c "from stressnet.config import save_root; print(save_root())"
ls -1 "$(uv run python -c 'from stressnet.config import save_root; print(save_root())')"
```

Checkpoints are `*.pt` files inside a per-run directory (each directory also
holds `*_loss.png`, `*_preds.png` and `*_preds.csv`). Record the location and the
run names in the conversation so later steps can reuse or fine-tune them. If the
user has no checkpoints, note that runs start from ImageNet-pretrained or random
weights.

## Step 3 — Agree a naming convention

The default convention is one directory per run:

```
save_root()/<arch>_<dataset>_<Xv>_<Yv>_<strategy>/
```

- `<arch>` — `resnet18` / `resnet34` (classifiers add `_classifier`).
- `<dataset>` — the logical key: `datav5`, `datav6` or `datav5.2`.
- `<Xv>` — input transform version (`Xv1`…`Xv4`).
- `<Yv>` — target transform version (`Yv1` scalar log-normalised stiffness,
  `Yv2` scaled stiffness); classification runs name the init/strategy instead.
- `<strategy>` — the training strategy, with knob tags added by the driver
  (e.g. `gradual_unfreeze_nolblsmooth_mx00_do00`, `..._catstrat`, `..._nonuc`).

Summarise the resolved convention with real values for the user's task (for
example `save_root(): <path>` → `resnet18_classifier_datav5_Xv1_gradual_unfreeze/`)
and **confirm before training**. The drivers construct this directory themselves;
fine-tuning always starts a **new** directory rather than overwriting an old one.

## Step 4 — Train or fine-tune

Pick the driver for the task:

| Task | Driver |
| --- | --- |
| Age `young`/`adult` (strategy sweep) | `train_age_classifier_v2.py` |
| Age (staged knob ladder) | `train_age_classifier_v3.py` |
| 2-way stiffness category (`low`/`high`) | `train_categorised_classifier.py` |
| 4-way stiffness category | `train_stiffness_classifier.py` |
| 3-way stiffness category | `train_stiffness_classifier_v2.py` |
| Scalar stiffness regression | `train_stiffness_regression.py` |

Key flags (all drivers accept `--help`):

- `--datasets <key> [...]` (regression uses `--dataset <key>`) — logical dataset(s).
- `--epochs`, `--batch-size`, `--val-frac` (internal Training_Data split).
- `--report <file>.csv` — where the run row is appended (default
  `training_report*.csv` in the working directory).
- `--debug` — fast smoke test on a small subset; run dirs get a `_debug` suffix.
  Delete `_debug` run directories afterwards.
- Age v3 extra knobs: `--ls`, `--mixup`, `--dropouts`, `--freeze-epochs`,
  `--no-nucleus`.
- Categorised extra knobs: `--arch`, `--balanced`, `--stress-threshold`,
  `--early-stopping`, `--save-best`.

Exact commands, all from the repository root:

```bash
# Age classifier — a strategy sweep (frozen_pretrained / gradual_unfreeze / regularized / mixup_augmented)
uv run python train_age_classifier_v2.py --datasets datav5 --strategies gradual_unfreeze --epochs 30

# Age classifier — one rung of the staged knob ladder (best known: plain gradual unfreeze)
uv run python train_age_classifier_v3.py --datasets datav5 --ls 0.0 --mixup 0.0 --dropouts 0.0 --epochs 40

# 2-way stiffness category (balance the training partition)
uv run python train_categorised_classifier.py --datasets datav5 --balanced

# 4-way stiffness categories
uv run python train_stiffness_classifier.py --datasets datav5

# 3-way stiffness categories
uv run python train_stiffness_classifier_v2.py --datasets datav5

# Scalar stiffness regression
uv run python train_stiffness_regression.py --dataset datav5.2
```

Smoke-test a driver first with `--debug` when iterating, e.g.
`uv run python train_stiffness_regression.py --dataset datav5.2 --debug`.

### Fine-tuning from an existing checkpoint

Start from the user's checkpoint (from Step 2) instead of ImageNet weights, and
write a **new** run directory (never overwrite the source run). Load it with the
matching loader so the head shape lines up:

- Regression (`Linear(512, 1)` head): `stressnet`/`models.resnet18(weights_path=...)`.
  Use `models.resnet18_seq` only for checkpoints whose head is a `Sequential`.
- Classification: `classifiers.resnet18_classifier(num_classes=<n>, weights_path=..., dropout=...)`.
  If the checkpoint was trained with `dropout > 0`, its head is
  `nn.Sequential(nn.Dropout, nn.Linear)` — pass the **same** `dropout` value when
  reloading, otherwise `load_state_dict` fails. `classifiers.resnet34_classifier`
  also exists.

Class-index mappings used by the heads: age `young=0, adult=1`; 2-way categories
`low=0, high=1`; 3-way `young_low=0, medium=1, adult_high=2`; 4-way
`adult_low=0, adult_high=1, young_low=2, young_high=3`.

For prediction/plots from an existing checkpoint, use `predict.py` (it takes the
weights, a data root, the labels spreadsheet, a channel directory suffix and a
phase). Confirm the exact flags with `uv run python predict.py --help`.

## Step 5 — Report on experiments

Each run appends one row to its `training_report*.csv` in the working directory:

| Driver | Report file |
| --- | --- |
| `train_age_classifier_v2.py` | `training_report_v2.csv` |
| `train_age_classifier_v3.py` | `training_report_v3.csv` |
| `train_categorised_classifier.py` | `training_report_categorised.csv` |
| `train_stiffness_classifier.py` | `training_report_stiffness.csv` |
| `train_stiffness_classifier_v2.py` | `training_report_stiffness_v2.csv` |
| `train_stiffness_regression.py` | `training_report_regression.csv` |

Read the reports (and the per-run `*_preds.csv`, `*_loss.png`, `*_preds.png`
under `save_root()`), then summarise per run: dataset, strategy/transform,
`best_val_loss`, final train/val accuracy or loss, and above all the **heldout**
score (`heldout_acc`; classification reports also carry `heldout_macro_acc`).
Rank runs by heldout performance and call out which run directory / weights file
each result came from.

Keep these retained findings in mind when interpreting results:

- **Age** classification works best with **plain gradual unfreeze** (no label
  smoothing, no MixUp, no dropout). The **Nucleus** channel is required —
  ablating it drops heldout accuracy to ≈ chance; collagen alone is not enough.
- **Stiffness-category** tasks (2-way, 3-way, 4-way) are **≈ chance / majority
  baseline** on the heldout split, regardless of balancing or longer schedules —
  treat near-baseline macro accuracy as expected, not as a bug.

## Step 6 — Keep paths out of code

Do not add dataset or save paths to any script. Resolve them through
`stressnet.config` (`dataset_config(name)`, `save_root()`); read the resolved
values with `uv run python -m stressnet.config --show`. Machine-specific values
go in `site_config.yaml` (gitignored) and are documented for humans in the
gitignored `HELP.md` — never in tracked code or docs.
