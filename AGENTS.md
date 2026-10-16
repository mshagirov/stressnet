# AGENTS.md

Fine-tuning CNNs (ResNet18/34) on 16-bit multi-channel microscopy TIFFs for
cardiac stiffness regression and age / stiffness-category classification.
Python 3.11, uv, `torch==2.3.1` + `torchvision==0.18.1`. No tests, no CI.

## Setup & commands

Run everything from the repository root (paths resolve relative to CWD):

```bash
uv sync                       # install into .venv
uv run python <script>.py     # e.g. train_stiffness_regression.py --debug
uv run ruff check <files>     # lint (dev dependency)
uv run python check_cuda.py   # torch CUDA availability
```

Verify changes by importing the modules (`uv run python -c "import dataset,
models, transforms, classifiers"`) or running a driver with `--debug`.

## Repo layout

- `stressnet/` — shared library: `config.py`, `data.py`, `engine.py`,
  `metrics.py`, `plots.py`, `reporting.py`. Import from here; do not re-implement.
- Root modules — `dataset.py` (Dataset classes), `models.py`, `classifiers.py`,
  `transforms.py`, `trainer.py`.
- `train_*.py` — thin CLI drivers (flags/run-dir/report-CSV names are stable).
- `predict.py` — config-driven prediction; `main.py` is a deprecated alias.

## Private-data rule (the repo is public)

Never commit site-specific values: no absolute paths, no dataset folder names,
no private remotes, no collaborator names. Add them to the gitignored
`HELP.md`; wire paths through `site_config.yaml` (copy
`site_config.example.yaml`) or the `STRESSNET_DATA_ROOT` / `STRESSNET_SAVE_ROOT`
env vars. Resolve via `stressnet.config` (`dataset_config(key)`, `save_root()`,
…); inspect with `uv run python -m stressnet.config --show`. Dataset keys
(`datav5`, `datav6`, `datav5.2`) are logical; only their `dir` is site data.

## Dataset & leakage warnings

- Labels are `.xlsx` with `Stiffness` + `Image`; ages live in `Group` (`Y`/`A`);
  `Location` (`LV`/`IVS`/`RV`) is parsed from the filename stem.
- Images: 16-bit TIFFs, `/65535`, divided by their 99th percentile, clipped.
  Two-channel data gets a zero third plane so models always see 3 channels.
- Rows are tiles of larger samples. Row-level random splits leak sibling tiles
  (~100%); use `Group`- or `Category`-stratified splits (`stressnet.data`).
- Category datasets require `stiffness_thresholds.yaml` at the **dataset root**
  (`<root>/<dir>/`, one level above `Training_Data`); missing file → `FileNotFoundError`.
- Training drivers split `Training_Data` internally for early stopping and score
  `Prediction_Data` once as `heldout` (never trained/checkpointed on).

## Model & trainer quirks

- `TORCH_DEVICE` (models.py) is CUDA-if-available and is the default device for
  both `trainer.train_model` and `stressnet.engine.train_classifier`; each moves
  the model onto the device itself.
- `models.resnet18` expects `fc = Linear(512, 1)`; `models.resnet18_seq` expects
  an `fc` `Sequential`. Wrong loader → `load_state_dict` mismatch.
- `classifiers.resnet18_classifier(num_classes, weights_path=..., dropout=...)`:
  `dropout > 0` makes the head `Sequential(Dropout, Linear)` — reload with the
  same `num_classes` and `dropout`.
- torch / torchvision are pinned as a pair; bump together only.

## Docs

Start at [`docs/index.md`](docs/index.md). Topic docs: configuration, data,
training, models, reproduction, architecture (all under `docs/`).
