# stressnet-tests

Fine-tuning CNNs (ResNet18/34) on 16-bit TIFF microscopy channels (Nucleus,
Collagen, optionally Actin) to:

- **regress** cardiac mechanical stiffness (scalar output), and
- **classify** age (young/adult) or stiffness categories (low/high, 3-way, 4-way).

Python 3.11, managed with [uv](https://docs.astral.sh/uv/), `torch==2.3.1` +
`torchvision==0.18.1`. There is no test suite and no CI; scripts are verified by
running them against real data.

> **No paths are hardcoded.** Dataset and output locations are resolved from a
> gitignored `site_config.yaml` (or environment variables) through
> `stressnet.config`. The public repository contains no machine- or
> data-specific values.

## Install

```bash
uv sync                       # creates .venv and installs pinned deps
uv run python check_cuda.py   # optional: confirm torch sees a GPU
```

## Quick start

**1. Configure** (once per machine), pointing at *your* data:

```bash
cp site_config.example.yaml site_config.yaml   # then edit your paths
uv run python -m stressnet.config --show       # verify resolution
```

See [docs/configuration.md](docs/configuration.md) for the full schema.
Reproducing the **original project** with its specific datasets, thresholds and
toolchain on a shared machine? The exact paths live in the gitignored
`HELP.md` (not in the public repo) — that file is **optional** and only needed
for that reproduction; day-to-day use of the library only needs
`site_config.yaml`.

**2. Train** (a fast smoke test, then a real run):

```bash
uv run python train_stiffness_regression.py --debug       # smoke test
uv run python train_stiffness_regression.py --dataset datav5
```

**3. Predict** with the checkpoint the run wrote under `save_root()`:

```bash
uv run python predict.py --help                # authoritative flag list
uv run python predict.py --task auto --dataset datav5 \
    --phase prediction --weights <run>/<weights>.pt
```

`predict.py` is config-driven: it resolves the dataset layout, label file and
channel conventions from `site_config.yaml`, so unlike the legacy CLI you do not
pass data directories, label files or channel suffixes by hand.

## Verify your install

`scripts/selfcheck.py` needs **no private data**: it writes a small synthetic
dataset (16-bit TIFFs, labels spreadsheets, `stiffness_thresholds.yaml`) into a
temporary directory and points `STRESSNET_CONFIG` at a throw-away config for
it. It then checks the whole stack — module imports, every `Dataset` class, the
train/val/held-out split builders, one classifier and one regression epoch, and
the `train_stiffness_regression.py` / `train_age_classifier_v2.py` / `predict.py`
CLIs as subprocesses — and prints `PASS`/`FAIL` per check:

```bash
uv run python scripts/selfcheck.py          # ~30 s on GPU, ~1 min on CPU
uv run python scripts/selfcheck.py --keep   # keep the temp tree for inspection
```

The exit code is 0 only if every check passes, so it doubles as a CI gate. All
artefacts stay inside the temporary tree (removed on success *and* failure);
nothing is written into the repository or your `site_config.yaml`.

## Repository layout

```text
stressnet-tests/
├── stressnet/            # shared library (importable)
│   ├── config.py         #   site config: dataset_config(), save_root(), ...
│   ├── data.py           #   splits, datasets/loaders builders, transforms
│   ├── engine.py         #   label maps, optimizer, unified classifier trainer
│   ├── metrics.py        #   prediction + accuracy helpers
│   ├── plots.py          #   matplotlib/plotly plotting helpers
│   └── reporting.py      #   append rows to training_report*.csv
├── dataset.py            # Dataset classes (all tasks)
├── models.py             # resnet18, resnet18_seq, ResNet18WithAgeLoc
├── classifiers.py        # resnet18_classifier, resnet34_classifier
├── transforms.py         # Xv1..Xv4 input transforms, Yv1..Yv3 target transforms
├── trainer.py            # regression training loop (train_model)
├── train_*.py            # thin CLI training drivers
├── predict.py            # config-driven prediction CLI
├── main.py               # deprecated alias for predict.py
├── scripts/
│   └── selfcheck.py      # synthetic-data self-check (no private data needed)
├── site_config.example.yaml  # tracked template — copy to site_config.yaml
├── HELP.md               # gitignored: site-specific values on your machine
├── LICENSE               # MIT
└── docs/                 # documentation (start at docs/index.md)
```

## Documentation

- [docs/index.md](docs/index.md) — overview, reading order, every entrypoint
- [docs/configuration.md](docs/configuration.md) — `site_config.yaml`, env vars, `HELP.md`
- [docs/data.md](docs/data.md) — dataset layout, labels, normalisation, thresholds
- [docs/training.md](docs/training.md) — every training driver, flags, artefacts, results
- [docs/models.md](docs/models.md) — checkpoints, loading, and fine-tuning from a run
- [docs/reproduction.md](docs/reproduction.md) — step-by-step experiment recipes
- [docs/architecture.md](docs/architecture.md) — module map and refactor rationale

## License

Released under the [MIT License](LICENSE).
