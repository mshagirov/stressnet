---
name: stressnet-datasets
description: >
  Load, resolve and validate the image datasets used by stressnet-tests, and
  help point the project at or prepare a compatible dataset when it is missing
  or mis-formatted. Use when asked to load or check a dataset, when a dataset is
  missing or not found, to prepare a dataset, or to point the project at data.
  Triggers: load dataset, check dataset, dataset missing, dataset not found,
  prepare dataset, point at data.
---

# Loading and checking stressnet datasets

Datasets are referenced by **logical names** (`datav5`, `datav6`, `datav5.2`),
never by hardcoded paths. The mapping from a logical name to a physical folder
lives in `site_config.yaml` (gitignored) and is resolved by `stressnet.config`.
The real values for the current machine are documented in the gitignored
`HELP.md`.

Run everything from the repository root with `uv run python ...`.

## Step 1 — Resolve the dataset

Always start by resolving the logical name to a concrete root and printing it:

```bash
uv run python -m stressnet.config --show
uv run python -c "from stressnet.config import dataset_config; print(dataset_config('datav5')['root'])"
```

`dataset_config(name)` returns the resolved `root` plus the channel settings
(`ch_names`, `ch_name_prefix`, `stiffness_col`). Report the resolved root to the
user before doing anything else. Supported logical keys and their defaults:

| Key | Channels | Prefixes | Stiffness column |
| --- | --- | --- | --- |
| `datav5` | `Nucleus`, `Collagen` | `C1-`, `C2-` | `Stiffness` |
| `datav6` | `Nucleus`, `Collagen`, `Actin` | `C1-`, `C2-`, `C3-` | `Stiffness` |
| `datav5.2` | `Nucleus`, `Collagen` | `C1-`, `C2-` | `Mean stiffness` |

If resolution fails with a guided error, no site config exists yet — see Step 3.

## Step 2 — Validate the layout

Check each item and record pass/fail. The split directory names and label file
names come from `stressnet.config` (`train_split()` / `prediction_split()`);
their defaults are used below.

1. **Dataset root exists.**
2. **Split directories exist** for every channel:
   `Training_Data/<Channel>_Train/` and `Prediction_Data/<Channel>_Prediction/`.
3. **Label spreadsheets exist**: `Training_Data/Stiffness_Data_Training.xlsx` and
   `Prediction_Data/Stiffness_Data_Prediction.xlsx`.
4. **Channel image prefixes match**: files are named `<prefix><name>.tif`, with
   `C1-`/`C2-`/`C3-` matching the configured `channel_prefixes`.
5. **A sample actually loads** — optionally instantiate the dataset class and
   index one item. Use `StiffnessDataset` (regression) or `AgeDataset`
   (classification) from the shared data module:

   ```bash
   uv run python - <<'PY'
   from stressnet.config import dataset_config
   from stressnet.data import AgeDataset

   cfg = dataset_config('datav5')
   ds = AgeDataset(
       'Stiffness_Data_Training.xlsx',
       cfg['root'] / 'Training_Data',
       ch_names=cfg['ch_names'],
       ch_name_prefix=cfg['ch_name_prefix'],
       ch_dir_suffix='_Train',
   )
   image, target, fname = ds[0]
   print(f'OK: {len(ds)} samples, image {tuple(image.shape)}, target {target!r}, {fname}')
   PY
   ```

6. **Category datasets need thresholds.** `AgeStiffnessDataset`,
   `AgeStiffnessDataset_v2` and `CategorisedStiffnessDataset` load per-group
   thresholds from `stiffness_thresholds.yaml` at the **dataset root** (the
   parent of the split directory, i.e. `root/stiffness_thresholds.yaml`, not
   inside `Training_Data`). If it is missing they raise `FileNotFoundError`;
   confirm the file exists and holds `young:` / `adult:` values.

Summarise as a checklist, then report the exact commands you ran and their
output.

## Step 3 — If the dataset cannot be found

**Ask the user** where the data actually lives, then search likely locations
(mounted volumes, shared storage, a parent data directory) for the marker files
rather than guessing:

```bash
ls <candidate-mount-point>
find <candidate-dir> -maxdepth 3 -name 'Stiffness_Data_Training.xlsx' 2>/dev/null
```

Explain that the **physical folder name is configured in `site_config.yaml`**,
under `datasets.<name>.dir` (relative to `data_root`), or via an absolute
`root:`. The machine's real values are documented in the gitignored `HELP.md`.
Once the user tells you the location, either:

- edit `datasets.<name>.dir` (or add a `root:`), or
- create the config from scratch:

  ```bash
  uv run python -m stressnet.config --init \
      --data-root <datasets-dir> --save-root <saved-models-dir>
  ```

then re-run Step 1 and Step 2. If the user's data is missing entirely, say so
plainly — do not fabricate a path.

## Step 4 — If the data format does not match

Walk the user through preparing a compatible dataset. The layout the code
expects:

```
<dataset-dir>/
  stiffness_thresholds.yaml          # category tasks only
  Training_Data/
    Stiffness_Data_Training.xlsx
    <Channel>_Train/                 # one directory per channel
  Prediction_Data/
    Stiffness_Data_Prediction.xlsx
    <Channel>_Prediction/
```

- **Images** — single-channel **16-bit** TIFFs, one file per tile, named
  `<prefix><name>.tif` (e.g. `C1-<name>.tif` for the first channel). The loader
  scales values by `/65535` and normalises each image by its 99th percentile, so
  keep the raw 16-bit data.
- **Channels and directory suffixes** — the channel directory is
  `<Channel><suffix>` (`Nucleus_Train`, `Collagen_Train`, ...); the channel name,
  prefix and suffix are configurable via `datasets.<name>.channels` /
  `channel_prefixes` in `site_config.yaml`.
- **Labels** — the two `.xlsx` files above, each with at least `Stiffness` and
  `Image` columns (`Image` matches the `<name>` part of the TIFFs). For age work,
  add a `Group` column with `A` (adult) / `Y` (young). `Location`
  (`0`/`1`/`2`) is optional and parsed from a filename stem ending in
  `LV`/`IVS`/`RV`. Grid-averaged datasets use a `Mean stiffness` column instead
  of `Stiffness` (configure it with `stiffness_col`).
- **Category thresholds** — for the stiffness-category tasks, add
  `stiffness_thresholds.yaml` at the dataset root with per-group thresholds, e.g.
  `young: <value>` and `adult: <value>`.
- **Point the project at it** — set `datasets.<name>.dir` (or an absolute
  `root:`) in `site_config.yaml`, or run `--init` as in Step 3, adjusting
  `channels`, `channel_prefixes` and `stiffness_col` if the prepared data differs
  from the defaults.

Then **re-run validation** (Steps 1–2) and report the result.

## Step 5 — Report the outcome

Give the user a short pass/fail checklist covering: root resolved, both splits
present, both label files present, channel prefixes matched, sample indexed
successfully, thresholds present (category tasks). Include the exact commands
used and their raw output. If anything failed, state which step of the
preparation walkthrough resolves it. Never commit dataset paths: they belong in
`site_config.yaml` / `HELP.md`, both gitignored.
