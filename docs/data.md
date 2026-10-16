---
title: Data
---

# Data

## Expected layout

Every logical dataset key in `site_config.yaml` points at a folder with one
train split and one prediction split:

```text
<data_root>/<dir>/
├── stiffness_thresholds.yaml          # required by the category datasets
├── Training_Data/
│   ├── Stiffness_Data_Training.xlsx    # labels
│   ├── Nucleus_Train/    C1-<image>.tif
│   └── Collagen_Train/   C2-<image>.tif
└── Prediction_Data/
    ├── Stiffness_Data_Prediction.xlsx  # labels
    ├── Nucleus_Prediction/    C1-<image>.tif
    └── Collagen_Prediction/   C2-<image>.tif
```

- Channel directories are `<Channel><suffix>` (`_Train` / `_Prediction`).
- Image files are `<prefix><name>.tif`, with prefixes `C1-`, `C2-`, `C3-`.
- The split directory names, channel suffixes and label filenames are
  configurable in `site_config.yaml`; the defaults shown above are the
  `stressnet.config.DEFAULT_SPLITS` values.

A three-channel dataset (Actin) adds `Actin_Train/`/`Actin_Prediction/` with
`C3-` files.

## Labels

The labels spreadsheet (`.xlsx`) must contain:

| Column | Meaning |
| --- | --- |
| `Image` | Image name, with or without a channel prefix (prefixes are stripped on load). |
| `Stiffness` | Target stiffness. `datav5.2` reads a different source column name (`Mean stiffness`). |
| `Group` | Age group, `Y` (young) or `A` (adult). |

`Location` (`0`/`1`/`2`) is not read from the spreadsheet: it is parsed from the
filename stem, which ends in `LV`, `IVS` or `RV`. The sample ID is the stem
before the first `_`; all rows sharing it are tiles of the same physical sample.

> **Leakage warning.** Rows are tiles of larger samples (2–27 tiles per sample),
> and the train and prediction splits share no sample. A random *row-level*
> split therefore leaks sibling tiles into both sides (measured ~100%). Use a
> group-aware or category-stratified split — that is what the internal
> `stressnet/data.py::stratified_split` does (on `Group` by default, or on
> `Category` for stiffness tasks).

## Image loading and normalisation

- Images are 16-bit TIFFs, scaled by `/65535`.
- The result is divided by its own **99th percentile**, then clipped to `[0, 1]`.
- Two-channel datasets get a zero-filled third plane appended, so every model
  receives a 3-channel tensor regardless of the dataset key.
- No ImageNet mean/std normalisation is applied.

## `stiffness_thresholds.yaml`

The stiffness-category datasets (`AgeStiffnessDataset`,
`AgeStiffnessDataset_v2`, `CategorisedStiffnessDataset`) read per-age-group
thresholds from a file at the **dataset root** — i.e.
`<data_root>/<dir>/stiffness_thresholds.yaml`, one level above `Training_Data`:

```yaml
stiffness_threshold:
  young: <young cut-off>
  adult: <adult cut-off>
```

The actual numbers come from your labels (see the dataset's own
`stiffness_thresholds.yaml`; the reference project's values are in `HELP.md`).

The file is mandatory for those datasets; if it is missing they raise
`FileNotFoundError` with the full expected path. The category definitions are:

| Dataset | Categories |
| --- | --- |
| `CategorisedStiffnessDataset` (2-way) | `low` if `stiffness <= (young+adult)/2`, else `high` (override with `--stress-threshold`). |
| `AgeStiffnessDataset_v2` (3-way) | `young_low` (young & `<= young`), `adult_high` (adult & `> adult`), else `medium`. |
| `AgeStiffnessDataset` (4-way) | `<group>_<low\|high>`, `high` iff `stiffness >= thresholds[group]`. |

Regression and age classification do not need this file.

## Pointing at your own dataset

1. Put the folder anywhere and add it under `datasets:` in `site_config.yaml`:

   ```yaml
   datasets:
     mydata:
       dir: my_dataset_folder          # or an absolute root: `/path/to/folder`
       channels: [Nucleus, Collagen]
       channel_prefixes: [C1-, C2-]
       stiffness_col: Stiffness
   ```

2. Make sure the folder matches the layout above and contains the label
   spreadsheets and `stiffness_thresholds.yaml` if you will train categories.
3. Verify:

   ```bash
   uv run python -m stressnet.config --show
   ```

4. Use the key with the drivers, e.g.
   `uv run python train_stiffness_regression.py --dataset mydata`.

New logical keys can be added freely; `DATASET_KEYS` only lists the keys the
shipped defaults describe. Any key present under `datasets:` in your config is
selectable on the CLI.

See also: [configuration.md](configuration.md) · [training.md](training.md)
