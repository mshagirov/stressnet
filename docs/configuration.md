---
title: Configuration
subtitle: "Configure paths and datasets"
---

# Configuration

Nothing about *your* machine, dataset location or collaborator lives in the
tracked repository. All site-specific values are read at runtime by
`stressnet.config`, so the same code runs unchanged on any machine.

## Resolution order

1. **Environment variables** — `STRESSNET_DATA_ROOT`, `STRESSNET_SAVE_ROOT`
   (and `STRESSNET_CONFIG` to point at a non-default config file).
2. **`site_config.yaml`** at the repository root (gitignored).
3. **A guided `ConfigError`** telling you to copy `site_config.example.yaml` and
   to read the gitignored `HELP.md`.

## Choosing a dataset: logical keys, not paths

Training scripts select data by a **logical key** — `--datasets datav5 datav6`
or `--dataset datav5.2` — never by path. `stressnet.config.DATASET_KEYS` lists
the keys the code knows about:

| Key | Channels | Channel prefixes | Stiffness column |
| --- | --- | --- | --- |
| `datav5` | Nucleus, Collagen | `C1-`, `C2-` | `Stiffness` |
| `datav6` | Nucleus, Collagen, Actin | `C1-`, `C2-`, `C3-` | `Stiffness` |
| `datav5.2` | Nucleus, Collagen | `C1-`, `C2-` | `Mean stiffness` |

Only the physical `dir:` of each key lives in `site_config.yaml`; channels,
prefixes and the stiffness column default to the table above and can be
overridden per dataset.

## `site_config.yaml` schema

```yaml
# Required
data_root: /path/to/datasets        # contains one directory per logical dataset
save_root: /path/to/saved_models    # every run gets its own sub-directory

# Required per logical dataset key
datasets:
  datav5:
    dir: my_v5_dataset              # mandatory: folder inside data_root
    channels: [Nucleus, Collagen]   # optional, defaults per DATASET_KEYS
    channel_prefixes: [C1-, C2-]    # optional
    stiffness_col: Stiffness        # optional
  datav5.2:
    dir: my_grid_averaged_dataset
    stiffness_col: Mean stiffness
```

| Field | Required | Meaning |
| --- | --- | --- |
| `data_root` | yes | Directory that holds the dataset folders. |
| `save_root` | no\* | Directory for run directories/checkpoints. Falls back to `data_root`. |
| `datasets.<key>.dir` | yes | Folder name inside `data_root`. |
| `datasets.<key>.root` | no | Absolute path instead of `data_root`/`dir` (wins if present). |
| `datasets.<key>.channels` | no | Channel names, i.e. the `<Channel>_Train` directory prefixes. |
| `datasets.<key>.channel_prefixes` | no | Image filename prefixes, e.g. `C1-`. |
| `datasets.<key>.stiffness_col` | no | Spreadsheet column holding the target. |

The tracked template `site_config.example.yaml` (at the repository root)
documents every field, including the rarely-changed split conventions
(`train_dir`, `train_suffix`, `train_labels`, `prediction_dir`,
`prediction_suffix`, `prediction_labels`) — see [data.md](data.md).

## Environment variables

| Variable | Effect |
| --- | --- |
| `STRESSNET_DATA_ROOT` | Overrides `data_root`. |
| `STRESSNET_SAVE_ROOT` | Overrides `save_root`. |
| `STRESSNET_CONFIG` | Path to an alternative config file (default: `site_config.yaml` at the repo root). |

```bash
STRESSNET_DATA_ROOT=/path/to/datasets \
STRESSNET_SAVE_ROOT=/path/to/saved_models \
  uv run python -m stressnet.config --show
```

## Inspecting and initialising config

```bash
uv run python -m stressnet.config --show
#   datav5: /path/to/datasets/my_v5_dataset
#   datav6: <unconfigured> (...)
# save_root: /path/to/saved_models

uv run python -m stressnet.config --init \
    --data-root /path/to/datasets --save-root /path/to/saved_models
# writes site_config.yaml from the tracked defaults; edit the `dir:` values
```

`--init` refuses to overwrite an existing file unless `--force` is passed.

## `HELP.md` (optional)

`HELP.md` at the repository root is **gitignored** and holds the concrete,
machine-specific values: this machine's paths, dataset folder names, private
remote, collaborator notes and where existing checkpoints live. It is
**optional** — the library works without it by resolving everything from
`site_config.yaml`. You only need it to reproduce the **original project** with
its specific datasets, thresholds and toolchain. Whenever a tracked file shows a
placeholder like `/path/to/datasets`, `HELP.md` (if present on your machine) is
where that real value belongs. Keep it local and never commit it.

On machines that also use the AI coding-agent harness, `HELP.md` additionally
keeps the agent tooling and its configs separate from regular library usage; see
the file itself for that breakdown.

See also: [data.md](data.md) · [models.md](models.md) · [training.md](training.md)
