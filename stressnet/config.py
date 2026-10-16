"""Site configuration for stressnet-tests.

Dataset locations and saved-model locations are **site-specific** and are
deliberately not hard-coded anywhere in this repository. Everything that is
machine- or data-specific is resolved through this module:

Resolution order
    1. environment variables ``STRESSNET_DATA_ROOT`` / ``STRESSNET_SAVE_ROOT``
       (and ``STRESSNET_CONFIG`` to point at a different config file)
    2. ``site_config.yaml`` at the repository root (gitignored)
    3. a guided :class:`ConfigError` that points back at
       ``site_config.example.yaml`` and ``HELP.md``

Typical use from a training script::

    from stressnet.config import dataset_config, save_root

    cfg = dataset_config("datav5")          # {"root": Path, "ch_names": ...}
    out = save_root() / "my_run"

Quick setup on a new machine::

    python -m stressnet.config --init --data-root /path/to/datasets \
        --save-root /path/to/saved_models
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = REPO_ROOT / "site_config.yaml"
EXAMPLE_CONFIG_FILE = REPO_ROOT / "site_config.example.yaml"

# Logical dataset keys referenced by the ``--datasets`` CLI flags. These are
# portable names; the physical folder for each one lives in site_config.yaml.
DEFAULT_DATASETS: dict[str, dict[str, Any]] = {
    "datav5": {
        "channels": ["Nucleus", "Collagen"],
        "channel_prefixes": ["C1-", "C2-"],
        "stiffness_col": "Stiffness",
    },
    "datav6": {
        "channels": ["Nucleus", "Collagen", "Actin"],
        "channel_prefixes": ["C1-", "C2-", "C3-"],
        "stiffness_col": "Stiffness",
    },
    "datav5.2": {
        "channels": ["Nucleus", "Collagen"],
        "channel_prefixes": ["C1-", "C2-"],
        "stiffness_col": "Mean stiffness",
    },
}
DATASET_KEYS = tuple(DEFAULT_DATASETS)

# Directory / filename conventions for the labels layout. Override in
# site_config.yaml only if your data uses a different layout.
DEFAULT_SPLITS: dict[str, str] = {
    "train_dir": "Training_Data",
    "train_suffix": "_Train",
    "train_labels": "Stiffness_Data_Training.xlsx",
    "prediction_dir": "Prediction_Data",
    "prediction_suffix": "_Prediction",
    "prediction_labels": "Stiffness_Data_Prediction.xlsx",
}


class ConfigError(RuntimeError):
    """Raised when site configuration is missing or incomplete."""


def config_path() -> Path:
    """Return the active config file path (``STRESSNET_CONFIG`` override aware)."""
    return Path(os.environ.get("STRESSNET_CONFIG", DEFAULT_CONFIG_FILE))


def _guidance() -> str:
    return (
        "No site configuration found.\n\n"
        f"Create {DEFAULT_CONFIG_FILE.name} from the tracked example:\n"
        f"    cp {EXAMPLE_CONFIG_FILE.name} {DEFAULT_CONFIG_FILE.name}\n"
        "then fill in `data_root`, `save_root` and the `datasets` section.\n\n"
        "On a shared machine the concrete values are documented in the\n"
        "gitignored HELP.md at the repository root. See also\n"
        "docs/configuration.md. Alternatively set the STRESSNET_DATA_ROOT\n"
        "and STRESSNET_SAVE_ROOT environment variables."
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - depends on user file
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level.")
    return raw


def load_site_config(*, required: bool = True) -> dict[str, Any]:
    """Load and merge the site configuration (env vars override the YAML file)."""
    path = config_path()
    data: dict[str, Any] = {}
    if path.exists():
        data = _read_yaml(path)

    if os.environ.get("STRESSNET_DATA_ROOT"):
        data["data_root"] = os.environ["STRESSNET_DATA_ROOT"]
    if os.environ.get("STRESSNET_SAVE_ROOT"):
        data["save_root"] = os.environ["STRESSNET_SAVE_ROOT"]

    if required and not data.get("data_root"):
        raise ConfigError(_guidance())
    return data


def data_root(config: dict[str, Any] | None = None) -> Path:
    """Return the directory that contains the dataset folders."""
    config = load_site_config() if config is None else config
    if not config.get("data_root"):
        raise ConfigError(_guidance())
    return Path(config["data_root"]).expanduser()


def save_root(config: dict[str, Any] | None = None) -> Path:
    """Return the directory where runs/checkpoints are written."""
    config = load_site_config() if config is None else config
    root = config.get("save_root") or config.get("data_root")
    if not root:
        raise ConfigError(_guidance())
    return Path(root).expanduser()


def dataset_config(name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a logical dataset name into concrete paths and channel settings."""
    config = load_site_config() if config is None else config
    entries = config.get("datasets") or {}
    entry = dict(entries.get(name, {}))
    defaults = DEFAULT_DATASETS.get(name, {})

    if "root" in entry:
        root = Path(entry["root"]).expanduser()
    else:
        directory = entry.get("dir")
        if not directory:
            configured = ", ".join(sorted(entries)) or "none"
            raise ConfigError(
                f"Dataset '{name}' is not configured (configured: {configured}).\n\n"
                f"Add a `{name}:` entry with a `dir:` to "
                f"{config_path()} — see {EXAMPLE_CONFIG_FILE.name} or HELP.md."
            )
        root = data_root(config) / directory

    return {
        "name": name,
        "root": root,
        "ch_names": tuple(entry.get("channels", defaults.get("channels", ("Nucleus", "Collagen")))),
        "ch_name_prefix": tuple(
            entry.get("channel_prefixes", defaults.get("channel_prefixes", ("C1-", "C2-")))
        ),
        "stiffness_col": entry.get("stiffness_col", defaults.get("stiffness_col", "Stiffness")),
    }


def dataset_configs(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Resolve every configured dataset (logical keys present in the config)."""
    config = load_site_config() if config is None else config
    names = list((config.get("datasets") or {}).keys()) or list(DATASET_KEYS)
    return {name: dataset_config(name, config) for name in names}


def splits(config: dict[str, Any] | None = None) -> dict[str, str]:
    """Return the label-file / split-directory conventions."""
    config = load_site_config(required=False) if config is None else config
    result = dict(DEFAULT_SPLITS)
    for key in DEFAULT_SPLITS:
        if config.get(key):
            result[key] = config[key]
    return result


def train_split(config: dict[str, Any] | None = None) -> tuple[str, str, str]:
    """Return ``(sub_dir, channel_suffix, labels_file)`` for Training_Data."""
    s = splits(config)
    return s["train_dir"], s["train_suffix"], s["train_labels"]


def prediction_split(config: dict[str, Any] | None = None) -> tuple[str, str, str]:
    """Return ``(sub_dir, channel_suffix, labels_file)`` for Prediction_Data."""
    s = splits(config)
    return s["prediction_dir"], s["prediction_suffix"], s["prediction_labels"]


def write_site_config(data_root_value: str, save_root_value: str, *, force: bool = False) -> Path:
    """Write a starter ``site_config.yaml`` from the tracked defaults."""
    path = config_path()
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists (use --force to overwrite).")
    payload: dict[str, Any] = {
        "data_root": data_root_value,
        "save_root": save_root_value,
        "datasets": {
            name: {"dir": f"{name}_dataset", **settings}
            for name, settings in DEFAULT_DATASETS.items()
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or initialise stressnet site config")
    parser.add_argument("--init", action="store_true", help="write a starter site_config.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite an existing config")
    parser.add_argument("--data-root", help="value for data_root")
    parser.add_argument("--save-root", help="value for save_root")
    parser.add_argument("--show", action="store_true", help="print the resolved configuration")
    args = parser.parse_args(argv)

    if args.init:
        if not (args.data_root and args.save_root):
            parser.error("--init requires --data-root and --save-root")
        path = write_site_config(args.data_root, args.save_root, force=args.force)
        print(f"Wrote {path}")
        print("Edit the dataset `dir:` values, then run scripts with `uv run python ...`.")

    if args.show or not args.init:
        try:
            config = load_site_config()
        except ConfigError as exc:
            print(exc, file=sys.stderr)
            return 1
        for name in DATASET_KEYS:
            try:
                cfg = dataset_config(name, config)
                print(f"{name:>8}: {cfg['root']}")
            except ConfigError as exc:
                print(f"{name:>8}: <unconfigured> ({exc})")
        print(f"save_root: {save_root(config)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
