#!/usr/bin/env python
"""Synthetic-data self-check: verify the whole stack without any private data.

There is no test suite in this repository, so this script is the only
verification path that runs end to end on a machine with no microscopy data.
It builds a tiny but structurally exact copy of the expected dataset layout in
a temporary directory, points ``stressnet.config`` at it through
``STRESSNET_CONFIG``, and then exercises every layer of the stack:

    imports           every public module imports cleanly
    datasets          one sample from each Dataset class (shapes, labels)
    splits            build_datasets/build_loaders + the category-stratified split
    classifier        engine.train_classifier (MixUp, checkpointing) + metrics
    regression        trainer.train_model + metrics.predict_regression
    cli:regression    `train_stiffness_regression.py --debug` as a subprocess
    cli:predict       `predict.py` on the checkpoint the CLI run wrote
    cli:age           `train_age_classifier_v2.py --debug` as a subprocess
    repo_clean        the run wrote nothing into the repository

Each check prints a ``PASS``/``FAIL`` line; the exit code is 0 only if every
check passed.  The temporary tree is removed on success and on failure (pass
``--keep`` to keep it for debugging).

Nothing here is site-specific: the dataset is generated from the conventions in
``stressnet.config`` and every artefact (weights, plots, report CSVs) lands
inside the temporary directory, never in the repository.

Usage::

    uv run python scripts/selfcheck.py            # ~30-60 s, GPU or CPU
    uv run python scripts/selfcheck.py --keep     # keep the temp tree
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
# The training drivers import each other as top-level modules (``import dataset``
# etc.), so the repository root has to be importable no matter where this script
# is launched from.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep the repository free of freshly written bytecode; the subprocesses get
# PYTHONDONTWRITEBYTECODE=1 for the same reason.
sys.dont_write_bytecode = True

SEED = 42
IMAGE_SIZE = 96
DATASET_KEY = "datav5"
DATASET_DIR = "selfcheck_v5"
BACKGROUND_LEVEL = 0.15

# Synthetic per-group stiffness thresholds; the schema is the one
# dataset._load_stiffness_thresholds() expects (a `stiffness_threshold` mapping).
YOUNG_THRESHOLD = 7.0
ADULT_THRESHOLD = 11.0
THRESHOLDS_YAML = f"""\
# Synthetic thresholds written by scripts/selfcheck.py.
stiffness_threshold:
  young: {YOUNG_THRESHOLD}
  adult: {ADULT_THRESHOLD}
"""

PUBLIC_MODULES = (
    "classifiers",
    "dataset",
    "main",
    "models",
    "predict",
    "trainer",
    "transforms",
    "stressnet",
    "stressnet.config",
    "stressnet.data",
    "stressnet.engine",
    "stressnet.metrics",
    "stressnet.plots",
    "stressnet.reporting",
)


class Sample(NamedTuple):
    """One synthetic tile: sample ID (stem), age group and stiffness in kPa."""

    stem: str
    group: str
    stiffness: float

    @property
    def image(self) -> str:
        """Value of the labels table's ``Image`` column.

        The name carries the channel-1 prefix but no extension: `dataset.py`
        strips the prefix and appends `.tif` itself when building file names.
        """
        return f"C1-{self.stem}_01"


# One row per sample, i.e. one tile per sample ID, so the row-level train/val
# split used by build_datasets cannot leak sibling tiles.  Both age groups and
# all four stiffness categories appear in Training_Data, and the stiffness
# values straddle both per-group thresholds.
TRAIN_SAMPLES = (
    Sample("Y1LV", "Y", 4.0),
    Sample("Y2IVS", "Y", 5.5),
    Sample("Y3RV", "Y", 6.5),
    Sample("Y4LV", "Y", 7.5),
    Sample("Y5IVS", "Y", 8.5),
    Sample("Y6RV", "Y", 9.0),
    Sample("A1LV", "A", 9.5),
    Sample("A2IVS", "A", 10.5),
    Sample("A3RV", "A", 11.5),
    Sample("A4LV", "A", 12.5),
    Sample("A5IVS", "A", 14.0),
    Sample("A6RV", "A", 13.0),
)
PREDICTION_SAMPLES = (
    Sample("PY1LV", "Y", 5.0),
    Sample("PY2IVS", "Y", 8.5),
    Sample("PY3RV", "Y", 6.5),
    Sample("PA1LV", "A", 10.0),
    Sample("PA2IVS", "A", 12.0),
    Sample("PA3RV", "A", 13.5),
)

EXPECTED_CATEGORIES = {
    "AgeStiffnessDataset": {"young_low", "young_high", "adult_low", "adult_high"},
    "AgeStiffnessDataset_v2": {"young_low", "medium", "adult_high"},
    "CategorisedStiffnessDataset": {"low", "high"},
}

_MODULES: dict[str, object] = {}


def _modules() -> dict[str, object]:
    """Import (once) every public module plus the third-party symbols used here."""
    if _MODULES:
        return _MODULES
    import importlib

    names = {name: importlib.import_module(name) for name in PUBLIC_MODULES}
    _MODULES["names"] = names
    _MODULES["torch"] = importlib.import_module("torch")
    _MODULES["nn"] = importlib.import_module("torch.nn")
    _MODULES["np"] = importlib.import_module("numpy")
    _MODULES["pd"] = importlib.import_module("pandas")
    _MODULES["pil_image"] = importlib.import_module("PIL.Image")
    _MODULES["DataLoader"] = importlib.import_module("torch.utils.data").DataLoader
    for alias, key in (
        ("config", "stressnet.config"),
        ("data", "stressnet.data"),
        ("dataset", "dataset"),
        ("engine", "stressnet.engine"),
        ("metrics", "stressnet.metrics"),
        ("models", "models"),
        ("trainer", "trainer"),
        ("transforms", "transforms"),
    ):
        _MODULES[alias] = names[key]
    return _MODULES


# --------------------------------------------------------------------------- #
# synthetic dataset
# --------------------------------------------------------------------------- #
def _blob_image(size: int, coverage: float, rng):
    """16-bit image holding a bright disk covering `coverage` of the field.

    Percentile normalisation happens per image at read time, so a pure
    intensity scale factor would cancel out; the *area* of the bright disk
    survives it and is what the networks can learn from.
    """
    np = _modules()["np"]
    yy, xx = np.mgrid[0:size, 0:size]
    radius = math.sqrt(coverage / math.pi) * size
    dist = np.hypot(yy - size / 2.0, xx - size / 2.0)
    img = np.where(dist <= radius, 1.0, BACKGROUND_LEVEL)
    img = img + rng.normal(0.0, 0.02, size=img.shape)
    return (np.clip(img, 0.0, 1.0) * 65535.0).round().astype(np.uint16)


def _nucleus_coverage(sample: Sample, _rng) -> float:
    """Nucleus area grows with stiffness (darker image -> softer sample)."""
    return 0.06 + (sample.stiffness - 4.0) / 10.0 * 0.49


def _collagen_coverage(sample: Sample, rng) -> float:
    """Adults lay down visibly more collagen than young samples."""
    if sample.group == "A":
        return float(rng.uniform(0.40, 0.62))
    return float(rng.uniform(0.08, 0.22))


def _write_channel(directory: Path, prefix: str, samples, seed_offset: int, coverage) -> None:
    """Write one 16-bit TIFF per sample into a channel directory."""
    directory.mkdir(parents=True, exist_ok=True)
    np = _modules()["np"]
    for idx, sample in enumerate(samples):
        rng = np.random.default_rng((SEED, seed_offset, idx))
        arr = _blob_image(IMAGE_SIZE, coverage(sample, rng), rng)
        _modules()["pil_image"].fromarray(arr).save(directory / f"{prefix}{sample.stem}_01.tif")


def _write_labels(path: Path, samples) -> None:
    """Write the labels spreadsheet (`Stiffness`, `Image`, `Group` columns)."""
    pd = _modules()["pd"]
    pd.DataFrame(
        {
            "Image": [sample.image for sample in samples],
            "Stiffness": [sample.stiffness for sample in samples],
            "Group": [sample.group for sample in samples],
        }
    ).to_excel(path, index=False)


def build_dataset_tree(data_root: Path, splits: dict) -> Path:
    """Create the synthetic dataset (returns its root, i.e. `<data_root>/<dir>`)."""
    dataset_root = data_root / DATASET_DIR
    layout = (
        (splits["train_dir"], splits["train_suffix"], splits["train_labels"], TRAIN_SAMPLES),
        (
            splits["prediction_dir"],
            splits["prediction_suffix"],
            splits["prediction_labels"],
            PREDICTION_SAMPLES,
        ),
    )
    for sub_dir, suffix, labels_file, samples in layout:
        split_dir = dataset_root / sub_dir
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_labels(split_dir / labels_file, samples)
        _write_channel(split_dir / f"Nucleus{suffix}", "C1-", samples, 0, _nucleus_coverage)
        _write_channel(split_dir / f"Collagen{suffix}", "C2-", samples, 1, _collagen_coverage)
    # AgeStiffnessDataset looks for this one level above the split directory.
    (dataset_root / "stiffness_thresholds.yaml").write_text(THRESHOLDS_YAML)
    return dataset_root


def write_site_config(tmp_root: Path) -> Path:
    """Write a throw-away site_config.yaml describing the synthetic dataset."""
    config_path = tmp_root / "site_config.yaml"
    config_path.write_text(
        "# Written by scripts/selfcheck.py; points at the synthetic dataset.\n"
        f"data_root: {tmp_root / 'data'}\n"
        f"save_root: {tmp_root / 'saved'}\n"
        "datasets:\n"
        f"  {DATASET_KEY}:\n"
        f"    dir: {DATASET_DIR}\n"
        "    channels: [Nucleus, Collagen]\n"
        "    channel_prefixes: [C1-, C2-]\n"
        "    stiffness_col: Stiffness\n"
    )
    return config_path


# --------------------------------------------------------------------------- #
# check helpers
# --------------------------------------------------------------------------- #
def _tiny_classifier(num_classes: int):
    """A tiny `fc`-headed CNN, so `engine.build_optimizer` works unchanged."""
    nn = _modules()["nn"]

    class TinyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = 8
            self.features = nn.Sequential(
                nn.Conv2d(3, width, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(2),
                nn.Flatten(),
            )
            self.fc = nn.Linear(width * 2 * 4, num_classes)

        def forward(self, x):
            return self.fc(self.features(x))

    return TinyClassifier()


def _assert(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _brief(detail: str, limit: int = 110) -> str:
    """First line of a check detail, for the one-line summary table."""
    line = str(detail).strip().splitlines()[0]
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _run_driver(args: list[str]) -> str:
    """Run a training driver as a subprocess; abort the check on a non-zero exit.

    `sys.executable` is the interpreter uv created for this project, i.e. the
    same one `uv run python <driver>` would use, minus a second uv resolution.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print(f"$ cd {REPO_ROOT} && uv run python {' '.join(args)}")
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout or ""
    if proc.returncode != 0:
        print(output)
        raise AssertionError(
            f"`uv run python {' '.join(args)}` exited with {proc.returncode} "
            "(full output above)"
        )
    for line in [ln for ln in output.splitlines() if ln.strip()][-8:]:
        print(f"    | {line}")
    return output


def _glob(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern)) if root.exists() else []


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_imports(_ctx: dict) -> str:
    """Every public module imports cleanly (no site config required)."""
    names = _modules()["names"]
    return f"{len(names)} modules: {', '.join(sorted(names))}"


def check_datasets(ctx: dict) -> str:
    """One sample from every Dataset class: shapes, labels and categories."""
    m = _modules()
    ds_mod, tf = m["dataset"], m["transforms"]
    cfg = m["config"].dataset_config(DATASET_KEY)
    train_dir, suffix, labels = m["config"].train_split()
    split_dir = cfg["root"] / train_dir
    common = {
        "ch_names": cfg["ch_names"],
        "ch_name_prefix": cfg["ch_name_prefix"],
        "ch_dir_suffix": suffix,
    }
    infer = tf.data_transforms_inference

    image, target = ds_mod.StiffnessDataset(labels, split_dir, **common)[0]
    _assert(
        tuple(image.shape) == (3, IMAGE_SIZE, IMAGE_SIZE),
        f"StiffnessDataset image shape {tuple(image.shape)}",
    )
    _assert(tuple(target.shape) == (1,), f"StiffnessDataset target {target}")

    image, target, fname = ds_mod.ValidationDataset(
        labels, split_dir, transform=infer, **common
    )[0]
    _assert(tuple(image.shape) == (3, 224, 224), f"ValidationDataset transformed {image.shape}")
    _assert(fname.endswith(".tif"), f"ValidationDataset fname {fname!r}")

    image, age, fname = ds_mod.AgeDataset(labels, split_dir, transform=infer, **common)[0]
    _assert(tuple(image.shape) == (3, 224, 224), f"AgeDataset transformed {image.shape}")
    _assert(age in ("young", "adult"), f"AgeDataset label {age!r}")

    counts = []
    for name in ("AgeStiffnessDataset", "AgeStiffnessDataset_v2", "CategorisedStiffnessDataset"):
        dataset = getattr(ds_mod, name)(labels, split_dir, transform=infer, **common)
        found = set(dataset.img_labels["Category"])
        _assert(
            found == EXPECTED_CATEGORIES[name],
            f"{name} categories {sorted(found)} != {sorted(EXPECTED_CATEGORIES[name])}",
        )
        categories = sorted(found)
        counts.append(f"{name}={len(categories)}")
        image, category, _fname = dataset[0]
        _assert(category in categories, f"{name} returned category {category!r}")
        _assert(tuple(image.shape) == (3, 224, 224), f"{name} transformed {image.shape}")

    return f"6 Dataset classes; categories {', '.join(counts)}"


def check_splits(ctx: dict) -> str:
    """build_datasets (Group- and Category-stratified) + build_loaders."""
    m = _modules()
    data_mod = m["data"]
    cfg = m["config"].dataset_config(DATASET_KEY)
    train_transform = data_mod.TRAIN_TRANSFORMS["Xv1"]

    datasets = data_mod.build_datasets(cfg, train_transform, 0.25)
    _assert(
        len(datasets["heldout"]) == len(PREDICTION_SAMPLES),
        "heldout is not the Prediction_Data split",
    )
    _assert(
        len(datasets["train"]) + len(datasets["val"]) == len(TRAIN_SAMPLES),
        "internal split lost samples",
    )
    _assert(len(datasets["val"]) > 0, "internal split has an empty val phase")

    by_category = data_mod.build_datasets(
        cfg,
        train_transform,
        0.25,
        dataset_cls=m["dataset"].AgeStiffnessDataset,
        stratify_col="Category",
    )
    _assert(len(by_category["val"]) > 0, "category-stratified split has an empty val phase")

    loaders = data_mod.build_loaders(datasets, 4)
    images, labels, _fnames = next(iter(loaders["train"]))
    _assert(tuple(images.shape[:2]) == (4, 3), f"loader batch image shape {tuple(images.shape)}")
    _assert(len(labels) == 4, f"loader batch yielded {len(labels)} labels")

    return (
        f"train/val/heldout={len(datasets['train'])}/{len(datasets['val'])}/"
        f"{len(datasets['heldout'])}; category-stratified val={len(by_category['val'])}"
    )


def check_classifier(ctx: dict) -> str:
    """engine.train_classifier (MixUp + checkpointing) and the metrics helpers."""
    m = _modules()
    torch, nn, engine = m["torch"], m["nn"], m["engine"]
    cfg = m["config"].dataset_config(DATASET_KEY)
    datasets = m["data"].build_datasets(cfg, m["data"].TRAIN_TRANSFORMS["Xv1"], 0.25)
    loaders = m["data"].build_loaders(datasets, 4)
    sizes = {"train": len(datasets["train"]), "val": len(datasets["val"])}
    device = m["models"].TORCH_DEVICE

    model = _tiny_classifier(num_classes=2)
    strategy = {"lr_head": 1e-3, "lr_backbone": 1e-5, "weight_decay": 1e-4}
    engine.set_backbone_trainable(model, False)
    optimizer = engine.build_optimizer(model, strategy, backbone_trainable=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    checkpoint = ctx["tmp"] / "tiny_classifier.pt"

    model, losses, accs, best_loss = engine.train_classifier(
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler,
        dataloaders=loaders,
        dataset_sizes=sizes,
        label_to_idx=engine.AGE_TO_IDX,
        num_epochs=2,
        device=device,
        checkpoint_path=checkpoint,
        early_stopping_patience=3,
        mixup_alpha=0.2,
    )
    _assert(len(losses["train"]) == 2, f"loss history {losses['train']}")
    _assert(len(accs["val"]) == 2, f"acc history {accs['val']}")
    _assert(
        all(math.isfinite(v) for v in losses["train"] + losses["val"] + accs["train"] + accs["val"]),
        f"non-finite history: {losses} {accs}",
    )
    _assert(math.isfinite(best_loss), f"best val loss {best_loss}")
    _assert(checkpoint.exists(), "best-val checkpoint was not written")

    preds = m["metrics"].predict_classifier(
        model,
        m["DataLoader"](datasets["heldout"], batch_size=4),
        device,
        class_names=list(engine.AGE_TO_IDX),
        desc="heldout",
    )
    expected_columns = {"Image", "label", "y_true", "pred", "correct", "p_young", "p_adult"}
    _assert(
        expected_columns <= set(preds.columns),
        f"prediction frame columns {sorted(preds.columns)}",
    )
    _assert(len(preds) == len(PREDICTION_SAMPLES), f"{len(preds)} predictions")
    accuracy = m["metrics"].accuracy(preds)
    macro = m["metrics"].macro_accuracy(preds, 2)
    _assert(0.0 <= accuracy <= 1.0 and 0.0 <= macro <= 1.0, "accuracy out of range")

    return (
        f"2 epochs, val loss {losses['val'][-1]:.4f}, val acc {accs['val'][-1]:.3f}, "
        f"heldout acc {accuracy:.3f} (macro {macro:.3f})"
    )


def check_regression(ctx: dict) -> str:
    """trainer.train_model on StiffnessDataset + metrics.predict_regression."""
    m = _modules()
    torch, nn = m["torch"], m["nn"]
    cfg = m["config"].dataset_config(DATASET_KEY)
    datasets = m["data"].build_datasets_regression(cfg, m["transforms"].data_transforms)
    loaders = {
        phase: m["DataLoader"](
            datasets[phase], batch_size=4, shuffle=(phase == "train"), num_workers=0
        )
        for phase in ("train", "val")
    }
    sizes = {phase: len(datasets[phase]) for phase in ("train", "val")}
    device = m["models"].TORCH_DEVICE

    # `train_model` moves the model and the batches onto `device` itself, so
    # pass a freshly built CPU model to exercise that path directly.
    model = m["models"].resnet18(pretrained=False)
    model, losses = m["trainer"].train_model(
        model,
        nn.MSELoss(),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        None,
        loaders,
        sizes,
        num_epochs=1,
        device=device,
    )
    _assert(set(losses) == {"train", "val"}, f"loss keys {sorted(losses)}")
    _assert(
        all(len(values) == 1 for values in losses.values()),
        f"loss history {losses}",
    )
    _assert(
        all(math.isfinite(v) for v in losses["train"] + losses["val"]),
        f"non-finite loss: {losses}",
    )

    result = m["metrics"].predict_regression(
        model, m["DataLoader"](datasets["val"], batch_size=4), device
    )
    _assert(
        result["y_tgt"].shape[0] == len(PREDICTION_SAMPLES)
        and result["y_pred"].shape == result["y_tgt"].shape,
        f"prediction shapes {result['y_tgt'].shape} / {result['y_pred'].shape}",
    )
    return f"1 epoch, train loss {losses['train'][-1]:.4f}, val loss {losses['val'][-1]:.4f}"


def check_cli_regression(ctx: dict) -> str:
    """`train_stiffness_regression.py --debug` writes weights, plots and a report."""
    report = ctx["tmp"] / "cli_regression_report.csv"
    args = [
        "train_stiffness_regression.py",
        "--dataset",
        DATASET_KEY,
        "--debug",
        "--report",
        str(report),
    ]
    _run_driver(args)
    _assert(report.exists(), f"report {report} was not written")
    _assert(report.read_text().count("\n") >= 2, "report has no data row")

    run_root = ctx["save_root"] / "stiffness"
    weights = _glob(run_root, "*.pt")
    plots = _glob(run_root, "*.png")
    _assert(weights, f"no checkpoint under {run_root}")
    _assert(len(plots) >= 2, f"expected loss + prediction plots, found {len(plots)}")
    _assert(_glob(run_root, "*_preds.csv"), "no prediction CSV")
    ctx["regression_weights"] = weights[0]
    return f"weights={weights[0].name}, {len(plots)} plots, report={report.name}"


def check_cli_predict(ctx: dict) -> str:
    """`predict.py` scores the CLI checkpoint and writes its artefacts."""
    weights = ctx.get("regression_weights")
    _assert(weights, "cli:regression produced no checkpoint (its FAIL is reported above)")
    out_dir = ctx["tmp"] / "predict_out"
    args = [
        "predict.py",
        "--weights",
        str(weights),
        "--dataset",
        DATASET_KEY,
        "--phase",
        "prediction",
        "--output",
        str(out_dir),
    ]
    _run_driver(args)
    run_dir = out_dir / weights.name
    preds = run_dir / "prediction_preds.csv"
    scatter = run_dir / "prediction_scatter.html"
    _assert(preds.exists(), f"{preds} was not written")
    _assert(scatter.exists(), f"{scatter} was not written")
    _assert(preds.read_text().count("\n") >= 2, "prediction CSV has no data row")
    return f"{preds.name} + {scatter.name} in {run_dir.relative_to(ctx['tmp'])}"


def check_cli_age(ctx: dict) -> str:
    """`train_age_classifier_v2.py --debug` (default strategy) runs end to end."""
    report = ctx["tmp"] / "cli_age_v2_report.csv"
    args = ["train_age_classifier_v2.py", "--debug", "--report", str(report)]
    _run_driver(args)
    _assert(report.exists(), f"report {report} was not written")
    _assert(report.read_text().count("\n") >= 2, "report has no data row")

    weights = _glob(ctx["save_root"], "*.pt")
    plots = _glob(ctx["save_root"], "*_preds.png")
    _assert(weights, f"no checkpoint under {ctx['save_root']}")
    _assert(plots, "no prediction plot")
    return f"weights={weights[0].name}, report={report.name}"


def repo_snapshot() -> dict:
    """Top-level repository entries (name -> mtime/size), ignoring tool caches."""
    snapshot = {}
    for path in sorted(REPO_ROOT.iterdir()):
        if path.name in {"__pycache__", ".ruff_cache"}:
            continue
        stat = path.stat()
        snapshot[path.name] = (stat.st_mtime_ns, stat.st_size if path.is_file() else None)
    return snapshot


def check_repo_clean(ctx: dict) -> str:
    """None of the runs may create or modify a file in the repository."""
    before, after = ctx["repo_before"], repo_snapshot()
    created = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(name for name in set(before) & set(after) if before[name] != after[name])
    _assert(
        not (created or removed or modified),
        f"repository changed: created={created} removed={removed} modified={modified}",
    )
    return f"{len(after)} top-level entries unchanged"


CHECKS = (
    ("imports", check_imports),
    ("datasets", check_datasets),
    ("splits", check_splits),
    ("classifier", check_classifier),
    ("regression", check_regression),
    ("cli:regression", check_cli_regression),
    ("cli:predict", check_cli_predict),
    ("cli:age", check_cli_age),
    ("repo_clean", check_repo_clean),
)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic-data self-check for the stressnet-tests stack",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temporary dataset/config/artefacts for inspection",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tmp_parent = Path("/tmp/opencode") if Path("/tmp/opencode").is_dir() else None
    tmp_root = Path(tempfile.mkdtemp(prefix="stressnet-selfcheck-", dir=tmp_parent))
    results: list[tuple[str, bool, str]] = []
    ctx: dict = {"tmp": tmp_root, "save_root": tmp_root / "saved"}
    exit_code = 1

    print(f"stressnet-tests self-check\n  temp tree : {tmp_root}")
    started = time.time()
    try:
        # 1. throw-away site config + synthetic dataset, then point the whole
        #    process tree at it before anything imports stressnet.config.
        config_path = write_site_config(tmp_root)
        os.environ["STRESSNET_CONFIG"] = str(config_path)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        # The temp config supplies both roots, so make sure roots exported in the
        # developer's shell cannot redirect the runs onto real data/artefacts.
        for var in ("STRESSNET_DATA_ROOT", "STRESSNET_SAVE_ROOT"):
            os.environ.pop(var, None)
        print(f"  config    : {config_path} (STRESSNET_CONFIG)")
        m = _modules()
        m["torch"].manual_seed(SEED)
        m["np"].random.seed(SEED)
        config = m["config"]
        ctx["ds_root"] = build_dataset_tree(tmp_root / "data", config.splits())
        print(f"  dataset   : {ctx['ds_root']}")
        print(f"  save_root : {ctx['save_root']}")
        print(f"  modules   : {REPO_ROOT}")

        ctx["repo_before"] = repo_snapshot()
        for name, check in CHECKS:
            print(f"\n--- {name} ---")
            try:
                detail = check(ctx)
            except Exception as exc:  # noqa: BLE001 - every failure is reported
                results.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL {name}: {_brief(f'{type(exc).__name__}: {exc}', 160)}")
                traceback.print_exc()
            else:
                results.append((name, True, str(detail)))
                print(f"PASS {name}: {detail}")

        failures = [name for name, ok, _detail in results if not ok]
        print("\n" + "=" * 72)
        print("self-check summary")
        print("=" * 72)
        for name, ok, detail in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} {_brief(detail)}")
        print(
            f"\n{len(results) - len(failures)}/{len(results)} checks passed "
            f"in {time.time() - started:.1f}s"
        )
        print("SELF-CHECK: " + ("PASS" if not failures else f"FAIL ({', '.join(failures)})"))
        exit_code = 1 if failures else 0
    except Exception:  # setup failure (temp tree / config / dataset)
        print("\nSELF-CHECK: FAIL (setup error)\n", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1
    finally:
        if args.keep:
            print(f"\nKept temp tree: {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"\nRemoved temp tree: {tmp_root}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
