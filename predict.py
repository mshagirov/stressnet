"""Config-driven prediction entrypoint for stressnet-tests.

Replaces the ad-hoc ``main.py`` / ``plot_finetuned.py`` scripts with a single
CLI. Dataset locations are resolved through :mod:`stressnet.config` (nothing
site-specific is hard-coded here), the task is detected from the checkpoint's
final fully-connected layer, and the artefacts keep the historical layout:

    <output>/<weights filename>/<phase>_scatter.html   (regression only)
    <output>/<weights filename>/<phase>_preds.csv

``val`` and ``prediction`` refer to the same held-out split; the output files
are always named after the phase that was passed on the command line, so the
old ``-p val`` command lines keep producing ``val_*`` artefacts.

Usage (from inside stressnet-tests/)::

    # logical dataset, resolved via site_config.yaml
    uv run python predict.py -w path/to/run_45epoch_AdamW.pt --dataset datav5

    # raw dataset layout (the previous main.py flags)
    uv run python predict.py -w w.pt -i <split_root> -l labels.xlsx -c _Prediction

    # age classifier checkpoint (task is auto-detected)
    uv run python predict.py -w classifier.pt --dataset datav5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from classifiers import resnet18_classifier
from dataset import AgeDataset, ValidationDataset, ValidationDatasetAgeLoc
from models import get_predict_func, resnet18
from stressnet.config import dataset_config, prediction_split, save_root, train_split
from stressnet.engine import AGE_TO_IDX
from stressnet.plots import plot_scatter
from transforms import data_transforms_inference, stiffness_transform

PHASES = ('train', 'val', 'prediction')
TASKS = ('auto', 'regression', 'classification')
DATASET_CLASSES = ('auto', 'ageloc', 'stiffness')

# Raw (-i/-l/-c) mode assumes the historical 2-channel v5 layout.
RAW_CH_NAMES = ('Nucleus', 'Collagen')
RAW_CH_PREFIXES = ('C1-', 'C2-')
RAW_STIFFNESS_COL = 'Stiffness'


@dataclass
class InputSpec:
    """Resolved description of the split that should be scored."""

    name: str
    root: Path
    labels: str
    ch_dir_suffix: str
    ch_names: tuple[str, ...]
    ch_name_prefix: tuple[str, ...]
    stiffness_col: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='predict.py',
        description='Run a trained StressNet checkpoint over train/prediction data.',
    )
    parser.add_argument('-w', '--weights', required=True,
                        help='path to the model checkpoint')

    source = parser.add_argument_group(
        'dataset selection (use either --dataset, or all of -i/-l/-c)')
    source.add_argument('--dataset',
                        help='logical dataset name from site_config.yaml')
    source.add_argument('-i', '--input',
                        help='root directory of the input split (raw mode)')
    source.add_argument('-l', '--labels',
                        help='labels spreadsheet inside the input root (raw mode)')
    source.add_argument('-c', '--chsuffix',
                        help='channel directory suffix inside the input root (raw mode)')

    parser.add_argument('-p', '--phase', choices=PHASES, default='prediction',
                        help="split to score; 'val' and 'prediction' both mean the "
                             "held-out split, outputs are named after this value "
                             "(default: prediction)")
    parser.add_argument('-o', '--output',
                        help='output directory (default: save_root()/<weights stem>)')
    parser.add_argument('--task', choices=TASKS, default='auto',
                        help='force the prediction task instead of detecting it '
                             'from the checkpoint head (default: auto)')
    parser.add_argument('--dataset-class', choices=DATASET_CLASSES, default='auto',
                        help='Dataset class used for regression: ageloc = '
                             'ValidationDatasetAgeLoc (historical main.py), '
                             'stiffness = ValidationDataset (2-channel), '
                             'auto = ageloc (default: auto)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='inference batch size (default: 2)')
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def inspect_checkpoint(weights: Path) -> tuple[int, str, float]:
    """Return ``(num_outputs, task, dropout)`` for a checkpoint's final layer.

    Handles both head layouts used in this repository: a plain ``Linear``
    (``fc.weight``) and the classifier dropout head
    ``Sequential(Dropout, Linear)`` (``fc.1.weight``).
    """
    state = torch.load(weights, weights_only=True, map_location='cpu')
    if not isinstance(state, dict):
        raise SystemExit(f'Unsupported checkpoint format in {weights}: expected a state_dict.')

    if 'fc.weight' in state:
        head, dropout = state['fc.weight'], 0.0
        head_desc = 'Linear'
    elif 'fc.1.weight' in state:
        # Sequential(Dropout, Linear): dropout has no parameters, so any
        # dropout > 0 rebuilds the exact same head keys as the checkpoint.
        head, dropout = state['fc.1.weight'], 0.5
        head_desc = 'Sequential(Dropout, Linear)'
    else:
        raise SystemExit(
            f"Could not find a supported final layer in {weights}: expected "
            "'fc.weight' (Linear head) or 'fc.1.weight' (dropout head)."
        )

    num_outputs = int(head.shape[0])
    task = 'regression' if num_outputs == 1 else 'classification'
    print(f'Checkpoint head: {head_desc}, {num_outputs} output(s) -> {task}')
    return num_outputs, task, dropout


def resolve_inputs(args: argparse.Namespace) -> InputSpec:
    """Resolve ``--dataset`` (via stressnet.config) or the raw -i/-l/-c flags."""
    raw = (args.input, args.labels, args.chsuffix)
    if args.dataset and any(value is not None for value in raw):
        raise SystemExit('Use either --dataset or -i/-l/-c, not both.')
    if not args.dataset and not all(raw):
        raise SystemExit('Provide --dataset NAME, or all of -i/--input, '
                         '-l/--labels and -c/--chsuffix.')

    if args.dataset:
        cfg = dataset_config(args.dataset)
        sub_dir, suffix, labels = (
            train_split() if args.phase == 'train' else prediction_split())
        return InputSpec(
            name=args.dataset,
            root=cfg['root'] / sub_dir,
            labels=labels,
            ch_dir_suffix=suffix,
            ch_names=tuple(cfg['ch_names']),
            ch_name_prefix=tuple(cfg['ch_name_prefix']),
            stiffness_col=cfg['stiffness_col'],
        )

    return InputSpec(
        name='raw',
        root=Path(args.input),
        labels=args.labels,
        ch_dir_suffix=args.chsuffix,
        ch_names=RAW_CH_NAMES,
        ch_name_prefix=RAW_CH_PREFIXES,
        stiffness_col=RAW_STIFFNESS_COL,
    )


def write_scatter(y_tgt, y_pred, x_names, phase: str, out_dir: Path) -> None:
    """Render the regression scatter under ``<phase>_scatter.html``.

    ``stressnet.plots.plot_scatter`` only titles the ``train``/``val`` phases,
    while this CLI also accepts ``prediction`` (the same held-out split as
    ``val``). Plot the held-out split under its ``val`` label and keep the
    requested ``<phase>_scatter.html`` file name either way.
    """
    plotted = phase if phase in ('train', 'val') else 'val'
    plot_scatter(y_tgt, y_pred, x_names, plotted, out_dir)
    produced = out_dir / f'{plotted}_scatter.html'
    target = out_dir / f'{phase}_scatter.html'
    if produced != target:
        produced.replace(target)
        print(f'Saved: {target}')


def run_regression(args: argparse.Namespace, weights: Path, device: torch.device,
                   spec: InputSpec, out_dir: Path) -> None:
    dataset_cls = (ValidationDataset if args.dataset_class == 'stiffness'
                   else ValidationDatasetAgeLoc)
    print(f'Regression dataset: {dataset_cls.__name__}')

    dataset = dataset_cls(
        spec.labels, spec.root,
        ch_names=spec.ch_names,
        ch_dir_suffix=spec.ch_dir_suffix,
        ch_name_prefix=spec.ch_name_prefix,
        stiffness_col=spec.stiffness_col,
        transform=data_transforms_inference,
        target_transform=stiffness_transform,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = resnet18(weights_path=weights, device=device).to(device)
    model.eval()
    predict = get_predict_func(model, device=device)

    y_tgt, y_pred, x_names = [], [], []
    for x, y, fnames in loader:
        x_names.extend(fnames)
        y_tgt.append(y.cpu().numpy())
        y_pred.append(predict(x).cpu().numpy())

    y_tgt = np.concatenate(y_tgt).ravel()
    y_pred = np.concatenate(y_pred).ravel()

    out_dir.mkdir(parents=True, exist_ok=True)
    write_scatter(y_tgt, y_pred, x_names, args.phase, out_dir)
    pd.DataFrame({'Image': x_names, 'Ytgt': y_tgt, 'Ypred': y_pred}).to_csv(
        out_dir / f'{args.phase}_preds.csv', index=False)
    print(f'Saved: {out_dir / f"{args.phase}_preds.csv"} ({len(x_names)} samples)')


def run_classification(args: argparse.Namespace, weights: Path, device: torch.device,
                       spec: InputSpec, out_dir: Path, num_classes: int,
                       dropout: float) -> None:
    if num_classes != 2:
        raise SystemExit(
            f'Classification inference currently supports the 2-class age head; '
            f'the checkpoint has {num_classes} outputs.'
        )

    dataset = AgeDataset(
        spec.labels, spec.root,
        ch_names=spec.ch_names,
        ch_dir_suffix=spec.ch_dir_suffix,
        ch_name_prefix=spec.ch_name_prefix,
        stiffness_col=spec.stiffness_col,
        transform=data_transforms_inference,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = resnet18_classifier(num_classes=num_classes, weights_path=weights,
                                device=device, dropout=dropout).to(device)
    model.eval()

    rows = []
    with torch.inference_mode():
        for images, ages, fnames in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
            for fname, age, p_adult in zip(fnames, ages, probs):
                y_true = AGE_TO_IDX[age]
                pred = int(p_adult >= 0.5)
                rows.append({
                    'Image': fname,
                    'age': age,
                    'y_true': y_true,
                    'p_adult': float(p_adult),
                    'pred': pred,
                    'correct': int(y_true == pred),
                })

    df = pd.DataFrame(rows, columns=['Image', 'age', 'y_true', 'p_adult', 'pred', 'correct'])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f'{args.phase}_preds.csv', index=False)
    acc = df['correct'].mean() if len(df) else float('nan')
    print(f'Saved: {out_dir / f"{args.phase}_preds.csv"} '
          f'({len(df)} samples, accuracy={acc:.3f})')


def run(args: argparse.Namespace) -> None:
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Can't find:\n  {weights}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_outputs, detected_task, dropout = inspect_checkpoint(weights)
    task = detected_task if args.task == 'auto' else args.task
    if task == 'regression' and num_outputs != 1:
        raise SystemExit(f'--task regression needs a 1-output head, found {num_outputs}.')
    if task == 'classification' and num_outputs < 2:
        raise SystemExit(f'--task classification needs a multi-output head, '
                         f'found {num_outputs}.')

    spec = resolve_inputs(args)
    output_root = Path(args.output) if args.output else save_root() / weights.stem
    out_dir = output_root / weights.name

    print(f'Using {device} device')
    print(f'Task: {task}')
    print(f'Dataset: {spec.name} -> {spec.root}')
    print(f'Output dir: {out_dir}')

    if task == 'regression':
        run_regression(args, weights, device, spec, out_dir)
    else:
        run_classification(args, weights, device, spec, out_dir, num_outputs, dropout)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == '__main__':
    main()
