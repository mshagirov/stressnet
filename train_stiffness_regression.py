'''Train a stiffness regression model (ResNet18, scalar output).

Datasets (--dataset), resolved from site_config.yaml:
  datav5  : tiled images, Xv1 transforms (crop 256)
  datav5.2: grid-averaged variant, Xv4 transforms (larger grid crop)

Recipe (previous best gradual-unfreeze strategy):
  - Yv1 target transform: log then (x-1.85)/3.0; loss = MSE
  - Stage 1: backbone frozen, head lr 1e-3, 15 epochs
  - Stage 2: unfrozen, head lr 1e-4 / backbone lr 1e-5, 30 epochs (45 total)
  - AdamW wd 1e-4, cosine annealing per stage, batch 16, seed 42
  - val phase = Prediction_Data (best-val checkpointing selects on it)

Outputs land in <save_root>/stiffness/<run_name>/ with best-val weights, loss
curves and a two-panel predicted-vs-true scatter (train / val) with a dashed
1-to-1 line. Dataset roots, the stiffness column and save_root come from
stressnet.config (site_config.yaml / environment variables); see HELP.md.

Usage (from the repository root):
    uv run python train_stiffness_regression.py --dataset datav5
    uv run python train_stiffness_regression.py --debug        # smoke test
'''

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import TORCH_DEVICE, resnet18
from stressnet.config import dataset_config, save_root
from stressnet.data import build_datasets_regression
from stressnet.engine import build_optimizer, set_backbone_trainable
from stressnet.metrics import predict_regression
from stressnet.plots import plot_regression_predictions
from stressnet.reporting import append_report
from trainer import train_model
from transforms import data_transforms, data_transforms_v4

# Per-dataset transforms: datav5 = Xv1 (tiles), datav5.2 = Xv4 (grids).
# Roots and the target column are resolved from stressnet.config at run time.
DATASET_TRANSFORMS = {
    'datav5': {
        'transforms': data_transforms,
        'xv_tag': 'Xv1',
    },
    'datav5.2': {
        'transforms': data_transforms_v4,
        'xv_tag': 'Xv4',
    },
}

STRATEGY = {
    'lr_head': 1e-3, 'lr_backbone': 1e-5, 'weight_decay': 1e-4,
    'freeze_epochs': 15, 'epochs': 45,
}
REPORT_COLUMNS = ['model_name', 'dataset', 'transform', 'epochs', 'freeze_epochs',
                  'batch_size', 'lr_head', 'lr_backbone', 'weight_decay',
                  'best_val_loss', 'final_train_loss', 'train_time_s',
                  'weights_file']


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Stiffness regression (gradual unfreeze fz15/45, Yv1)')
    parser.add_argument('--dataset', choices=list(DATASET_TRANSFORMS),
                        default='datav5.2')
    parser.add_argument('--epochs', type=int, default=STRATEGY['epochs'])
    parser.add_argument('--freeze-epochs', type=int, default=STRATEGY['freeze_epochs'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--report', type=Path,
                        default=Path('training_report_regression.csv'))
    parser.add_argument('--debug', action='store_true',
                        help='smoke test: 2 epochs, small subset')
    return parser.parse_args(args)


def run_training(data_key: str, s: dict, args) -> dict:
    spec = DATASET_TRANSFORMS[data_key]
    cfg = dataset_config(data_key)
    stamp = datetime.now(tz=UTC).strftime('%d%m%y-%H%M%S')
    model_name = (f'resnet18_{data_key}_{spec["xv_tag"]}_Yv1_gradual_unfreeze'
                  f'_fz{s["freeze_epochs"]:02d}_{stamp}')
    if args.debug:
        model_name += '_debug'
    model_dir = save_root()/'stiffness'/model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {model_name} ===')
    datasets = build_datasets_regression(cfg, spec['transforms'])
    if args.debug:
        datasets['train'] = torch.utils.data.Subset(datasets['train'], range(8))
        datasets['val'] = torch.utils.data.Subset(datasets['val'], range(4))
    dataloaders = {phase: DataLoader(datasets[phase], batch_size=args.batch_size,
                                     shuffle=(phase == 'train'), num_workers=4)
                   for phase in ('train', 'val')}
    dataset_sizes = {phase: len(datasets[phase]) for phase in ('train', 'val')}

    model = resnet18(pretrained=True).to(TORCH_DEVICE)
    criterion = nn.MSELoss()
    weights_file = f'{model_name}_{s["epochs"]}epoch_AdamW.pt'
    checkpoint_path = model_dir/weights_file

    since = time.time()
    losses = {'train': [], 'val': []}
    stage_boundary = None

    # Stage 1: frozen backbone, head-only training.
    freeze_epochs = min(s['freeze_epochs'], s['epochs'] - 1)
    print(f'-- Stage 1: frozen backbone, {freeze_epochs} epochs, '
          f'head lr={s["lr_head"]} --')
    set_backbone_trainable(model, False)
    optimizer = build_optimizer(model, s, backbone_trainable=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=freeze_epochs)
    model, l1 = train_model(model, criterion, optimizer, scheduler,
                            dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                            num_epochs=freeze_epochs, device=TORCH_DEVICE)
    for phase in losses:
        losses[phase] += l1[phase]
    stage1_best = min(l1['val'])
    stage1_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Stage 2: unfreeze, differential LR (head lr/10), best-val checkpointing.
    stage2_epochs = s['epochs'] - freeze_epochs
    stage_boundary = freeze_epochs
    lr_head = s['lr_head']/10
    print(f'-- Stage 2: unfrozen, {stage2_epochs} epochs, '
          f'head lr={lr_head}, backbone lr={s["lr_backbone"]} --')
    set_backbone_trainable(model, True)
    optimizer = build_optimizer(model, s, backbone_trainable=True, lr_head=lr_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, stage2_epochs))
    model, l2 = train_model(model, criterion, optimizer, scheduler,
                            dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                            num_epochs=stage2_epochs, device=TORCH_DEVICE,
                            return_best_val=True)
    for phase in losses:
        losses[phase] += l2[phase]
    stage2_best = min(l2['val'])

    if stage1_best < stage2_best:
        print(f'Stage 1 best val loss {stage1_best:.4f} < stage 2 '
              f'{stage2_best:.4f}; keeping stage 1 weights')
        model.load_state_dict(stage1_state)
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Best-val weights: {checkpoint_path}')

    elapsed = time.time() - since

    # Loss curves with the unfreeze point marked.
    fig, ax = plt.subplots(figsize=[8, 5])
    for phase, values in losses.items():
        ax.plot(values, label=phase)
    ax.axvline(stage_boundary - 0.5, ls=':', c='k', alpha=.5, label='unfreeze')
    ax.set_title('Loss')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.tight_layout()
    fig.savefig(model_dir/f'{model_name}_loss.png', dpi=150)
    plt.close(fig)
    print(f'Saved: {model_dir/f"{model_name}_loss.png"}')

    # Final predictions with inference transforms (full datasets, not subsets).
    full_datasets = build_datasets_regression(
        cfg, {'train': spec['transforms']['val'], 'val': spec['transforms']['val']})
    preds_by_phase = {}
    for phase in ('train', 'val'):
        pred_loader = DataLoader(full_datasets[phase], batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        res = predict_regression(model, pred_loader, TORCH_DEVICE, desc=phase)
        preds_by_phase[phase] = res
        pd.DataFrame({
            'Image': full_datasets[phase].img_labels['Nucleus'].tolist(),
            'y_tgt': res['y_tgt'].ravel(), 'y_pred': res['y_pred'].ravel(),
            'true_kPa': np.exp(res['y_tgt'].ravel()*3.0 + 1.85),
            'pred_kPa': np.exp(res['y_pred'].ravel()*3.0 + 1.85),
        }).to_csv(model_dir/f'{model_name}_{phase}_preds.csv', index=False)
    plot_regression_predictions(preds_by_phase, model_dir/f'{model_name}_preds.png')

    row = {
        'model_name': model_name,
        'dataset': data_key,
        'transform': spec['xv_tag'],
        'epochs': s['epochs'],
        'freeze_epochs': s['freeze_epochs'],
        'batch_size': args.batch_size,
        'lr_head': s['lr_head'],
        'lr_backbone': s['lr_backbone'],
        'weight_decay': s['weight_decay'],
        'best_val_loss': min(losses['val']),
        'final_train_loss': losses['train'][-1],
        'train_time_s': round(elapsed, 1),
        'weights_file': weights_file,
    }
    append_report(row, args.report, REPORT_COLUMNS)
    return row


def main():
    args = parse_args(sys.argv[1:])
    torch.manual_seed(42)
    np.random.seed(42)
    print(f'Using device: {TORCH_DEVICE}')

    s = dict(STRATEGY)
    s['epochs'] = args.epochs
    s['freeze_epochs'] = args.freeze_epochs

    print(f'Strategy: gradual_unfreeze Yv1 on {args.dataset}: {s}')
    row = run_training(args.dataset, s, args)
    print('\n=== Summary ===')
    print(row)


if __name__ == '__main__':
    main()
