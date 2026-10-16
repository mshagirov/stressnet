'''Train regularized age classifiers (young vs adult) — v2.

Targets the overfitting seen in the train_age_classifier.py runs
(~0.99 train acc vs ~0.50 val acc). Five causes, five fixes:

1. No regularization       -> AdamW weight decay, dropout before the FC head,
                              label smoothing (criterion-level).
2. No LR scheduling        -> cosine annealing; differential LRs for the
                              pretrained backbone vs the new head.
3. Train/val domain shift  -> the val phase now uses an internal stratified
                              80/20 split of Training_Data; Prediction_Data is
                              NEVER trained on and is scored only once, after
                              training, as a held-out test set.
4. Weak augmentation       -> Xv4 transforms (random-resized crop, elastic,
                              brightness/contrast jitter, Gaussian noise) + MixUp.
5. Pretrained features destroyed early -> frozen-backbone and gradual-unfreeze
                              strategies.

Strategies (all ImageNet-pretrained, dropout head, label smoothing, AdamW):
    frozen_pretrained : backbone frozen for the whole run; only the head trains.
    gradual_unfreeze  : head-only for freeze_epochs, then full model with
                        differential LR (head lr_head/10, backbone lr_backbone).
    regularized       : full fine-tune with differential LR from epoch 0.
    mixup_augmented   : regularized + Xv4 strong transforms + MixUp.

Dataset locations, save locations and the label-file layout come from
stressnet.config (site_config.yaml / environment variables); see HELP.md.

Usage (from the repository root):
    uv run python train_age_classifier_v2.py                    # all runs
    uv run python train_age_classifier_v2.py --debug            # fast smoke test
    uv run python train_age_classifier_v2.py --datasets datav5 \
        --strategies regularized mixup_augmented
'''

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from classifiers import resnet18_classifier
from models import TORCH_DEVICE
from stressnet.config import DATASET_KEYS, dataset_config, save_root
from stressnet.data import TRAIN_TRANSFORMS, build_datasets, build_loaders
from stressnet.engine import (
    AGE_TO_IDX,
    build_optimizer,
    set_backbone_trainable,
    train_classifier,
)
from stressnet.metrics import accuracy, predict_classifier
from stressnet.plots import plot_classifier_predictions, plot_history
from stressnet.reporting import REPORT_COLUMNS, append_report

CLASS_NAMES = list(AGE_TO_IDX)

# Strategy hyperparameters. lr_backbone=0.0 means "keep backbone frozen".
STRATEGIES = {
    'frozen_pretrained': {
        'transform': 'Xv1', 'dropout': 0.5, 'label_smoothing': 0.1, 'weight_decay': 1e-4,
        'lr_head': 1e-3, 'lr_backbone': 0.0, 'freeze_epochs': None, 'epochs': 30,
        'patience': 8, 'mixup_alpha': 0.0,
    },
    'gradual_unfreeze': {
        'transform': 'Xv1', 'dropout': 0.5, 'label_smoothing': 0.1, 'weight_decay': 1e-4,
        'lr_head': 1e-3, 'lr_backbone': 1e-5, 'freeze_epochs': 10, 'epochs': 30,
        'patience': 10, 'mixup_alpha': 0.0,
    },
    'regularized': {
        'transform': 'Xv1', 'dropout': 0.5, 'label_smoothing': 0.1, 'weight_decay': 1e-4,
        'lr_head': 1e-4, 'lr_backbone': 1e-5, 'freeze_epochs': 0, 'epochs': 30,
        'patience': 10, 'mixup_alpha': 0.0,
    },
    'mixup_augmented': {
        'transform': 'Xv4', 'dropout': 0.5, 'label_smoothing': 0.1, 'weight_decay': 1e-4,
        'lr_head': 1e-4, 'lr_backbone': 1e-5, 'freeze_epochs': 0, 'epochs': 40,
        'patience': 10, 'mixup_alpha': 0.2,
    },
}


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Train regularized age classifiers (young/adult), v2')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_KEYS),
                        default=None)
    parser.add_argument('--strategies', nargs='+', choices=list(STRATEGIES),
                        default=None)
    parser.add_argument('--epochs', type=int, default=None,
                        help='override per-strategy epoch count')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-frac', type=float, default=0.2,
                        help='fraction of Training_Data held out for internal validation')
    parser.add_argument('--report', type=Path, default=Path('training_report_v2.csv'))
    parser.add_argument('--debug', action='store_true',
                        help='smoke test: 1 dataset x 1 strategy, 2 epochs, small subset')
    return parser.parse_args(args)


def run_training(data_key: str, strategy_name: str, args) -> dict:
    s = dict(STRATEGIES[strategy_name])
    if args.epochs is not None:
        s['epochs'] = args.epochs
        if s['freeze_epochs']:
            s['freeze_epochs'] = min(s['freeze_epochs'], max(1, args.epochs//3))

    model_name = f'resnet18_classifier_{data_key}_{s["transform"]}_{strategy_name}'
    if args.debug:
        model_name += '_debug'
    model_dir = save_root()/model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {model_name} ===')
    train_transform = TRAIN_TRANSFORMS[s['transform']]
    datasets = build_datasets(dataset_config(data_key), train_transform, args.val_frac)
    if args.debug:
        datasets['train'] = Subset(datasets['train'].dataset,
                                   datasets['train'].indices[:64])
        datasets['val'] = Subset(datasets['val'].dataset,
                                 datasets['val'].indices[:32])
        datasets['train_infer'] = datasets['val']
        datasets['val_infer'] = datasets['val']
        s['epochs'] = 2
        if s['freeze_epochs']:
            s['freeze_epochs'] = 1
    dataloaders = build_loaders(datasets, args.batch_size)
    dataset_sizes = {phase: len(datasets[phase]) for phase in ('train', 'val')}

    model = resnet18_classifier(num_classes=2, pretrained=True, dropout=s['dropout'])
    criterion = nn.CrossEntropyLoss(label_smoothing=s['label_smoothing'])
    weights_file = f'{model_name}_{s["epochs"]}epoch_AdamW.pt'
    checkpoint_path = model_dir/weights_file

    since = time.time()
    losses, accs = {'train': [], 'val': []}, {'train': [], 'val': []}
    stage_boundary = None

    if s['freeze_epochs'] is None or s['freeze_epochs'] > 0:
        # Stage 1: backbone frozen, head-only training.
        freeze_epochs = s['freeze_epochs'] if s['freeze_epochs'] else s['epochs']
        print(f'-- Stage 1: frozen backbone, {freeze_epochs} epochs, '
              f'head lr={s["lr_head"]} --')
        set_backbone_trainable(model, False)
        optimizer = build_optimizer(model, s, backbone_trainable=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=freeze_epochs)
        model, l1, a1, best_loss = train_classifier(
            model, criterion, optimizer, scheduler,
            dataloaders=dataloaders, dataset_sizes=dataset_sizes,
            label_to_idx=AGE_TO_IDX,
            num_epochs=freeze_epochs, device=TORCH_DEVICE,
            checkpoint_path=checkpoint_path,
            early_stopping_patience=(s['patience'] if s['freeze_epochs'] is None else None),
            mixup_alpha=s['mixup_alpha'],
            load_best_at_end=(s['freeze_epochs'] is None),
        )
        for phase in losses:
            losses[phase] += l1[phase]
            accs[phase] += a1[phase]

    if s['freeze_epochs'] is not None:
        # Stage 2 (gradual_unfreeze) or single-stage full fine-tune.
        stage2_epochs = s['epochs'] - (s['freeze_epochs'] or 0)
        if s['freeze_epochs']:
            stage_boundary = s['freeze_epochs']
            # smaller head LR once the backbone is unfrozen
            lr_head = s['lr_head']/10
            print(f'-- Stage 2: unfrozen, {stage2_epochs} epochs, '
                  f'head lr={lr_head}, backbone lr={s["lr_backbone"]} --')
        else:
            lr_head = s['lr_head']
            best_loss = float('inf')
            print(f'-- Full fine-tune: {stage2_epochs} epochs, '
                  f'head lr={lr_head}, backbone lr={s["lr_backbone"]} --')
        set_backbone_trainable(model, True)
        optimizer = build_optimizer(model, s, backbone_trainable=True, lr_head=lr_head)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage2_epochs))
        model, l2, a2, best_loss = train_classifier(
            model, criterion, optimizer, scheduler,
            dataloaders=dataloaders, dataset_sizes=dataset_sizes,
            label_to_idx=AGE_TO_IDX,
            num_epochs=stage2_epochs, device=TORCH_DEVICE,
            checkpoint_path=checkpoint_path,
            initial_best_loss=best_loss,
            early_stopping_patience=s['patience'],
            mixup_alpha=s['mixup_alpha'],
            load_best_at_end=True,
        )
        for phase in losses:
            losses[phase] += l2[phase]
            accs[phase] += a2[phase]

    elapsed = time.time() - since
    stopped_epoch = len(losses['train'])
    print(f'Best-val weights: {checkpoint_path}')

    plot_history(losses, accs, model_dir/f'{model_name}_loss.png',
                 stage_boundary=stage_boundary)

    # Final predictions with inference transforms; Prediction_Data (heldout)
    # is scored only here, after training is fully done.
    model = model.to(TORCH_DEVICE).eval()
    preds_by_phase = {}
    for phase, key in (('train', 'train_infer'), ('val', 'val_infer'), ('heldout', 'heldout')):
        pred_loader = DataLoader(datasets[key], batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        preds_df = predict_classifier(model, pred_loader, TORCH_DEVICE,
                                      class_names=CLASS_NAMES, desc=phase)
        preds_df.to_csv(model_dir/f'{model_name}_{phase}_preds.csv', index=False)
        preds_by_phase[phase] = preds_df
    plot_classifier_predictions(preds_by_phase, model_dir/f'{model_name}_preds.png',
                                CLASS_NAMES)

    row = {
        'model_name': model_name,
        'dataset': data_key,
        'strategy': strategy_name,
        'transform': s['transform'],
        'epochs': s['epochs'],
        'batch_size': args.batch_size,
        'lr_head': s['lr_head'],
        'lr_backbone': s['lr_backbone'],
        'weight_decay': s['weight_decay'],
        'label_smoothing': s['label_smoothing'],
        'dropout': s['dropout'],
        'mixup_alpha': s['mixup_alpha'],
        'best_val_loss': min(losses['val']),
        'final_train_loss': losses['train'][-1],
        'final_val_loss': losses['val'][-1],
        'final_train_acc': accs['train'][-1],
        'final_val_acc': accs['val'][-1],
        'best_train_acc': max(accs['train']),
        'best_val_acc': max(accs['val']),
        'heldout_acc': accuracy(preds_by_phase['heldout']),
        'stopped_epoch': stopped_epoch,
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

    if args.debug:
        datasets = args.datasets or list(DATASET_KEYS)[:1]
        strategies = args.strategies or ['regularized']
    else:
        # Historical default: the tiled datasets only. `datav5.2` is a
        # grid-averaged regression dataset; pass it explicitly if wanted.
        datasets = args.datasets or ['datav5', 'datav6']
        strategies = args.strategies or list(STRATEGIES)

    results = []
    for data_key in datasets:
        for strategy_name in strategies:
            results.append(run_training(data_key, strategy_name, args))

    print('\n=== Summary ===')
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
