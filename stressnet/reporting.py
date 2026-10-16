'''CSV report bookkeeping shared by the training drivers.

Every driver appends one row per run to its ``training_report*.csv`` in the
working directory; the report file is created with a header on first append.
The column layout is fixed by the caller so that the historical CSV schemas
keep working: :data:`REPORT_COLUMNS` is the age-classifier schema, and the
categorised / stiffness drivers use it plus a trailing ``heldout_macro_acc``.
'''

from __future__ import annotations

import csv
from pathlib import Path

# Age-classifier report schema (v2/v3); the categorised/stiffness drivers use
# this plus 'heldout_macro_acc'.
REPORT_COLUMNS = [
    'model_name', 'dataset', 'strategy', 'transform', 'epochs', 'batch_size',
    'lr_head', 'lr_backbone', 'weight_decay', 'label_smoothing', 'dropout',
    'mixup_alpha', 'best_val_loss', 'final_train_loss', 'final_val_loss',
    'final_train_acc', 'final_val_acc', 'best_train_acc', 'best_val_acc',
    'heldout_acc', 'stopped_epoch', 'train_time_s', 'weights_file',
]


def append_report(row: dict, report_path: Path, columns):
    '''Append `row` to `report_path`, writing the header if the file is new.

    `columns` is the full ordered column list of the report schema.
    '''
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = report_path.exists()
    with report_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f'Report row appended: {report_path}')
