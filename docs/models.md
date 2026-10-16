---
title: Models
subtitle: "Checkpoints, reuse, fine-tuning"
---

# Models and checkpoints

## What gets saved

A run saves a **state dict** (`model.state_dict()`) with
`torch.save`, not a pickled module:

```python
torch.save(model.state_dict(), checkpoint_path)
```

Classification runs keep the best-val weights (early-stopping checkpoint) and
write them at the end; the 2-way categorised driver saves last weights unless
`--save-best` is passed. Regression keeps the best-val checkpoint on the
`Prediction_Data` phase.

Checkpoints live under `save_root()` (see [configuration.md](configuration.md)),
one directory per run. Regression runs are nested in a `stiffness/`
sub-directory.

## Naming convention

```text
<run_name>/<run_name>_<epochs>epoch_AdamW.pt
```

`<run_name>` encodes dataset, transform version and recipe:

| Task | Run name pattern |
| --- | --- |
| Age (v2) | `resnet18_classifier_<data>_Xv<1\|4>_<strategy>` |
| Age (v3) | `resnet18_classifier_<data>_Xv1_gradual_unfreeze_<...tags>` |
| 2-way category | `resnet18_<data>_CategorisedStiffness_<ddmmyy-HHMMSS>` |
| 4-way category | `resnet18_stiffness_classifier_<data>_Xv1_..._catstrat` |
| 3-way category | `resnet18_stiffness_classifier_v2_<data>_Xv1_..._catstrat` |
| Regression | `resnet18_<data>_Xv<1\|4>_Yv1_gradual_unfreeze_fz<NN>_<ddmmyy-HHMMSS>` |

The `Xv<N>` tag is the input-transform version and `Yv<N>` the target-transform
version (regression only), both defined in `transforms.py`:
Xv1–Xv4 are input transform sets, Yv1–Yv3 are target transforms. The
`_<epochs>epoch_AdamW.pt` suffix records the total epoch count and optimizer.

## Loading a checkpoint

Use the loader that matches how the head was built.

**Regression** (scalar head, `fc = Linear(512, 1)`):

```python
from models import resnet18, TORCH_DEVICE

model = resnet18(weights_path="<run>/<run>_45epoch_AdamW.pt").to(TORCH_DEVICE)
model.eval()
```

**Classification** (`fc = Linear(512, num_classes)`):

```python
from classifiers import resnet18_classifier, resnet34_classifier
from models import TORCH_DEVICE

model = resnet18_classifier(
    num_classes=2,                    # must match the run
    weights_path="<run>/<run>_40epoch_AdamW.pt",
).to(TORCH_DEVICE)
```

`resnet34_classifier` has the same signature.

### Head-shape caveats

The loader reconstructs the head *before* calling `load_state_dict`, so the head
shape must match the checkpoint exactly:

- `models.resnet18` expects `fc` to be a single `Linear(512, 1)`. Use
  `models.resnet18_seq` instead if the checkpoint's `fc` is a `Sequential`
  (as produced by the `fc_layers` helper). Passing the wrong one fails on
  `load_state_dict` with a key/shape mismatch.
- `classifiers.resnet18_classifier` builds `fc = Linear(512, num_classes)`, or
  `Sequential(Dropout, Linear)` when `dropout > 0`. Checkpoints from a dropout
  head must be reloaded with the **same** `dropout` value (and `num_classes`).
- Both `resnet18` and the classifiers accept `pretrained=False` for random
  initialisation when no checkpoint is involved.

### Device and dtype

`models.TORCH_DEVICE` selects CUDA when available and CPU otherwise, and is the
default device for both `trainer.train_model` and
`stressnet.engine.train_classifier` (each moves the model onto the device).
PyTorch/ torchvision are pinned as a pair (`torch==2.3.1`,
`torchvision==0.18.1`) — bump them together only, since their C++/ABI pairing is
version-sensitive.

## Fine-tuning from an existing checkpoint

Starting a new run from a saved checkpoint is just "construct with
`weights_path`, then train". The training drivers do exactly this internally
except from ImageNet weights; to continue from your own:

```python
from classifiers import resnet18_classifier
from models import TORCH_DEVICE
from stressnet.engine import build_optimizer, set_backbone_trainable, train_classifier

model = resnet18_classifier(num_classes=2, weights_path="<run>/<run>.pt")
model.to(TORCH_DEVICE)

# `strategy` holds the LR/weight-decay values used to build the optimizer
# groups, e.g. {"lr_head": 1e-3, "lr_backbone": 1e-5, "weight_decay": 1e-4}.

# Stage 1: freeze the backbone, train only the head.
set_backbone_trainable(model, False)
optimizer = build_optimizer(model, strategy, backbone_trainable=False)   # head lr 1e-3
# ... train_classifier(model, criterion, optimizer, scheduler, ...)

# Stage 2: unfreeze and fine-tune with a differential LR.
set_backbone_trainable(model, True)
optimizer = build_optimizer(model, strategy, backbone_trainable=True, lr_head=1e-4)
```

`set_backbone_trainable` flips `requires_grad` on everything except `fc.*`;
`build_optimizer` then gives the head and the backbone separate learning rates.
This is the same gradual-unfreeze schedule every driver uses: head-only first
(at the higher head LR), then full fine-tuning with head LR divided by 10 and
backbone LR `1e-5`.

To instead fine-tune a whole new *task* head (different `num_classes`) on top of
a checkpoint, load the backbone weights but size the head for the new task; a
straight `load_state_dict` of the old head will not fit.

See also: [training.md](training.md) · [reproduction.md](reproduction.md)
