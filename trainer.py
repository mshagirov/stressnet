'''
Training loop shared by the regression and classification drivers.

The module-level `device` (and therefore `train_model`'s default) follows
`models.TORCH_DEVICE`, i.e. CUDA when available and CPU otherwise; callers can
still pass an explicit device.
'''

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from models import TORCH_DEVICE

device = TORCH_DEVICE

def train_model(
    model: torch.nn.Module,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=25,
    return_best_val=False,
    device=device
):
    '''
    Run `num_epochs` of train/val epochs over `dataloaders`.

    Batches may be plain image tensors or `(image, age_loc, ...)` tuples, which
    is auto-detected and forwarded as `model(image, age_loc)`.

    Returns `(model, losses)`; `losses` maps each phase in `dataset_sizes` to
    its per-epoch loss. With `return_best_val=True` the weights with the best
    validation loss are loaded back into `model` before returning.
    '''
    since = time.time()
    model = model.to(device)

    phases = list(dataset_sizes.keys())
    losses = {k:[] for k in phases}
    
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir)/'best_model_params.pt'

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = float('inf')

        for epoch in range(num_epochs):
            print(f'Epoch {epoch:0>3}/{num_epochs - 1:0>3}; ', end='')

            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for batch in dataloaders[phase]:
                    sample, labels = batch
                    batch_size = sample.size(0) if isinstance(sample, torch.Tensor) else sample[0].size(0)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if isinstance(sample, (tuple, list)):
                            # e.g., StiffnessDatasetWithActinAgeLoc yields (image, age_loc, ...)
                            image, x_ageloc = sample[0].to(device), sample[1].to(device)
                            outputs = model(image, x_ageloc)
                        else:
                            # e.g., StiffnessDataset yields image
                            outputs = model(sample.to(device))
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * batch_size
                if (phase == 'train') and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f}; ',end='')
                losses[phase].append(epoch_loss)

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    if return_best_val:
                        torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print('-' * 10)
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val loss: {best_loss:4f}')

        if return_best_val:
            # load best model weights
            model.load_state_dict(
                torch.load(best_model_params_path, weights_only=True, map_location=device)
            )
    return model, losses
