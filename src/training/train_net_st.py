"""
Train PyTorch network for one epoch and report the results back to the tuner.
To be used with the syne-tune library.
"""
import time

import numpy as np
from tqdm import tqdm
import torch

SUMMARY = False # Must be false before training, because of encoding error with syne_tune
if SUMMARY:
    from torchinfo import summary

from dl_utils import evaluate_metrics

def train_net(train_loader, val_loader, state, params, epoch):
    batch_size = params['data']['batch_size']
    input_dim = params['data']['input_dim']
    input_size = [batch_size] + input_dim
    input_size = tuple(input_size)

    # Load model
    model = state['model']
    opt = state['optimizer']
    scheduler = state['scheduler']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if SUMMARY:
        summ = str(summary(model, input_size=input_size))
        print(summ)

    # Loss function
    if params['loss'] == 'BCEWithLogitsLoss':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif params['loss'] == 'MSELoss':
        loss_fn = torch.nn.MSELoss()
    elif params['loss'] == 'CrossEntropyLoss':
        label_smoothing = params.get('label_smoothing', 0.0)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise NotImplementedError('Loss function not supported')

    # Out activation
    if params['out_activation'] == 'Sigmoid':
        out_activation = torch.nn.Sigmoid()
    elif params['out_activation'] == 'None':
        out_activation = torch.nn.Identity()
    elif params['out_activation'] == 'Softmax':
        out_activation = torch.nn.Softmax(dim=1)
    else:
        raise NotImplementedError('Activation function not supported')

    train_loss = []
    val_loss = []

    model.train()

    # Training
    loss_per_step = []

    first = True
    enum_time_start = time.time()
    train_time_start = enum_time_start
    start_time = enum_time_start

    with tqdm(train_loader, unit='batch',) as per_epoch:
        for x,y in per_epoch:
            if first:
                enum_time_end = time.time()
                first = False
            opt.zero_grad(set_to_none=True)
            per_epoch.set_description(f"Epoch: {epoch}")
            x,y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss_per_step.append(loss.item())

            loss.backward()
            opt.step()
            per_epoch.set_postfix(train_loss=loss.item())

    train_loss.append(sum(loss_per_step)/len(loss_per_step))

    val_loss_per_step = []
    train_time_end = time.time()

    # Evaluation
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, unit='batch',) as per_val_epoch:
            y_true = []
            y_pred = []
            for x_val,y_val in per_val_epoch:
                per_val_epoch.set_description("Model Evaluation: ")
                x_val,y_val = x_val.to(device), y_val.to(device)
                y_hat_val = model(x_val)

                y_pred.append(out_activation(y_hat_val).cpu().detach().numpy())

                loss_val = loss_fn(y_hat_val, y_val)
                y_hat_val = (out_activation(
                    y_hat_val).cpu().detach().numpy()>=.5).astype(int)
                val_loss_per_step.append(loss_val.item())
                per_val_epoch.set_postfix(val_loss=loss_val.item())
                y_true.append(y_val.cpu().detach().numpy())

    val_loss.append(sum(val_loss_per_step)/len(val_loss_per_step))
    losses = {'val_loss': val_loss[-1], 'train_loss': train_loss[-1]}
    metrics = evaluate_metrics(params, np.concatenate(y_true), np.concatenate(y_pred))
    metrics.update(losses)

    if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler.step(val_loss[-1])
    else:
        scheduler.step()
    end_time = time.time()

    print(f"Training end. Enum: {enum_time_end - enum_time_start}, Train: {train_time_end - train_time_start}, Total: {end_time - start_time}")
    print (f"loss: {train_loss[-1]:.4e}, val_loss: {val_loss[-1]:.4e} ")

    return metrics