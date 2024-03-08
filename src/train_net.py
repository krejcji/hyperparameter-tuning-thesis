from tqdm import tqdm
import torch
import wandb
import optuna
from torchinfo import summary
import numpy as np

from load_model import load_model
import dl_utils

def train_net(train_loader, val_loader, params, config, trial=None):
    n_epochs = params['epochs']
    batch_size = config['data']['batch_size']
    input_channels = config['data']['input_dim']
    input_length = config['data']['input_length']

    model = load_model(config)
    summary(model, input_size=(batch_size, input_channels, input_length))

    wandb.watch(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if params['optimizer'] == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    else:
        raise NotImplementedError('Optimizer not supported')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=5)

    if params['loss'] == 'BCEWithLogitsLoss':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif params['loss'] == 'MSELoss':
        loss_fn = torch.nn.MSELoss()
    else:
        raise NotImplementedError('Loss function not supported')

    if params['out_activation'] == 'Sigmoid':
        out_activation = torch.nn.Sigmoid()
    elif params['out_activation'] == 'None':
        out_activation = torch.nn.Identity()
    else:
        raise NotImplementedError('Activation function not supported')

    train_loss = []
    val_loss = []

    for epoch in range(n_epochs):
        # Training
        loss_per_step = []
        model.train()

        with tqdm(train_loader, unit='batch',) as per_epoch:
            for x,y in per_epoch:
                opt.zero_grad(set_to_none=True)
                per_epoch.set_description(f"Epoch: {epoch+1}/{n_epochs}")
                x,y = x.to(device, torch.float32), y.to(device, torch.float32)
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                loss_per_step.append(loss.item())

                loss.backward()
                opt.step()
                per_epoch.set_postfix(train_loss=loss.item())
            if scheduler.__module__ == 'torch.optim.lr_scheduler':
                scheduler.step(1)
        train_loss.append(sum(loss_per_step)/len(loss_per_step))

        val_loss_per_step = []

        # Evaluation
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, unit='batch',) as per_val_epoch:
                y_true = []
                y_pred = []
                for x_val,y_val in per_val_epoch:
                    per_val_epoch.set_description("Model Evaluation: ")
                    x_val,y_val = x_val.to(device, torch.float32
                                        ), y_val.to(device,
                                                    torch.float32)
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
        metrics = dl_utils.evaluate_metrics(config, np.concatenate(y_true), np.concatenate(y_pred))
        metrics.update(losses)

        wandb.log(metrics, step=epoch)

        # Early-stopping
        if 'pruning' in config['hp_optimizer'] and config['hp_optimizer']['pruning']:
            if config['hp_optimizer']['name'] == 'Optuna':
                if trial is None:
                    raise ValueError('Trial object not provided')
                trial.report(val_loss[-1], epoch)
                # Don't prune first 4 epochs because possible unstability
                if epoch > 4 and trial.should_prune():
                    wandb.finish()
                    raise optuna.TrialPruned()

    wandb.finish()
    return val_loss[-1]
