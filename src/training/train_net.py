"""
Train netwrok for n epochs, even resuming a paused trial.
Log metrics. Also supports checpointing and loading.
To be used when optimizing hyperparameters with libraries
such as Optuna or SMAC.
"""
from pathlib import Path
import time
from datetime import datetime

from tqdm import tqdm
import torch
import optuna
from torchinfo import summary
import numpy as np

from load_model import load_model
import dl_utils

def train_net(train_loader, val_loader, params, logger,
        trial=None, previous_epoch=0, end_epoch=0, checkpoint_path=None, run=0):
    batch_size = params['data']['batch_size']
    input_dim = params['data']['input_dim']
    input_size = [batch_size] + input_dim
    input_size = tuple(input_size)

    # Load model
    model = load_model(params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    #summary(model, input_size=input_size)

    # Optimizer
    lr = params['learning_rate']
    if params['optimizer'] == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif params['optimizer'] == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    elif params['optimizer'] == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('Optimizer not supported')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=5)

    # Set the number of training epochs
    if end_epoch != 0:
        n_epochs = end_epoch - previous_epoch
    else:
        n_epochs = params['epochs']
        previous_epoch = 0
        end_epoch = n_epochs

    # Scheduler  TODO: Test if saving and loading works correctly
    if 'decay' not in params:
        pass
    elif params['decay'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params['epochs'], eta_min=params['eta_min']*lr)
    else:
        raise NotImplementedError('Scheduler not supported')

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

    # Set checkpoint, experiment id, epochs
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        last_dir = path.parent.name
        name = checkpoint_path.split('_')[-1]
        id = f"{last_dir}_{name}"
        if previous_epoch > 0:
            load_checkpoint(model, opt, scheduler, checkpoint_path)
        n_epochs = end_epoch - previous_epoch
    else:
        id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model.to(device)

    # Logging
    logger.init_run(model, id, params)
    train_loss = []
    val_loss = []

    for epoch in range(previous_epoch, end_epoch):
        start_time = time.time()
        # Training
        loss_per_step = []
        model.train()

        with tqdm(train_loader, unit='batch',) as per_epoch:
            for x,y in per_epoch:
                opt.zero_grad(set_to_none=True)
                per_epoch.set_description(f"Epoch: {epoch+1}/{n_epochs}")
                x,y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                loss_per_step.append(loss.item())

                loss.backward()
                opt.step()
                per_epoch.set_postfix(train_loss=loss.item())
            #if scheduler.__module__ == 'torch.optim.lr_scheduler':
            #    scheduler.step()

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
        metrics = dl_utils.evaluate_metrics(params, np.concatenate(y_true), np.concatenate(y_pred))
        metrics.update(losses)
        end_time = time.time()
        elapsed_time = end_time - start_time

        if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(val_loss[-1])
        else:
            scheduler.step()

        print (f"loss: {train_loss[-1]:.4e}, val_loss: {val_loss[-1]:.4e} ")

        logger.log(metrics, epoch+1, elapsed_time)

        # Early-stopping for Optuna
        if trial is not None:
            trial.report(val_loss[-1], epoch)
            if trial.should_prune():
                logger.finish()
                raise optuna.TrialPruned()

    logger.finish()

    # Save model with the config file to enable loading
    if 'save_model' in params and params['save_model']:
        savedir = logger.get_logdir() / 'models'
        if not savedir.exists():
            savedir.mkdir(parents=True, exist_ok=True)
        model_path = savedir / f'model_{id}.pt'
        config_path = savedir / f'model_{id}.yaml'
        torch.save(model.state_dict(), model_path)
        logger.save_config(params, config_path)

    # Save checkpoint (for optimization purposes) and return evaluation info
    if checkpoint_path is not None:
        save_checkpoint(model, opt, scheduler, end_epoch, checkpoint_path)
        evaluated_info = [
            {'epoch': i + previous_epoch + 1, 'metric': val_loss[i], 'loss': train_loss[i]
            } for i in range(len(val_loss))
        ]
        return evaluated_info
    else:
        return val_loss[-1]

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path):
    if scheduler is None:
        scheduler_dict = dict()
    else:
        scheduler_dict = scheduler.state_dict()
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler_dict
    }, checkpoint_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
