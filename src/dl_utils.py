import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np

def should_prune_ptbxl(train_loss, val_loss):
    return np.var(val_loss[-5:]) < 0.2e-6 or \
            val_loss[-1]-train_loss[-1] > 0.6 or \
            val_loss[-1] > val_loss[-2] and val_loss[-2] > val_loss[-3] and val_loss[-3] > val_loss[-4]

def evaluate_mse(net, criterion, testloader, verbose=False):
    with torch.no_grad():
        net.eval()
        loss = 0.0
        iter = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            y_pred = net(inputs)
            loss += criterion(y_pred, labels)
            iter += 1
    mean_loss = loss / iter

    if verbose:
        print(f'Mean loss: {mean_loss}')
    return mean_loss

def evaluate_metrics(config, y_true, y_pred):
    metrics = config['metrics']
    results = {}
    for metric in metrics:
        if metric == 'accuracy':
            results['accuracy'] = evaluate_accuracy(y_true, y_pred)
        elif metric == 'mse':
            results['mse'] = evaluate_mse(y_true, y_pred)
        elif metric == 'macro_auc':
            results['macro_auc'] = evaluate_auc(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return results

def evaluate_accuracy(y_true, y_pred, verbose=False):
    accuracy = accuracy_score(y_true, y_pred.argmax(axis=1))
    if verbose:
        print(f'Accuracy: {accuracy}')
    return accuracy

def evaluate_auc(y_true, y_pred, verbose=False):
    auc = roc_auc_score(y_true[:,:5], y_pred[:,:5], average='macro') # TODO: Fix
    if verbose:
        print(f'Macro AUC: {auc}')
    return auc