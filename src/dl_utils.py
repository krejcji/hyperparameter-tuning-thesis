"""
Contains several utility function for training including:
- helper function for evaluating metrics
    - MSE metric
    - AUC metric
"""
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
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
    try:
        for metric in metrics:
            if metric == 'accuracy':
                results['accuracy'] = evaluate_accuracy(y_true, y_pred)
            elif metric == 'mse':
                results['mse'] = evaluate_mse(y_true, y_pred)
            elif metric == 'macro_auc':
                results['macro_auc'] = evaluate_auc(y_true, y_pred)
            elif metric == 'hamming_dist':
                results['hamming_dist'] = hamming_dist(y_true, y_pred)
            elif metric == 'debug':
                debug_metrics(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
    except Exception as e:
        print(f"Error evaluating metrics: {e}")
        for metric in metrics:
            results[metric] = 0.0
    return results

def debug_metrics(y_true, y_pred):
    # Calculate per class accuracy
    avg = 0
    for i in range(y_true.shape[1]):
        acc = accuracy_score(y_true[:,i], y_pred[:,i]>0.5)
        avg = avg + acc
        print(f'Class {i}: {acc}')
    print (f'Average: {avg/y_true.shape[1]}')


def hamming_dist(y_true, y_pred, verbose=True):
    preds = F.sigmoid(torch.from_numpy(y_pred))
    hl = 1-hamming_loss(y_true, preds > 0.5)
    if verbose:
        print(f'Hamming loss: {hl}')
    return hl

def evaluate_accuracy(y_true, y_pred, verbose=False):
    accuracy = accuracy_score(y_true, y_pred.argmax(axis=1))
    if verbose:
        print(f'Accuracy: {accuracy}')
    return accuracy

def evaluate_auc(y_true, y_pred, verbose=False):
    preds = F.sigmoid(torch.from_numpy(y_pred))
    auc = roc_auc_score(y_true, preds, average='macro')
    if verbose:
        print(f'Macro AUC: {auc}')
    return auc

def get_predictions(net, dataloader):
    net.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            y_true.append(labels)
            y_pred.append(net(inputs))
    return torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)