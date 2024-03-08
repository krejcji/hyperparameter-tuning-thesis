import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import pathlib

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

def evaluate_accuracy(net, data_loader, verbose=False):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        for data in data_loader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    if verbose:
        print(f'Accuracy: {accuracy:.3f}')
    return accuracy

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

def evaluate_auc(y_true, y_pred, verbose=False):
    auc = roc_auc_score(y_true[:,:5], y_pred[:,:5], average='macro') # TODO: Fix
    if verbose:
        print(f'Macro AUC: {auc}')
    return auc

class CSV_Logger():
    def __init__(self, path, header):
        self.header = header
        i = 0

        dir = path.parent
        name = path.name
        filename = name

        while((dir / (filename + '.csv')).exists()):
            filename = name + f'_{i}'
            i = i + 1

        self.filename = dir / (filename + '.csv')

        print(f'Filename: {self.filename}')

        with open(self.filename, 'w') as f:
            f.write(header + '\n')

    def log_error(self, *args):
        with open(self.filename, 'a') as f:
            for i in range(len(args)):
                if i != 0:
                    f.write(', ')
                f.write(str(args[i]))
            f.write('\n')