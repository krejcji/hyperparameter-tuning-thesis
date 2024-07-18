"""
Load the model, optimizer, and scheduler.
By default, the learning rate decay is set to ReduceLROnPlateau.
"""
from load_model import load_model
import torch

def model_and_optimizer(params):
    state = dict()
        # Load model
    model = load_model(params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer
    lr = params['learning_rate']
    if params['optimizer'] == 'SGD':
        weight_decay = params.get('weight_decay', 0.0)
        momentum = params.get('momentum', 0.0)
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif params['optimizer'] == 'RMSprop':
        weight_decay = params.get('weight_decay', 0.0)
        momentum = params.get('momentum', 0.0)
        opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif params['optimizer'] == 'AdamW':
        weight_decay = params.get('weight_decay', 0.01)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError('Optimizer not supported')

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.2, patience=5)

    if 'decay' not in params:
        pass
    elif params['decay'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params['max_epochs'], eta_min=params['eta_min']*lr)
    elif params['decay'] == 'ReduceLROnPlateau':
        ... # It is the default option
    else:
        raise NotImplementedError('Scheduler not supported')

    state['model'] = model
    state['optimizer'] = opt
    state['scheduler'] = scheduler

    return state