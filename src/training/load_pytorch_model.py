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
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif params['optimizer'] == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    elif params['optimizer'] == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('Optimizer not supported')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=5)

    # TODO: max epochs in non syne-tune case
    if 'decay' not in params:
        pass
    elif params['decay'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params['max_epochs'], eta_min=params['eta_min']*lr)
    else:
        raise NotImplementedError('Scheduler not supported')

    state['model'] = model
    state['optimizer'] = opt
    state['scheduler'] = scheduler

    return state