import sys
import os
from pathlib import Path

import optuna
import wandb
import yaml
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical

from load_data import load_data
from train_net import train_net

# Implemented optimizers - Optuna and SMAC

# Optuna objective function
def objective_optuna(trial, trainloader=None, valloader=None, config=None):
    # Suggest new parameter values
    params = {}
    for param in config['tunable_params']:
        name = param['name']
        if param['type'] == 'float':
            params[name] = trial.suggest_float(param['name'], param['low'], param['high'], log=param['log'])
        elif param['type'] == 'int':
            params[name] = trial.suggest_int(param['name'], param['low'], param['high'])
        elif param['type'] == 'categorical':
            params[name] = trial.suggest_categorical(param['name'], param['choices'])
        else:
            raise ValueError(f"Unknown parameter type: {param['type']}")

    # Prepare non-optimized parameters from config file
    for param in config['fixed_params']:
        params[param['name']] = param['value']

    wandb.init(project='metacentrum_exp', group=config['experiment'], config=params)

    loss = train_net(trainloader, valloader, params, config, trial)

    wandb.finish()

    return loss

def optimize_optuna(config, trainloader, valloader):
    # Prepare database
    if not os.path.exists(f"experiments/{exp_name}/outputs"):
        os.makedirs(f"experiments/{exp_name}/outputs")
    storage_url = f"sqlite:///experiments/{exp_name}/outputs/{exp_name}.db"

    study = optuna.create_study(storage=storage_url, direction='minimize', load_if_exists=True, study_name=exp_name)
    study.optimize(lambda trial: objective_optuna(trial, trainloader=trainloader, valloader=valloader, config=config),
                    n_trials=config['hp_optimizer']['n_trials'])

# SMAC objective function
def objective_smac(config, seed: int=0, trainloader=None, valloader=None, configuration=None):
    params = dict(config)
    for param in configuration['fixed_params']:
        params[param['name']] = param['value']

    wandb.init(project='metacentrum_exp', group=configuration['experiment'], config=params)

    loss = train_net(trainloader, valloader, params, configuration)

    return loss

def optimize_smac(config, trainloader, valloader):
    configspace = ConfigurationSpace()
    for param in config['tunable_params']:
        name = param['name']
        if param['type'] == 'float':
            hp = Float(name, (param['low'], param['high']), log=param['log'])
        elif param['type'] == 'int':
            hp = Integer(name, (param['low'], param['high']))
        elif param['type'] == 'categorical':
            hp = Categorical(name, param['choices'])
        else:
            raise ValueError(f"Unknown parameter type: {param['type']}")
        configspace.add_hyperparameter(hp)

    scenario = Scenario(configspace, n_trials=config['hp_optimizer']['n_trials'])

    smac = HyperparameterOptimizationFacade(
        scenario,
        lambda x, seed:objective_smac(x, seed, trainloader=trainloader, valloader=valloader, configuration=config),
        overwrite=True,)
    incumbent = smac.optimize()

if __name__ == '__main__':
    seed = 42
    # Experiment name has to be defined
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "kaggle_exp"

    # Load the configuration file
    with open(Path('experiments') / exp_name /'config.yaml') as file:
        config = yaml.safe_load(file)

    # Load the data
    trainloader, valloader = load_data(config)

    # Branch on the optimizer (supported - Optuna, SMAC)
    if config['hp_optimizer']['name'] == 'Optuna':
        optimize_optuna(config, trainloader, valloader)
    elif config['hp_optimizer']['name'] == 'SMAC':
        optimize_smac(config, trainloader, valloader)
    else:
        raise ValueError(f"Unknown optimizer: {config['hp_optimizer']['name']}")