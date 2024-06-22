import argparse
import time
import yaml
from pathlib import Path

import numpy as np
import torch
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, lograndint, loguniform, uniform, choice
from syne_tune.optimizer.baselines import (
    ASHA,
    SyncBOHB,
    DEHB,
    DyHPO,
    MOBSTER,
    HyperTune,
    RandomSearch
)

from load_data import load_data

def run_hpo(config, args, seed):
    # Hyperparameter search space to consider
    config_space = dict()
    config_space.update(config)

    # Parse the fixed and tunable parameters from config
    for param in config['fixed_params']:
            config_space[param['name']] = param['value']

    for param in config['tunable_params']:
        name = param['name']
        if param['type'] == 'float':
            if param['log']:
                config_space[name] = loguniform(param['low'], param['high'])
            else:
                config_space[name] = uniform(param['low'], param['high'])
        elif param['type'] == 'int':
            if param['log']:
                config_space[name] = lograndint(param['low'], param['high'])
            else:
                config_space[name] = randint(param['low'], param['high'])
        elif param['type'] == 'categorical':
            config_space[name] = choice(param['choices'])
        else:
            raise ValueError(f"Unknown parameter type: {param['type']}")

    config_space['max_epochs'] = config_space['epochs']

    # Common scheduler arguments
    method_kwargs = dict(
        metric=config.get('optimization_metric', 'val_loss'),
        mode=config.get('optimization_mode', 'min'),
        max_resource_attr='epochs',
        random_seed=seed
    )

    # Choose HPO method
    if args.method != 'RS':
        method_kwargs['resource_attr'] = 'epoch'

    if args.method == 'RS':
        scheduler=RandomSearch(config_space, **method_kwargs)
    elif args.method == 'ASHA':
        scheduler=ASHA(
            config_space,
            type="stopping",
            **method_kwargs
        )
    elif args.method == 'DEHB':
        scheduler = DEHB(
            config_space,
            **method_kwargs
        )
    elif args.method == 'BOHB':
        scheduler = SyncBOHB(
            config_space,
            #type="promotion",
            **method_kwargs
        )
    elif args.method == 'DyHPO':
        scheduler = DyHPO(
            config_space,
            search_options={"debug_log": True},
           # grace_period=2, # Minimal number of epochs to run for a new solution
            **method_kwargs
        )
    elif args.method == 'MOBSTER':
        scheduler = MOBSTER(
            config_space,
            type="promotion",
            **method_kwargs
        )
    elif args.method == 'HyperTune':
        scheduler = HyperTune(
            config_space,
            **method_kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.method}")

    budget = config.get('budget', 0) # Number of full evaluations to perform
    max_num_evaluations = config_space['epochs'] * budget if budget > 0 else None
    max_wallclock_time = config.get('max_wallclock_time', 86400)

    metadata = { 'experiment_tag': args.experiment_tag,
                 'experiment_definition': args.experiment_definition,
                 'seed': seed,
                 'budget': budget,
                 'max_resource': config_space['epochs'],
                 'max_wallclock_time': max_wallclock_time,
                 'max_num_evaluations': max_num_evaluations,
                 'optimization_metric': config.get('optimization_metric', 'val_loss'),
                 'n_workers': args.n_workers
    }

    # Initialize the dataset in the main thread in the case that shared memory is used
    start_time = time.time()
    train, _ = load_data(config, create=True)
    end_time = time.time()
    print(f"Initial dataset loading took: {end_time - start_time} sec")

    tuner = Tuner(
        trial_backend=LocalBackend(entry_point='./src/objective_synetune.py',
            pass_args_as_json=True,
            delete_checkpoints=True,
        ),
        tuner_name=args.experiment_tag,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=max_wallclock_time,
                                        max_num_evaluations=max_num_evaluations),
        n_workers=args.n_workers,  # how many trials are evaluated in parallel
        metadata=metadata,
        sleep_time=1.0,
        #results_update_interval=1.0,
    )
    tuner.run()

    # Unlink shared memory or other resources
    if callable(getattr(train.dataset, 'close', None)):
        train.dataset.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument('--experiment_tag', default='shared-mem-3', type=str, help='Experiment name.')
    #parser.add_argument('--experiment_definition', default='ptbxl_lstm', type=str, help='Experiment definition.')
    parser.add_argument('--experiment_definition', default='ptbxl_lstm_shared', type=str, help='Experiment definition.')
    parser.add_argument('--experiment_tag', default='cifar10-setup-1', type=str, help='Experiment name.')
    #parser.add_argument('--experiment_definition', default='cifar10_simple', type=str, help='Experiment definition.')
    parser.add_argument('--master_seed', default=42, type=int, help='Master seed.')
    parser.add_argument('--num_seeds', default=1, type=int, help='Number of seeds.')
    parser.add_argument('--n_workers', default=1, type=int, help='Number of workers.')
    parser.add_argument('--method', default='DyHPO', type=str, help='HPO algorithm.')

    args = parser.parse_args()

    seed = args.master_seed

    with open(Path('experiment_definitions') / args.experiment_definition /'config.yaml') as file:
        config = yaml.safe_load(file)

    for i in range(args.num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        run_hpo(config, args, seed)

        seed += 1