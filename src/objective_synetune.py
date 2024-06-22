import logging
import time

from syne_tune import Reporter
from argparse import ArgumentParser
from syne_tune.utils import add_config_json_to_argparse, load_config_json
from syne_tune.utils import (
    resume_from_checkpointed_model,
    checkpoint_model_at_rung_level,
    add_checkpointing_to_argparse,
    pytorch_load_save_functions,
)
from syne_tune.constants import ST_CHECKPOINT_DIR

from load_data import load_data
from training.train_net_st import train_net
from training.load_pytorch_model import model_and_optimizer

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = ArgumentParser()
    add_config_json_to_argparse(parser)
    add_checkpointing_to_argparse(parser)
    args, _ = parser.parse_known_args()
    config = load_config_json(vars(args))

    report = Reporter()

    # Load the dataloaders
    start_t = time.time()
    trainloader, valloader = load_data(config)
    end_t = time.time()
    print(f"Objective dataset loading took: {end_t - start_t}s")

    # Load the model, optimizer, and scheduler and their states
    start_t = time.time()
    state = model_and_optimizer(config)

    load_model_fn, save_model_fn = pytorch_load_save_functions(
        {"model": state["model"], "optimizer": state["optimizer"],
         "scheduler": state["scheduler"]}
    )
    resume_from = resume_from_checkpointed_model(config, load_model_fn)
    end_t = time.time()
    print(f"Model loading took: {end_t - start_t}s")

    # Iterate over the epochs
    for step in range(resume_from+1, config['epochs']+1):
        start_t = time.time()
        result = train_net(trainloader,
            valloader,
            state,
            config,
            epoch=step)
        end_t = time.time()
        checkpoint_model_at_rung_level(config, save_model_fn, step)
        print(f"Epoch {step} took: {end_t - start_t}s, val_loss: {result['val_loss']}")

        # Feed the score back to Syne Tune.
        report(epoch=step, **result)

    # Clean up resources (e.g. shared memory)
    if callable(getattr(trainloader.dataset, 'close', None)):
        trainloader.dataset.close()
        valloader.dataset.close()

