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
from train_net_st import train_net
#from logger import Logger
#from logger import BudgetExceededException

class Logger:
  def __getattr__(self, attr):
    def dummy(*args, **kwargs):
        return None
    return dummy

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = ArgumentParser()
    #parser.add_argument(f"--{ST_CHECKPOINT_DIR}", type=str)
    add_config_json_to_argparse(parser)
    add_checkpointing_to_argparse(parser)
    args, _ = parser.parse_known_args()
    config = load_config_json(vars(args))

    report = Reporter()

    trainloader, valloader = load_data(config)

    if ST_CHECKPOINT_DIR not in args:
        raise ValueError(f"Missing {ST_CHECKPOINT_DIR} argument.")
    else:
        checkpoint_path = getattr(args, ST_CHECKPOINT_DIR)
    logger = Logger()

    resume_from = resume_from_checkpointed_model(config, load_model_fn)

    for step in range(config['epochs']):
        loss = train_net(trainloader,
            valloader,
            config,
            previous_epoch=step,
            end_epoch=step + 1,
            checkpoint_path=checkpoint_path,
            logger=logger)
        # Feed the score back to Syne Tune.
        print(f"Epoch {step + 1} loss: {loss[0]['loss']}")
        report(epoch=step + 1, val_loss=loss[0]['loss'])

