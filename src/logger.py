from pathlib import Path
import wandb
from datetime import datetime
import time
import yaml


class Logger:
    def __init__(self, config, wandb=False, dir=None, budget=None, start_time=None, max_time=None):
        self.exp_id = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config
        self.logdir = dir

        self.wandb_enabled = wandb

        # Local logging setup
        if dir is not None:
            self.local_enabled = True
            if not dir.is_dir():
                dir.mkdir(parents=True)

            self.logfile = open(dir / (self.exp_id+".csv"), 'w')
        else:
            self.local_enabled = False

        # Monitor budget spending
        self.trial=0
        self.budget = budget

        # Monitor time budget
        if start_time is not None:
            self.start_time = start_time
            self.max_time = max_time
        else:
            self.start_time = None

    def _init_csv(self, results):
        self.logfile.write("trial,epoch,configuration_id,time,")
        for key, _ in results.items():
            self.logfile.write(f"{key},")
        self.logfile.write("configurations\n")

    def _time_exceeded(self):
        if self.start_time is not None:
            now = time.time()
            if now - self.start_time > self.max_time:
                return True
        else:
             return False

    def init_run(self, model, id, params):
        if self.trial + 1 > self.budget or self._time_exceeded():
            raise BudgetExceededException()

        self.curr_id = id
        self.curr_params = params

        if self.wandb_enabled:
            wandb.init(project=self.config['project'], group=self.config['group'], id=id , config=params, reinit=True, resume='auto')
            wandb.watch(model)

    def log(self, results, epoch, time):
        self.trial += 1

        # Stop optimization if budget is exceeded and do not log the results
        if self.trial > self.budget:
            raise BudgetExceededException()

        if self.trial == 1:
            self._init_csv(results)

        if self.wandb_enabled:
            wandb.log(results, step=epoch)

        if self.local_enabled:
            self.logfile.write(f'{self.trial},{epoch},{self.curr_id},{time},')
            for _, value in results.items():
                self.logfile.write(f'{value},')
            self.logfile.write(f'"{self.curr_params}"\n')
            self.logfile.flush()

        # Stop if time budget is exceeded
        if self._time_exceeded():
            raise BudgetExceededException()

    def finish(self):
        if self.wandb_enabled:
            wandb.finish()

    def get_logdir(self):
        return self.logdir

    def save_config(self, params, path):
        with open(path, 'w') as f:
            yaml.dump(params, f)

class BudgetExceededException(Exception):
    pass