# Efficient hyperparameter tuning

The goal of this thesis is to research hyperparameter tuning with limited resources. We study hyperparameters in the deep learning scenario, with the focus on medical datasets.

## Experiments

Experiments will be performed on several datasets. So far, two datasets are considered:

* PPG -> blood pressure - from Kaggle

* ECG -> diagnosis - PTB-XL dataset

### config.yaml Documentation

Configuration file config.yaml contains all the necessary information for running an experiment.

* `experiment` - experiment description for logging

* `metrics` - 'accuracy', 'mse', 'macro_auc'

**hp_optimizer**

* `name` - 'Optuna', 'SMAC'

* `n_trials` - number of trials for the optimizer to perform

* `pruning` - some optimizers support early pruning of trials

**Data**

* `name`: 'PTB-XL', 'Kaggle_PPG'

* `batch_size` 

**Model**

Two models are supported. CNN is a simple model with a few of CNN layers before Dense hidden layer and classification layer. It's used for experimental purposes, because it's more customizable than the xresnet1d model, which performs much better.

* `name`: 'xresnet1d', 'CNN'

**tunable_params**

Specify parameters that should be optimized.

**fixed_params**