# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# The code is modified so that the max_wallclock_time is set to be enough
# for >20 full evaluations
from collections import defaultdict

from syne_tune.experiments.benchmark_definitions.common import (
    SurrogateBenchmarkDefinition,
)

# NAS-Bench-201 definitions
NAS201_MAX_WALLCLOCK_TIME = {
    "cifar10": 5 * 5 * 3600,
    "cifar100": 4 * 6 * 3600,
    "ImageNet16-120": 9 * 8 * 3600,
}


NAS201_N_WORKERS = {
    "cifar10": 4,
    "cifar100": 4,
    "ImageNet16-120": 8,
}


def nas201_benchmark(dataset_name):
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=360000,
        n_workers=NAS201_N_WORKERS[dataset_name],
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
        max_num_evaluations=4000,
    )


nas201_benchmark_definitions = {
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
}

# FCNet definitions
FCNET_MAX_WALLCLOCK_TIME = defaultdict(lambda: 3600)
FCNET_MAX_WALLCLOCK_TIME.update({
        "protein_structure": 2 * 3600,
        "slice_localization": 3 * 3600,
        "naval_propulsion": 3600,
        "parkinsons_telemonitoring": 1800,
    })

def fcnet_benchmark(dataset_name):
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=360000,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_loss",
        mode="min",
        blackbox_name="fcnet",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
        max_num_evaluations=2000,
    )


fcnet_benchmark_definitions = {
    "fcnet-protein": fcnet_benchmark("protein_structure"),
    "fcnet-naval": fcnet_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_benchmark("slice_localization"),
}

# LCBench selected definitions
LCBENCH_MAX_WALLCLOCK_TIME = defaultdict(lambda: 7200)
LCBENCH_MAX_WALLCLOCK_TIME.update({
        "Fashion-MNIST": 5 * 7200,
        "airlines": 4 * 7200,
        "albert": 4 * 7200,
        "covertype": 3 * 7200,
        "christine": 10 * 7200,
    })

def lcbench_benchmark(dataset_name: str, datasets=None) -> SurrogateBenchmarkDefinition:
    """
    The default is to use nearest neighbour regression with ``K=1``. If
    you use a more sophisticated surrogate, it is recommended to also
    define ``add_surrogate_kwargs``, for example:

    .. code-block:: python

       surrogate="RandomForestRegressor",
       add_surrogate_kwargs={
           "predict_curves": True,
           "fit_differences": ["time"],
       },

    :param dataset_name: Value for ``dataset_name``
    :param datasets: Used for transfer learning
    :return: Definition of benchmark
    """
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=360000,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",  # 1-nn surrogate
        surrogate_kwargs={"n_neighbors": 1},
        max_num_evaluations=1000,
        datasets=datasets,
        max_resource_attr="epochs",
    )


lcbench_datasets = [
    "KDDCup09_appetency",
    "covertype",
    "Amazon_employee_access",
    "adult",
    "nomao",
    "bank-marketing",
    "shuttle",
    "Australian",
    "kr-vs-kp",
    "mfeat-factors",
    "credit-g",
    "vehicle",
    "kc1",
    "blood-transfusion-service-center",
    "cnae-9",
    "phoneme",
    "higgs",
    "connect-4",
    "helena",
    "jannis",
    "volkert",
    "MiniBooNE",
    "APSFailure",
    "christine",
    "fabert",
    "airlines",
    "jasmine",
    "sylvine",
    "albert",
    "dionis",
    "car",
    "segment",
    "Fashion-MNIST",
    "jungle_chess_2pcs_raw_endgame_complete",
]

lcbench_benchmark_definitions = {
    "lcbench-" + task: lcbench_benchmark(task) for task in lcbench_datasets
}


# 5 most expensive lcbench datasets
lcbench_selected_datasets = [
    "Fashion-MNIST",
    "airlines",
    "albert",
    "covertype",
    "christine",
]

# Extended lcbench definitions
lcbench_extended_datasets = [
   "higgs",
   "jannis",
   "volkert",
   "dionis",
   "helena",
]

lcbench_selected_benchmark_definitions = {
    "lcbench-"
    + task.replace("_", "-").replace(".", ""): lcbench_benchmark(
        task, datasets=lcbench_selected_datasets
    )
    for task in lcbench_selected_datasets
}

lcbench_extended_benchmark_definitions = {
    "lcbench-"
    + task.replace("_", "-").replace(".", ""): lcbench_benchmark(
        task, datasets=lcbench_extended_datasets
    )
    for task in lcbench_extended_datasets
}

benchmark_definitions = {
    **nas201_benchmark_definitions,
    **fcnet_benchmark_definitions,
    **lcbench_benchmark_definitions,
}
