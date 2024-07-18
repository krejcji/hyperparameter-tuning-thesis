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
from syne_tune.experiments.default_baselines import (
    ASHA,
    MOBSTER,
    HyperTune,
    SyncHyperband,
    SyncBOHB,
    DyHPO,
    DEHB,
    RandomSearch
)


class Methods:
    RS = "RS"
    ASHA = "ASHA"
    MOBSTER_JOINT = "MOBSTER-JOINT"
    MOBSTER_INDEP = "MOBSTER-INDEP"
    HYPERTUNE_INDEP = "HYPERTUNE-INDEP"
    HYPERTUNE_JOINT = "HYPERTUNE-JOINT"
    SYNCHB = "SYNCHB"
    BOHB = "BOHB"
    DyHPO = "DyHPO"
    DyHPO_lim = "DyHPO-lim"
    DEHB = "DEHB"


methods = {
    Methods.RS : lambda method_arguments: RandomSearch(method_arguments),
    Methods.DEHB: lambda method_arguments: DEHB(method_arguments),
    Methods.ASHA: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
    ),
    Methods.MOBSTER_JOINT: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
    ),
    Methods.MOBSTER_INDEP: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
        search_options=dict(model="gp_independent"),
    ),
    Methods.HYPERTUNE_INDEP: lambda method_arguments: HyperTune(
        method_arguments,
        type="promotion",
        search_options=dict(model="gp_independent"),
    ),
    Methods.HYPERTUNE_JOINT: lambda method_arguments: HyperTune(
        method_arguments,
        type="promotion",
        search_options=dict(model="gp_multitask"),
    ),
    Methods.SYNCHB: lambda method_arguments: SyncHyperband(method_arguments),
    Methods.BOHB: lambda method_arguments: SyncBOHB(method_arguments),
    Methods.DyHPO: lambda method_arguments: DyHPO(method_arguments),
    Methods.DyHPO_lim: lambda method_arguments: DyHPO(
        method_arguments,
        search_options=dict(max_size_data_for_model=120, # The default is 500
                            opt_skip_init_length=40,
                            opt_skip_period=6)), # The default is 3
}