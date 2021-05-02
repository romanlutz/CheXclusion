# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Utilities to use an Azure Machine Learning workspace to track the performance test results
over time.
"""

import logging
import os
import argparse
import time

from azureml.core import Workspace, Experiment, RunConfiguration, ScriptRunConfig
from azureml.core.environment import Environment
from azureml.core.compute import ComputeTarget
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml._base_sdk_common.common import resource_client_factory
from azure.mgmt.resource.resources.models import ResourceGroup


# Users may define their own variable names where they store the following data.
# We use VARIABLE_NAME_* to identify where they stored it, and subsequently look it up.
# The reason for these levels of indirection are mostly key restrictions:
# Azure Key Vault doesn't allow underscores, Powershell doesn't allow dashes.
# Furthermore, user-specific secrets in Key Vault should have use case specific names such as
# <use-case>-sp-id as opposed to SERVICE_PRINCIPAL_ID.
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP_NAME = os.getenv("RESOURCE_GROUP_NAME")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
WORKSPACE_LOCATION = os.getenv("WORKSPACE_LOCATION")
COMPUTE_NAME = os.getenv("COMPUTE_NAME")

logger = logging.getLogger(__file__)


def get_workspace(compute_name, compute_config=None):
    if SUBSCRIPTION_ID is None or \
            RESOURCE_GROUP_NAME is None or \
            WORKSPACE_NAME is None or \
            WORKSPACE_LOCATION is None or \
            COMPUTE_NAME is None:
        print("One of the required environment variables is not set. "
              "Running locally instead.")
        return None

    logger.info("Logging in.")
    auth = InteractiveLoginAuthentication()
    logger.info("Successfully logged in.")

    logger.info("Ensuring resource group {} exists.".format(RESOURCE_GROUP_NAME))
    resource_management_client = resource_client_factory(auth, SUBSCRIPTION_ID)
    resource_group_properties = ResourceGroup(location=WORKSPACE_LOCATION)
    resource_management_client.resource_groups.create_or_update(WORKSPACE_NAME,
                                                                resource_group_properties)
    logger.info("Ensured resource group {} exists.".format(RESOURCE_GROUP_NAME))

    logger.info("Ensuring workspace {} exists.".format(WORKSPACE_NAME))
    workspace = Workspace.create(name=WORKSPACE_NAME, auth=auth, subscription_id=SUBSCRIPTION_ID,
                                 resource_group=RESOURCE_GROUP_NAME, location=WORKSPACE_LOCATION,
                                 create_resource_group=False, exist_ok=True)
    logger.info("Ensured workspace {} exists.".format(WORKSPACE_NAME))
    logger.info("Ensuring compute exists.")
    get_or_create_compute_cluster(workspace, compute_name, compute_config)
    logger.info("Ensured compute exists.")
    return workspace


def get_or_create_compute_cluster(workspace, compute_name, config=None):
    if compute_name not in workspace.compute_targets:
        if config is None:
            config = ComputeTarget.provisioning_configuration(
                vm_size='STANDARD_NC24',
                vm_priority="dedicated",
                min_nodes=0,
                max_nodes=4,
                idle_seconds_before_scaledown=300
            )
        compute_target = ComputeTarget.create(workspace, compute_name, config)
        compute_target.wait_for_completion(show_output=True)
    else:
        compute_target = workspace.compute_targets[compute_name]
    return compute_target


def run(dataset, seed, mode):
    workspace = get_workspace(compute_name=COMPUTE_NAME)
    experiment_name = f"{dataset}-{seed}-{int(time.time())}"
    print(f"Start experiment {experiment_name}")
    experiment = Experiment(workspace=workspace, name=experiment_name)
    compute_target = workspace.compute_targets[COMPUTE_NAME]
    run_config = RunConfiguration()
    run_config.target = compute_target

    environment = Environment.from_conda_specification("f1", os.path.join("f1.yml"))
    run_config.environment = environment
    environment.register(workspace=workspace)
    print("environment successfully configured")
    project_directory = os.path.join(dataset)
    script_name = "main.py"
    script_run_config = ScriptRunConfig(source_directory=project_directory,
                                        script=script_name,
                                        run_config=run_config,
                                        arguments=[
                                            '--dataset', dataset,
                                            '--seed', seed,
                                            '--mode', mode],
)
    print("submitting run")
    experiment.submit(config=script_run_config)
    print("submitted run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed")
    parser.add_argument("--dataset")
    parser.add_argument("--mode", default="train")

    args = parser.parse_args()

    if args.seed is None or args.dataset is None:
        raise Exception("Arguments seed and dataset need to be specified.")

    run(args.dataset, args.seed, args.mode)
