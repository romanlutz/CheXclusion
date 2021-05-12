import torch
from train import *
import argparse
import os
from azureml.core import Dataset, Run, Workspace, Experiment
from azureml.core.authentication import InteractiveLoginAuthentication

import time

from LearningCurve import *
from predictions import *
from nih import *

import pandas as pd

#---------------------- on q
path_image = "NIH/"

train_df_path ="train.csv"
test_df_path = "test.csv"
val_df_path = "valid.csv"


diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
# age_decile = ['0-20', '20-40', '40-60', '60-80', '80-']
age_decile = ['40-60', '60-80', '20-40', '80-', '0-20']
gender = ['M', 'F']

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    # to run locally create a .sh file with the following three env vars and run
    # source ./setenv.sh
    SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
    RESOURCE_GROUP_NAME = os.getenv("RESOURCE_GROUP_NAME")
    WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
    TENANT_ID = os.getenv("TENANT_ID")
    if SUBSCRIPTION_ID is None or RESOURCE_GROUP_NAME is None or WORKSPACE_NAME is None or TENANT_ID is None:
        # a remote run - context should be set
        run = Run.get_context()
        ws = run.experiment.workspace
    else:
        # a local run - obtain information from env vars and create run manually
        auth = InteractiveLoginAuthentication(tenant_id=TENANT_ID)
        ws = Workspace(subscription_id=SUBSCRIPTION_ID,
                       resource_group=RESOURCE_GROUP_NAME,
                       workspace_name=WORKSPACE_NAME,
                       auth=auth)
        experiment_name = f"{args.dataset}-{args.seed}-{int(time.time())}"
        experiment = Experiment(workspace=ws, name=experiment_name)
        run = experiment.start_logging()

    path_image = ".."
    path_cxp_image = f'{path_image}/CheXpert-v1.0'
    seed = args.seed
    path_split = f'{path_cxp_image}/split_{seed}'
    train_df_path = f'{path_split}/train_{seed}.csv'
    test_df_path = f'{path_split}/test_{seed}.csv'
    val_df_path = f'{path_split}/valid_{seed}.csv'

    if not os.path.exists(os.path.join(path_split)):
        os.makedirs(os.path.join(path_split))

    MODE = "train"  # Select "train" or "test", "Resume", "plot", "Threshold", "plot15"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CPU/GPU selection: using {device}")

    split_dataset(df_cxp, seed, run, train_df_path, test_df_path, val_df_path)
    print("completed preprocessing and splitting dataset")

    if MODE == "train":
        ModelType = "densenet"  # select 'densenet'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device,LR)

        PlotLearningCurve()


    if MODE =="test":
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)

        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, path_image, device)

    if MODE == "plot":
        gt = pd.read_csv("./results/True.csv")
        pred = pd.read_csv("./results/bipred.csv")
        factor = [gender, age_decile]
        factor_str = ['Patient Gender', 'Patient Age']
        for i in range(len(factor)):
            # plot_frequency(gt, diseases, factor[i], factor_str[i])
            # plot_TPR_NIH(pred, diseases, factor[i], factor_str[i])
            plot_sort_median(pred, diseases, factor[i], factor_str[i])
           # Actual_TPR(pred, diseases, factor[i], factor_str[i])

    #         plot_Median(pred, diseases, factor[i], factor_str[i])


if __name__ == "__main__":
    main()
