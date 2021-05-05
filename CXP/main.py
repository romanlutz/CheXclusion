import torch
from train import *

from LearningCurve import *
from predictions import *
import pandas as pd
from plot import *

import argparse
from azureml.core import Dataset, Run, Workspace, Experiment
from azureml.core.authentication import InteractiveLoginAuthentication


import tempfile
import time
import os
import zipfile

from sklearn.model_selection import GroupShuffleSplit


diseases = ['Lung Opacity', 'Atelectasis', 'Cardiomegaly',
            'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
            'Lung Lesion', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices']

Age = ['60-80', '40-60', '20-40', '80-', '0-20']
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

    path_image = "../.."
    seed = args.seed
    train_df_path = f'split_{seed}/train_{seed}.csv'
    test_df_path = f'split_{seed}/test_{seed}.csv'
    val_df_path = f'split_{seed}/valid_{seed}.csv'

    MODE = args.mode  # Select "train" or "test", "plot"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CPU/GPU selection: using {device}")

    # combine train and valid since we do our own splits
    df_cxp_t = pd.read_csv('../CXP/train.csv')
    df_cxp_v = pd.read_csv('../CXP/valid.csv')
    df_cxp = pd.concat([df_cxp_t, df_cxp_v], ignore_index=True)

    # extract patient id from path and make it a column
    paths = list(df_cxp['Path'])             # path that includes patient00001
    patient = [p.split('/')[2] for p in paths]     # patient00001
    df_cxp.insert(0, 'PATIENT', patient)     # add 'Patient' PATIENT with patient00001

    split_dataset(df_cxp, seed, run, train_df_path, test_df_path, val_df_path)
    print("completed preprocessing and splitting dataset")

    if MODE == "train":
        ModelType = "densenet"
        CriterionType = 'BCELoss'
        LR = 5e-5

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device, LR, seed)

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
        factor = [gender, Age]
        factor_str = ['Sex', 'Age']
        for i in range(len(factor)):
            # plot_frequency(gt, diseases, factor[i], factor_str[i])
            # plot_TPR_CXP(pred, diseases, factor[i], factor_str[i])
             plot_sort_14(pred, diseases, factor[i], factor_str[i])
            # distance_max_min(pred, diseases, factor[i], factor_str[i])
            #plot_14(pred, diseases, factor[i], factor_str[i])
    # if MODE == "mean":
    #     pred = pd.read_csv("./results/bipred.csv")
    #     factor = [gender, age_decile]
    #     factor_str = ['Sex', 'Age']
    #     for i in range(len(factor)):
    #         mean(pred, diseases, factor[i], factor_str[i])

    # if MODE == "plot_14":
    #     pred = pd.read_csv("./results/bipred.csv")
    #     factor = [Age]
    #     factor_str = ['Age']
    #     for i in range(len(factor)):
    #         plot_14(pred, diseases, factor[i], factor_str[i])
    #         plot_Median(pred, diseases, factor[i], factor_str[i])

    run.complete()

def split_dataset(df, seed, run, train_df_path, test_df_path, val_df_path):
    # Split dataset into train and test/validation sets, then the latter into test and validation,
    # but ensure that each patient only occurs in one of the three without any overlap.
    X_tr_idx, X_testval_idx = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state=seed).split(df, groups=df['PATIENT']))
    X_tr = df.iloc[X_tr_idx]
    X_testval = df.iloc[X_testval_idx]
    
    X_test_idx, X_val_idx = next(GroupShuffleSplit(test_size=.50, n_splits=2, random_state=seed).split(X_testval, groups=X_testval['PATIENT']))
    X_test = X_testval.iloc[X_test_idx]
    X_val = X_testval.iloc[X_val_idx]
    
    n = df.shape[0]
    print('Seed ', seed)
    print('Train ', X_tr.shape[0]/n)
    print('Test ', X_test.shape[0]/n)
    print('Valid ', X_val.shape[0]/n)
    print()
    
    X_tr.to_csv(train_df_path)
    X_test.to_csv(test_df_path)
    X_val.to_csv(val_df_path)
    run.upload_folder(name=f'split_{seed}', path=".")


if __name__ == "__main__":
    main()
