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
from sklearn.model_selection import GroupShuffleSplit

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
    path_nih_image = f'{path_image}/NIH'
    seed = args.seed
    path_split = f'{path_nih_image}/split_{seed}'
    train_df_path = f'{path_split}/train_{seed}.csv'
    test_df_path = f'{path_split}/test_{seed}.csv'
    val_df_path = f'{path_split}/valid_{seed}.csv'

    df_nih = pd.read_csv(f'{path_nih_image}/Data_Entry_2017_v2020.csv')

    if not os.path.exists(os.path.join(path_split)):
        os.makedirs(os.path.join(path_split))

    MODE = "train"  # Select "train" or "test", "Resume", "plot", "Threshold", "plot15"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CPU/GPU selection: using {device}")

    split_dataset(df_nih, seed, run, train_df_path, test_df_path, val_df_path)
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

def split_dataset(df, seed, run, train_df_path, test_df_path, val_df_path):
    df = preprocess_NIH(df)
    print(df.head())
    # Split dataset into train and test/validation sets, then the latter into test and validation,
    # but ensure that each patient only occurs in one of the three without any overlap.
    X_tr_idx, X_testval_idx = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state=seed).split(df, groups=df['subject_id']))
    X_tr = df.iloc[X_tr_idx]
    X_testval = df.iloc[X_testval_idx]
    
    X_test_idx, X_val_idx = next(GroupShuffleSplit(test_size=.50, n_splits=2, random_state=seed).split(X_testval, groups=X_testval['subject_id']))
    X_test = X_testval.iloc[X_test_idx]
    X_val = X_testval.iloc[X_val_idx]
    
    n = df.shape[0]
    print('Seed ', seed)
    print('Train ', X_tr.shape[0]/n)
    print('Test ', X_test.shape[0]/n)
    print('Valid ', X_val.shape[0]/n)
    print()
    
    X_tr.to_csv(train_df_path, mode='w')
    X_test.to_csv(test_df_path, mode='w')
    X_val.to_csv(val_df_path, mode='w')
    run.upload_folder(name=f'split_{seed}', path=".")

def preprocess_NIH(df):
    df['Patient Age'] = np.where(df['Patient Age'].between(0,19), 19, df['Patient Age'])
    df['Patient Age'] = np.where(df['Patient Age'].between(20,39), 39, df['Patient Age'])
    df['Patient Age'] = np.where(df['Patient Age'].between(40,59), 59, df['Patient Age'])
    df['Patient Age'] = np.where(df['Patient Age'].between(60,79), 79, df['Patient Age'])
    df['Patient Age'] = np.where(df['Patient Age']>=80, 81, df['Patient Age'])
    
    copy_subjectid = df['Patient ID'] 
    df.drop(columns = ['Patient ID'])
    
    df = df.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81], 
                     [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
   
    df['subject_id'] = copy_subjectid
    df.rename(columns={"Image Index": "path", "Patient Gender": "Sex", "Patient Age": "Age"}, inplace=True)
    print(df.head())

    return df

if __name__ == "__main__":
    main()
