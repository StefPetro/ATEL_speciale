import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.model_selection import KFold
from atel.data import BookCollection
from data_clean import *
import glob
import yaml
from yaml import CLoader
from typing import Tuple
sns.set_style('whitegrid')

NUM_SPLITS = 10
SEED = 42
set_seed(SEED)

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

with open('target_info.yaml', 'r', encoding='utf-8') as f:
    target_info = yaml.load(f, Loader=CLoader)

with open('translation.yaml', 'r', encoding='utf-8') as f:
    translation = yaml.load(f, Loader=CLoader)

# .replace(" ", "_").replace("å", "aa")
logs_path_dict = {
    'BERT_MLM':        lambda t: f'./huggingface_logs'\
                                +f'/BERT_mlm_gyldendal'\
                                +f'/{t}'\
                                +f'/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05',
    'LSTM':            lambda t: f'./lightning_logs'\
                                +f'/{t}'\
                                +f'/num_epoch_20000-embedding_size_100-lstm_layers_4-lstm_size_256-l1_size_256-l2_size_128'
                                +f'/bs_64',
    'BabyBERTA':       lambda t: f'./babyberta_logs'\
                                +f'/models'\
                                +f'/BabyBERTa_131022_GW'\
                                +f'/{t}'
                                +f'/BERT-BS16-BA4-ep500-seed42-WD0.01-LR2e-05',
    'GaussianNB':      lambda t: f'./baseline_logs'\
                                +f'/GaussianNB'\
                                +f'/{t}',
    'RandomForest':    lambda t: f'./baseline_logs'\
                                +f'/RandomForest'\
                                +f'/{t}',
    'RidgeClassifier': lambda t: f'./baseline_logs'\
                                +f'/RidgeClassifier'\
                                +f'/{t}',
}

print(logs_path_dict['BERT_MLM']('test'))

def get_logit_func(problem_type: str):
    if problem_type == 'multilabel':
        logit_func = torch.nn.Sigmoid()
        return logit_func
    elif problem_type == 'multiclass':
        logit_func = torch.nn.Softmax(dim=1)
        return logit_func
    else:
        raise Exception(f'Problem type as to be "multiclass" or "multilabel" - not "{problem_type}"')

TARGET = list(target_info.keys())[0]
problem_type = target_info[TARGET]['problem_type']

def get_labels_texts(book_col: BookCollection, TARGET: str):

    df, label_names = get_pandas_dataframe(book_col, TARGET)
    df = df.reset_index()
    
    kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
    all_splits = [k for k in kf.split(df)]

    # Get the order of the data, from the validation splits
    idx = np.array([])
    for _, val_idx in all_splits:
        idx = np.hstack([idx, val_idx])

    df = df.loc[idx, :]
    labels = np.array([a for a in df.labels.values])
    texts  = np.array([t for t in df.text.values])
    return labels, texts, label_names, idx
    

def get_model_preds(model: str, TARGET: str, problem_type: str) -> np.ndarray:
    baseline_models = ['GaussianNB', 'RandomForest', 'RidgeClassifier']
    model_preds = np.array([])
    for i in range(10):
        filepath = logs_path_dict[model](TARGET.replace(" ", "_").replace("å", "aa")) + f'/CV_{i+1}'
        
        preds_dir = glob.glob(filepath + '/*.pt')
        assert len(preds_dir) == 1, "There can only be a single prediction file"

        logits = torch.load(preds_dir[0], map_location=torch.device('cpu'))
                
        if type(logits) == np.ndarray:
            logits = torch.from_numpy(logits)
        
        if model not in baseline_models:  # baseline models already provide predictions
            # rounds down 0.5, which is the behaviour in torchmetrics too (as default)
            preds = torch.round(logit_func(logits.detach()))
            if problem_type == 'multiclass':
                preds = torch.argmax(preds, dim=1)
        else:
            preds = logits
        
        if problem_type == 'multiclass':
            model_preds = np.hstack([model_preds, preds.numpy()])
        else:
            if model_preds.shape[0] == 0:
                model_preds = preds.numpy()
            else:
                model_preds = np.vstack([model_preds, preds.numpy()])
        
    return model_preds


def get_right_wrong_preds(labels: np.ndarray, model_preds: np.ndarray) -> np.ndarray:
    # compare labels with preds
    if len(labels.shape) == 1:
        mask = (labels == model_preds)
    else:
        mask = (labels == model_preds).all(axis=1)
    # Finds which data points the preds get right (True) and wrong (False)
    # right_preds = idx[mask]
    # wrong_preds = idx[~mask]
    return mask


for TARGET in target_info.keys():
    print(f'Getting results for {TARGET}')
    problem_type = target_info[TARGET]['problem_type']
    logit_func = get_logit_func(problem_type)
    
    labels, texts, label_names, idx = get_labels_texts(book_col, TARGET)
    df = pd.DataFrame(idx, columns=['index'])
    
    for model in logs_path_dict.keys():
        print(f'{model}')
        model_preds = get_model_preds(model, TARGET, problem_type)
        mask = get_right_wrong_preds(labels, model_preds)
        df.loc[:, model] = mask.astype(int)
    
    df.to_csv(f'prediction_analysis/{TARGET.replace(" ", "_").replace("å", "aa")}/overview.csv', index=False)
    
    if problem_type == 'multilabel':
        for i, l in enumerate(label_names):
            print(f'Label: {l}')
            df = pd.DataFrame(idx, columns=['index'])
            for model in logs_path_dict.keys():
                print(f'{model}')
                model_preds = get_model_preds(model, TARGET, problem_type)
                mask = get_right_wrong_preds(labels[:, i], model_preds[:, i])
                df.loc[:, model] = mask.astype(int)
            
            df.to_csv(f'prediction_analysis/{TARGET.replace(" ", "_").replace("å", "aa")}/label_{l}.csv', index=False)
    
    

