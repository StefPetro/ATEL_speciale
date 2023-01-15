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


def get_logit_func(problem_type: str):
    if problem_type == 'multilabel':
        logit_func = torch.nn.Sigmoid()
        return logit_func
    elif problem_type == 'multiclass':
        logit_func = torch.nn.Softmax(dim=1)
        return logit_func
    else:
        raise Exception(f'Problem type as to be "multiclass" or "multilabel" - not "{problem_type}"')


def get_labels_texts(book_col: BookCollection, TARGET: str):

    df, label_names = get_pandas_dataframe(book_col, TARGET)
    df = df.reset_index()
    
    kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
    all_splits = [k for k in kf.split(df)]

    # Get the order of the data, from the validation splits
    idx = np.array([])
    cvs = np.array([])
    for i, (_, val_idx) in enumerate(all_splits):
        idx = np.hstack([idx, val_idx])
        cvs = np.hstack([cvs, np.full(len(val_idx), i+1)])

    df = df.loc[idx, :]
    labels = np.array([a for a in df.labels.values])
    texts  = np.array([t for t in df.text.values])
    return labels, texts, label_names, idx, cvs
    

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
    return mask


def overview_csv(models: list, TARGET: str, problem_type: str, labels: np.ndarray):
    df = pd.DataFrame(idx, columns=['index'])
    for model in models:
        model_preds = get_model_preds(model, TARGET, problem_type)
        mask = get_right_wrong_preds(labels, model_preds)
        df.loc[:, model] = mask.astype(int)
    df.to_csv(f'prediction_analysis/{TARGET.replace("å", "aa")}/overview.csv', index=False)


def label_csv(models: list, TARGET: str, problem_type: str, labels: np.ndarray, label_names: list):
    assert problem_type == 'multilabel', 'Problem type needs to be multilabel'
    
    for i, l in enumerate(label_names):
        df = pd.DataFrame(idx, columns=['index'])
        for model in models:
            model_preds = get_model_preds(model, TARGET, problem_type)
            mask = get_right_wrong_preds(labels[:, i], model_preds[:, i])
            df.loc[:, model] = mask.astype(int)
        
        df.to_csv(f'prediction_analysis/{TARGET.replace("å", "aa")}/label_{l}.csv', index=False)


def ensemble_preds_csv(ensemble_models: list, idx: np.ndarray, cvs: np.ndarray, TARGET: str, problem_type: str, label_names: list):
    if problem_type == 'multilabel':
        multilabel_ensemble(ensemble_models, idx, cvs, TARGET, problem_type, label_names)
    elif problem_type == 'multiclass':
        multiclass_ensemble(ensemble_models, idx, cvs, TARGET, problem_type)
    else:
        raise Exception(f'Problem type not "multilabel" or "multiclass" but {problem_type}')


def multilabel_ensemble(ensemble_models: list, idx: np.ndarray, cvs: np.ndarray, TARGET: str, problem_type: str, label_names: list):
    assert problem_type == 'multilabel', 'Problem type needs to be multilabel'
    
    df = pd.DataFrame(data={'index': idx, 'cvs': cvs})
    ensemble_preds = np.array([])
    for model in ensemble_models:
        model_preds = get_model_preds(model, TARGET, problem_type)
        if ensemble_preds.shape[0] == 0:
            ensemble_preds = model_preds[np.newaxis, :, :]
        else:
            ensemble_preds = np.vstack([ensemble_preds, model_preds[np.newaxis, :, :]])
            
    ensemble_preds = (ensemble_preds.sum(axis=0) >= 2).astype(int)
    df[label_names] = ensemble_preds
    df.to_csv(f'prediction_analysis/{TARGET.replace("å", "aa")}/ensemble_preds.csv', index=False)


def multiclass_ensemble(ensemble_models: list, idx: np.ndarray, cvs: np.ndarray, TARGET: str, problem_type: str):
    assert problem_type == 'multiclass', 'Problem type needs to be multiclass'
    
    df = pd.DataFrame(data={'index': idx, 'cvs': cvs})
    ensemble_preds = np.array([])
    for model in ensemble_models:
        model_preds = get_model_preds(model, TARGET, problem_type)
        if ensemble_preds.shape[0] == 0:
            ensemble_preds = model_preds
        else:
            ensemble_preds = np.vstack([ensemble_preds, model_preds])
    
    bincounts = np.array([np.bincount(row, minlength=target_info[TARGET]['num_labels']) for row in ensemble_preds.T.astype(int)])
    preds = np.array([np.argmax(row) if row[row > 1].any() else np.random.choice(row.shape[0], 1, p=row.astype(float)/np.sum(row))[0] for row in bincounts])
    
    df['predictions'] = preds
    df.to_csv(f'prediction_analysis/{TARGET.replace("å", "aa")}/ensemble_preds.csv', index=False)
        

for TARGET in target_info.keys():
    print(f'Getting results for {TARGET}')
    problem_type = target_info[TARGET]['problem_type']
    logit_func = get_logit_func(problem_type)
    
    labels, texts, label_names, idx, cvs = get_labels_texts(book_col, TARGET)
    
    overview_csv(logs_path_dict.keys(), TARGET, problem_type, labels)
    
    ensemble_models = ['BERT_MLM', 'BabyBERTA', 'RandomForest']
    ensemble_preds_csv(ensemble_models, idx, cvs, TARGET, problem_type, label_names)
    
    if problem_type == 'multilabel':
        label_csv(logs_path_dict.keys(), TARGET, problem_type, labels, label_names)
        
    

TARGET = 'Semantisk univers'
l = 'Dyr og natur'
def find_example(book_col: BookCollection, model: str, target: str, label: str):
    labels, texts, label_names, idx, cvs = get_labels_texts(book_col, target)
    
    df = pd.read_csv(f'./prediction_analysis/{target}/label_{label}.csv')
    
    right_preds = df[model].values
    labels = labels[:, label_names.index(label)]
    
    # find true positive
    mask = np.logical_and(labels == 0, right_preds == 0)
    print(sum(mask))
    print(texts[mask])
    print(idx[mask])
    print(cvs[mask])

# find_example(book_col, 'BERT_MLM', TARGET, l)


