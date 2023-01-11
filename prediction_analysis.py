import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.model_selection import KFold
from atel.data import BookCollection
from data_clean import *
import yaml
from yaml import CLoader
sns.set_style('whitegrid')

NUM_SPLITS = 10
SEED = 42
set_seed(SEED)

logs_path_dict = {
    'BERT_MLM':      '',
    'LSTM':          '',
    'BabyBERTA':     '',
    'Ridge':         '',
    'NaiveBayes':    '',
    'Random_forest': ''
}

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

with open('target_info.yaml', 'r', encoding='utf-8') as f:
    target_info = yaml.load(f, Loader=CLoader)

with open('translation.yaml', 'r', encoding='utf-8') as f:
    translation = yaml.load(f, Loader=CLoader)

TARGET = list(target_info.keys())[0]
problem_type = target_info[TARGET]['problem_type']

if problem_type == 'multilabel':
    logit_func = torch.nn.Sigmoid()
else:
    logit_func = torch.nn.Softmax()

df, label_names = get_pandas_dataframe(book_col, TARGET)
df = df.reset_index()
labels = np.array([a for a in df.labels.values])

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
all_splits = [k for k in kf.split(df)]


# Get the order of the data, from the validation splits
idx = np.array([])
for train_idx, val_idx in all_splits:
    idx = np.hstack([idx, val_idx])

df = df.loc[idx, :]

labels = np.array([a for a in df.labels.values])
texts  = np.array([t for t in df.text.values])

model_preds = np.array([])
for i in range(10):
    filepath = f'./huggingface_logs'\
                +f'/BERT_mlm_gyldendal'\
                +f'/{TARGET.replace(" ", "_").replace("å", "aa")}'\
                +f'/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05'\
                +f'/CV_{i+1}'\
                +f'/{TARGET.replace("å", "aa")}_CV{i+1}_logits.pt'
            
    
    logits = torch.from_numpy(torch.load(filepath))
    preds  = torch.round(logit_func(logits))  # rounds down 0.5, which is the behaviour in torchmetrics too (as default)
    
    if model_preds.shape[0] == 0:
        model_preds = preds.numpy()
    else:
        model_preds = np.vstack([model_preds, preds.numpy()])
    

# compare labels with preds
comparision = (labels == model_preds)

print(idx[((~comparision[:, 0]) == (labels[:, 0] == 1))])
print()
        
# Finds which data points the preds get right (True) and wrong (False)
check = comparision.all(axis=1)

wrong_preds = idx[~check]

