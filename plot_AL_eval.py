import json
import torch
import numpy as np
from sklearn.model_selection import KFold
from transformers import AutoTokenizer
from torchmetrics.functional.classification import (
    multilabel_exact_match,
    multilabel_accuracy, multiclass_accuracy,
    multilabel_f1_score, multiclass_f1_score,
    multilabel_recall, multiclass_recall,
    multilabel_precision, multiclass_precision,
    multilabel_auroc, multiclass_auroc
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from atel.data import BookCollection
from data_clean import *
sns.set_style('whitegrid')

NUM_SPLITS = 10
SEED = 42
TARGET = 'Semantisk univers'
NUM_LABELS = 5
problem_type = 'multilabel'
set_seed(SEED)

metric_dict = {
    'accuracy_exact':  'Val. Subset Acc.',
    'accuracy_micro':  'Val. Acc. Micro',
    'accuracy_macro':  'Val. Acc. Macro',
    'precision_macro': 'Val. Precision Macro',
    'recall_macro':    'Val. Recall Macro',
    'f1_macro':        'Val. F1 Macro',
    'AUROC_macro':     'Val. AUROC Macro',
}

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

tokenizer = AutoTokenizer.from_pretrained("../../../../../work3/s173991/huggingface_models/BERT_mlm_gyldendal")
# tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

df, labels = get_pandas_dataframe(book_col, TARGET)

# using .reset_index(), to get the index of each row
dataset = Dataset.from_pandas(df.reset_index())
token_dataset = dataset.map(tokenize_function, batched=True)

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
all_splits = [k for k in kf.split(token_dataset)]

all_logits = {}
for func in ['entropy', 'random']:
    all_logits[func] = {}
    for i in range(0, 10):
        filepath = f'./huggingface_logs'\
                +f'/active_learning'\
                +f'/BERT_mlm_gyldendal'\
                +f'/{func}'\
                +f'/{TARGET.replace(" ", "_")}'\
                +f'/BS16-BA4-MS3300-seed42-WD0.01-LR2e-05'\
                +f'/CV_{i+1}'\
                +f'/test_logits.json'
    
        with open(filepath, 'r') as loadfile:
            data = json.load(loadfile)

        all_logits[func][i] = {}
        
        all_logits[func][i]['num_samples'] = num_samples = data['num_train_samples']
        all_logits[func][i]['logits'] = torch.tensor(data['logits'])


def compute_metrics(logits, labels, problem_type: str='multilabel'):
    labels = labels.int()
    
    if problem_type == 'multilabel':
        logit_func = torch.nn.Sigmoid()
        preds = logit_func(logits)
        acc_exact = multilabel_exact_match(preds, labels, num_labels=NUM_LABELS)
        acc_macro = multilabel_accuracy(preds, labels, num_labels=NUM_LABELS)
        
        # How are they calculated?:
        # The metrics are calculated for each label. 
        # So if there is 4 labels, then 4 recalls are calculated.
        # These 4 values are then averaged, which is the end score that is logged.
        # The default average applied is 'macro' 
        precision_macro = multilabel_precision(preds, labels, num_labels=NUM_LABELS)
        recall_macro = multilabel_recall(preds, labels, num_labels=NUM_LABELS)
        f1_macro = multilabel_f1_score(preds, labels, num_labels=NUM_LABELS)
        
        # AUROC score of 1 is a perfect score
        # AUROC score of 0.5 corresponds to random guessing.
        auroc_macro = multilabel_auroc(preds, labels, num_labels=NUM_LABELS, average="macro", thresholds=None)
        
        metrics = {
            'accuracy_exact':  acc_exact,
            'accuracy_macro':  acc_macro,
            'precision_macro': precision_macro,
            'recall_macro':    recall_macro,
            'f1_macro':        f1_macro,
            'AUROC_macro':     auroc_macro
        }
    else:
        logit_func = torch.nn.Softmax()
        preds = logit_func(logits)
        acc_micro = multiclass_accuracy(preds, labels, num_classes=NUM_LABELS, average='micro')
        acc_macro = multiclass_accuracy(preds, labels, num_classes=NUM_LABELS, average='macro')
        precision_macro = multiclass_precision(preds, labels, num_classes=NUM_LABELS)
        recall_macro = multiclass_recall(preds, labels, num_classes=NUM_LABELS)
        f1_macro = multiclass_f1_score(preds, labels, num_classes=NUM_LABELS)
        auroc_macro = multiclass_auroc(preds, labels, num_classes=NUM_LABELS, average="macro", thresholds=None)
        
        metrics = {
            'accuracy_micro':  acc_micro,
            'accuracy_macro':  acc_macro,
            'precision_macro': precision_macro,
            'recall_macro':    recall_macro,
            'f1_macro':        f1_macro,
            'AUROC_macro':     auroc_macro
        }
        
    return metrics


all_ys = {}
for func, cvs in all_logits.items():  # func name and dict of all cvs
    all_ys[func] = {}
    for cv, val in cvs.items():
        num_samples, logits = val['num_samples'], val['logits']
        
        train_idx, val_idx = all_splits[cv]
        val_dataset = token_dataset.select(val_idx)
        targets = torch.tensor(val_dataset['labels'])
        
        ys = {}
        for l in logits:
            metrics = compute_metrics(l, targets, problem_type)
            for key in metrics.keys():
                if key not in ys.keys():
                    ys[key] = [metrics[key].item()]
                else:
                    ys[key].append(metrics[key].item())
        
        for key in ys.keys():
            if key not in all_ys[func].keys():
                all_ys[func][key] = np.array(ys[key])
            else:
                all_ys[func][key] = np.vstack([all_ys[func][key], ys[key]])
    


print('Plotting')

for metric in all_ys['entropy'].keys():
    plt.figure(figsize=(7, 5), dpi=300)
    for func in all_ys.keys():
        sem  = np.std(all_ys[func][metric], axis=0)/np.sqrt(all_ys[func][metric].shape[0])
        mean = np.mean(all_ys[func][metric], axis=0)
        plt.plot(num_samples, mean, marker='o', label=func.capitalize())
        plt.fill_between(num_samples, mean-sem, mean+sem, alpha=0.33)

    plt.legend(loc='upper left')
    plt.title(f'Danish BERT + Gyldendal\n{TARGET} - {metric_dict[metric]}', fontsize=16)
    plt.xlabel(f'Num. of Training Samples', fontsize=14)
    plt.ylabel(f'{metric_dict[metric]}', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'imgs/AL/{TARGET}/{metric}.png', bbox_inches="tight")
    plt.close()

print('Done')
