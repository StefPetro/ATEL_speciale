import json
import torch
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
TARGET = 'Genre'
NUM_LABELS = 15
set_seed(SEED)

all_logits = {}
for func in ['entropy', 'random']:
    filepath = f'huggingface_logs'\
            +f'/active_learning'\
            +f'/BERT_mlm_gyldendal'\
            +f'/{func}'\
            +f'/Genre'\
            +f'/BS16-BA4-MS3300-seed42-WD0.01-LR2e-05'\
            +f'/CV_1'\
            +f'/test_logits.json'
   
    with open(filepath, 'r') as loadfile:
        data = json.load(loadfile)

    print(data.keys())
    num_samples = data['num_train_samples']
    logits = torch.tensor(data['logits'])
    all_logits[func] = logits
    
book_col = BookCollection(data_file="./data/book_col_271120.pkl")

tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

df, labels = get_pandas_dataframe(book_col, TARGET)

# using .reset_index(), to get the index of each row
dataset = Dataset.from_pandas(df.reset_index())
token_dataset = dataset.map(tokenize_function, batched=True)

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
all_splits = [k for k in kf.split(token_dataset)]

train_idx, val_idx = all_splits[0]
val_dataset   = token_dataset.select(val_idx)

targets = torch.tensor(val_dataset['labels'])
print(targets.shape)

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


print(all_logits.keys())
for func, logits in all_logits.items():
    y = []
    for l in logits:
        metrics = compute_metrics(l, targets, problem_type='multilabel')
        y.append(metrics['accuracy_exact'].item())
    plt.plot(num_samples, y, marker='o', label=func)

print('Plotting')
# plt.plot(num_samples, y, marker='o')
plt.legend()
plt.show()
