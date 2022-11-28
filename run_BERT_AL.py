import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback
from torchmetrics.functional.classification import multilabel_exact_match
from torchmetrics.functional.classification import multilabel_accuracy, multilabel_f1_score
from torchmetrics.functional.classification import multilabel_recall, multilabel_precision
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score
from torchmetrics.functional.classification import multiclass_recall, multiclass_precision
from torchmetrics.functional.classification import multiclass_auroc, multilabel_auroc
from datasets import Dataset
from sklearn.model_selection import KFold
from scipy.stats import entropy
from atel.data import BookCollection
from data_clean import *
import argparse
import yaml
from yaml import CLoader
import os
import shutil

# parser = argparse.ArgumentParser(description='Arguments for running the BERT finetuning')
# parser.add_argument(
#     '--target_col',
#     help='The target column to train the BERT model on.', 
#     default=None
# )
# args = parser.parse_args()

TARGET = 'Semantisk univers'  # args.target_col

SEED = 42
NUM_SPLITS = 10
BATCH_SIZE = 16
BATCH_ACCUMALATION = 2
NUM_EPOCHS = 100
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
set_seed(SEED)

with open('target_info.yaml', 'r', encoding='utf-8') as f:
    target_info = yaml.load(f, Loader=CLoader)
    
# assert TARGET in target_info.keys()  # checks if targets is part of the actual problem columns

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


problem_type = target_info[TARGET]['problem_type']
NUM_LABELS   = target_info[TARGET]['num_labels']

print(f'STARTED TRAINING FOR: {TARGET}')
print(f'PROBLEM TYPE: {problem_type}')

df, labels = get_pandas_dataframe(book_col, TARGET)

label2id = dict(zip(labels, range(NUM_LABELS)))
id2label = dict(zip(range(NUM_LABELS), labels))

if problem_type == 'multilabel':
    multilabel = True
    p_t = "multi_label_classification"
    logit_func = torch.nn.Sigmoid()
    
else:
    multilabel = False
    p_t = "single_label_classification"
    logit_func = torch.nn.Softmax()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logit_func(torch.tensor(logits))
    labels = torch.tensor(labels).int()
    
    if problem_type == 'multilabel':
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


# using .reset_index(), to get the index of each row
dataset = Dataset.from_pandas(df.reset_index())
token_dataset = dataset.map(tokenize_function, batched=True)

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
all_splits = [k for k in kf.split(token_dataset)]


def calc_entropy(pk, problem_type='multi_class'):
    idx = pk > 0
    if problem_type == 'multi_class':
        H = -torch.sum(pk[idx] * torch.log2(pk[idx]))
    elif problem_type == 'multi_label':
        H = None
    return H


def train_e():
    pass


def active_learning_train(k, aq_func=calc_entropy):
    train_dataset   = token_dataset.select(  # Choose random subset of data
        None
    )

    unlabel_dataset = token_dataset.filter(  # filter index for data not in training set
        lambda x: x['index'] not in train_dataset['index']
    )
    
    
    model = AutoModelForSequenceClassification.from_pretrained("Maltehb/danish-bert-botxo", 
                                                               num_labels=NUM_LABELS, 
                                                               problem_type=p_t,
                                                               label2id=label2id,
                                                               id2label=id2label)
    
    # test_dataset = val_dataset.remove_columns("labels")
    # outputs = trainer.predict(test_dataset)

    logging_name = f'huggingface_logs'\
                    +f'/active_learning'\
                    +f'/{TARGET.replace(" ", "_")}'\
                    +f'/BERT-BS{BATCH_SIZE}'\
                    +f'-BA{BATCH_ACCUMALATION}'\
                    +f'-ep{NUM_EPOCHS}'\
                    +f'-seed{SEED}'\
                    +f'-WD{WEIGHT_DECAY}'\
                    +f'-LR{LEARNING_RATE}'\
                    +f'/CV_{k+1}'

    # Using max_steps instead of train_epoch since we want all experiment to train for the same
    # number of itterations.
    ## (700 samples / 32 batch size) * 5 epochs = 110 steps
    ## (700 samples / 16 batch size) * 5 epochs = 219 steps
    
    training_args = TrainingArguments(
            # [17:] removes 'huggingface_logs'
            output_dir=f'../../../../../work3/s173991/huggingface_saves/{logging_name[17:]}',
            save_total_limit=3,
            # metric_for_best_model='f1_macro',  # f1-score for now
            # greater_is_better=True,
            # load_best_model_at_end=True,
            logging_strategy='epoch',
            logging_dir=logging_name,
            report_to='tensorboard',
            evaluation_strategy='steps',  # we are NOT using epochs, but max steps
            eval_steps=10,
            seed=SEED,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=BATCH_ACCUMALATION,
            # num_train_epochs=NUM_EPOCHS,
            max_steps=110,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
    
    trainer = Trainer(
        model=model,    
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    trainer.train(resume_from_checkpoint=True)





for k in range(NUM_SPLITS):
    train_idx, val_idx = all_splits[k]
    train_dataset = token_dataset.select(train_idx)
    val_dataset   = token_dataset.select(val_idx)

    # ds = train_dataset.train_test_split(train_size=0.2)
    
    labelled_ds   = train_dataset.select(  # Choose random subset of data
        np.random.choice(len(train_dataset), int(len(train_dataset)*0.2), replace=False)
    )

    unlabelled_ds = train_dataset.filter(  # filter index for data not in training set
        lambda x: x['index'] not in labelled_ds['index']
    )

    # Loop through the "unlabelled" data until we are training on the whole set
    ##  Start dataset size: 700*0.2  = 140
    ##  560 samples//32 batch size (+ 1 if not even) = 18 steps
    for c in range(18):
        
        
        
        labelled_ds   = train_dataset.select(  # Choose random subset of data
            None  # TODO: Fill with topk indices
        )

        unlabelled_ds = train_dataset.filter(  # filter index for data not in training set
            lambda x: x['index'] not in labelled_ds['index']
        )
        
        break
        
    
    break
    
pk = torch.tensor([0.5, 0., 0.5])
print(calc_entropy(pk))

print(entropy(pk, base=2))





