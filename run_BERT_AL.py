import numpy as np
import torch
from transformers import AutoTokenizer
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
BATCH_ACCUMALATION = 4
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

# using .reset_index(), to get the index of each row
dataset = Dataset.from_pandas(df.reset_index())
token_dataset = dataset.map(tokenize_function, batched=True)
print(token_dataset)

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
all_splits = [k for k in kf.split(token_dataset)]

def calc_entropy(pk, problem_type='multi_class'):
    idx = pk > 0
    if problem_type == 'multi_class':
        H = -torch.sum(pk[idx] * torch.log2(pk[idx]))
    elif problem_type == 'multi_label':
        H = None
    return H


train_dataset   = token_dataset.select(  # Choose random subset of data
        None
    )

unlabel_dataset = token_dataset.filter(  # filter index for data not in training set
        lambda x: x['index'] not in train_dataset['index']
    )

def train_e():
    pass


def active_learning_train(k, aq_func=calc_entropy):
    train_idx, val_idx = all_splits[k]
    train_dataset = token_dataset.select(train_idx)
    val_dataset   = token_dataset.select(val_idx)
    
    # trainer = Trainer(
    #     model=model,    
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     compute_metrics=compute_metrics,
    #     tokenizer=tokenizer
    # )
    
    # trainer.train()
    
    # test_dataset = val_dataset.remove_columns("labels")
    # outputs = trainer.predict(test_dataset)


pk = torch.tensor([0.5, 0., 0.5])
print(calc_entropy(pk))

print(entropy(pk, base=2))





