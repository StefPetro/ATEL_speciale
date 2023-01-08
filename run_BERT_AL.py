import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from torchmetrics.functional.classification import (
    multilabel_exact_match,
    multilabel_accuracy, multiclass_accuracy,
    multilabel_f1_score, multiclass_f1_score,
    multilabel_recall, multiclass_recall,
    multilabel_precision, multiclass_precision,
    multilabel_auroc, multiclass_auroc
)
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import KFold
from atel.data import BookCollection
from data_clean import *
from acquisition_functions import *
import argparse
import json
import yaml
from yaml import CLoader
import os
import shutil

parser = argparse.ArgumentParser(description='Arguments for running the BERT finetuning')
parser.add_argument(
    '--target_col',
    help='The target column to train the BERT model on.', 
    default=None
)
parser.add_argument(
    '--cv',
    help='Which cross-validation fold to use - Can be 1-10.',
    default=1,
    type=int
)
parser.add_argument(
    '--acq_function',
    help='Which acquisition function to use when choosing unlabeled data. Can be "entropy" or "random". Defaults to "entropy"',
    default='entropy',
    type=str
)
args = parser.parse_args()

TARGET = args.target_col
CV = args.cv - 1  # minus 1 as we want the --cv argument to be 1-10
ACQ_FUNC = args.acq_function

print(f'Acquisition function: {ACQ_FUNC} chosen')
if ACQ_FUNC == 'entropy':
    acq_function = calc_entropy
elif ACQ_FUNC == 'random':
    acq_function = random_acquisition

assert ACQ_FUNC in ['entropy', 'random']

SEED = 42
NUM_SPLITS = 10
BATCH_SIZE = 16
BATCH_ACCUMALATION = 4
NUM_EPOCHS = 75
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
set_seed(SEED)

with open('target_info.yaml', 'r', encoding='utf-8') as f:
    target_info = yaml.load(f, Loader=CLoader)
    
# assert TARGET in target_info.keys()  # checks if targets is part of the actual problem columns

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

tokenizer = AutoTokenizer.from_pretrained("../../../../../work3/s173991/huggingface_models/BERT_mlm_gyldendal")

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


def AL_train(labeled_ds: Dataset, unlabeled_ds: Dataset, test_ds: Dataset):    
    
    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained("../../../../../work3/s173991/huggingface_models/BERT_mlm_gyldendal", 
                                                                num_labels=NUM_LABELS, 
                                                                problem_type=p_t,
                                                                label2id=label2id,
                                                                id2label=id2label)
        return model
        
    # +f'-ep{NUM_EPOCHS}'\
    logging_name = f'huggingface_logs'\
                    +f'/active_learning'\
                    +f'/BERT_mlm_gyldendal'\
                    +f'/{ACQ_FUNC}'\
                    +f'/{TARGET.replace(" ", "_")}'\
                    +f'/BS{BATCH_SIZE}'\
                    +f'-BA{BATCH_ACCUMALATION}'\
                    +f'-MS{3300}'\
                    +f'-seed{SEED}'\
                    +f'-WD{WEIGHT_DECAY}'\
                    +f'-LR{LEARNING_RATE}'\
                    +f'/CV_{CV+1}'\
                    +f'/num_samples_{labeled_ds.num_rows}'

    # Using max_steps instead of train_epoch since we want all experiment to train for the same
    # number of itterations.
    ## (700 samples / 32 batch size) * 100 epochs = 2187.5 steps
    ## (700 samples / 16 batch size) * 100 epochs = 4375 steps
    ## (700 samples / 16 batch size) * 75 epochs = 3281.25 steps
    
    training_args = TrainingArguments(
            # [17:] removes 'huggingface_logs'
            output_dir=f'../../../../../work3/s173991/huggingface_saves/{logging_name[17:]}',
            # save_strategy='steps',
            # save_steps=43,
            save_total_limit=1,
            # metric_for_best_model='f1_macro',  # f1-score for now
            # greater_is_better=True,
            # load_best_model_at_end=True,
            logging_strategy='steps',
            logging_steps=50,
            logging_dir=logging_name,
            report_to='tensorboard',
            evaluation_strategy='steps',  # 'epoch'
            eval_steps=50,
            seed=SEED,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=BATCH_ACCUMALATION,
            # num_train_epochs=NUM_EPOCHS,
            max_steps=3300,  # If using max_steps switch strategies to steps
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
    
    trainer = Trainer(
        model_init=model_init,    
        args=training_args,
        train_dataset=labeled_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    trainer.train()  # resume_from_checkpoint=True
    
    eval_ds      = unlabeled_ds.remove_columns("labels")
    eval_logits  = trainer.predict(eval_ds).predictions
    
    test_ds      = test_ds.remove_columns("labels")
    test_logits  = trainer.predict(test_ds).predictions
    
    return eval_logits, test_logits


def create_initial_labeled_dataset(full_ds: Dataset, init_size: float=0.2) -> Dataset:
    total_size   = full_ds.num_rows
    labeled_ds   = full_ds.select(  # Choose random subset of data
        np.random.choice(total_size, int(total_size*init_size), replace=False)
    )

    unlabeled_ds = train_dataset.filter(  # filter index for data not in training set
        lambda x: x['index'] not in labeled_ds['index']
    )
    
    return labeled_ds, unlabeled_ds
    

def update_datasets(
    labeled_ds:   Dataset, 
    unlabeled_ds: Dataset,
    eval_logits,
    problem_type: str='multilabel',
    acq_size:      int=32,
    acq_func=calc_entropy
) -> Tuple[Dataset, Dataset]:

    entropy     = acq_func(eval_logits, problem_type=problem_type)
    top_samples = torch.topk(entropy,
                             acq_size if unlabeled_ds.num_rows >= acq_size else unlabeled_ds.num_rows)
    
        
    # Usually you would have an oracle that would label the unlabaled data points.
    # In this case we already know the labels.
    new_train_samples = unlabeled_ds.select(top_samples.indices.tolist())
    # new_train_samples = ask_oracle(new_train_samples)

    labeled_ds = concatenate_datasets([labeled_ds, new_train_samples])
    
    unlabeled_ds = train_dataset.filter(  # filter index for data not in training set
        lambda x: x['index'] not in labeled_ds['index']
    )
        
    return labeled_ds, unlabeled_ds
        

all_test_logits = {
    'num_train_samples': [],
    'logits': []
}
train_idx, val_idx = all_splits[CV]
train_dataset = token_dataset.select(train_idx)
val_dataset   = token_dataset.select(val_idx)

# ds = train_dataset.train_test_split(train_size=0.2)

unlabeled_ds = token_dataset.select(train_idx)

while unlabeled_ds.num_rows > 0:
    
    if unlabeled_ds.num_rows == train_dataset.num_rows:
        labeled_ds, unlabeled_ds = create_initial_labeled_dataset(train_dataset, init_size=0.5)
    else:
        labeled_ds, unlabeled_ds = update_datasets(
            labeled_ds, 
            unlabeled_ds, 
            eval_logits, 
            problem_type=problem_type,
            acq_size=32,
            acq_func=acq_function
        )
    
    print(f'Labeled dataset size: {labeled_ds.num_rows}/{train_dataset.num_rows}')
    
    eval_logits, test_logits = AL_train(labeled_ds, unlabeled_ds, val_dataset)
    
    all_test_logits['num_train_samples'].append(labeled_ds.num_rows)
    all_test_logits['logits'].append(test_logits.tolist())
    
    
    logging_name = f'huggingface_logs'\
                    +f'/active_learning'\
                    +f'/BERT_mlm_gyldendal'\
                    +f'/{ACQ_FUNC}'\
                    +f'/{TARGET.replace(" ", "_")}'\
                    +f'/BS{BATCH_SIZE}'\
                    +f'-BA{BATCH_ACCUMALATION}'\
                    +f'-MS{3300}'\
                    +f'-seed{SEED}'\
                    +f'-WD{WEIGHT_DECAY}'\
                    +f'-LR{LEARNING_RATE}'\
                    +f'/CV_{CV+1}'\
                    +f'/num_samples_{labeled_ds.num_rows}'
    
    
    ## Removes the saved checkpoints, as they take too much space
    for f in os.listdir(f'../../../../../work3/s173991/huggingface_saves/{logging_name[17:]}'):
        shutil.rmtree(os.path.join(f'../../../../../work3/s173991/huggingface_saves/{logging_name[17:]}', f))

# Save the test logits for future analysis
# +f'-ep{NUM_EPOCHS}'\
filepath = f'huggingface_logs'\
            +f'/active_learning'\
            +f'/BERT_mlm_gyldendal'\
            +f'/{ACQ_FUNC}'\
            +f'/{TARGET.replace(" ", "_")}'\
            +f'/BS{BATCH_SIZE}'\
            +f'-BA{BATCH_ACCUMALATION}'\
            +f'-MS{3300}'\
            +f'-seed{SEED}'\
            +f'-WD{WEIGHT_DECAY}'\
            +f'-LR{LEARNING_RATE}'\
            +f'/CV_{CV+1}'
    
with open(f'{filepath}/test_logits.json', 'w') as outfile:
    outfile.write(json.dumps(all_test_logits))

print('File is saved. Run is finished!')
