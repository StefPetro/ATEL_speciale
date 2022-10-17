from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelAccuracy, MulticlassAccuracy
from sklearn.model_selection import KFold
from data_clean import *
from atel.data import BookCollection
import argparse
import yaml
from yaml import CLoader

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--target_col',
    help='The target column to train the BERT model on.', 
    default=None
)
args = parser.parse_args()

TARGET = args.target_col

SEED = 42
NUM_SPLITS = 10
BATCH_SIZE = 16
NUM_EPOCHS = 50
set_seed(SEED)

with open('target_problem_type.yaml', 'r', encoding='utf-8') as f:
    target_problems = yaml.load(f, Loader=CLoader)

assert TARGET in target_problems.keys()  # checks if targets is part of the actual problem columns

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


problem_type = target_problems[TARGET]

  
print(f'STARTED TRAINING FOR: {TARGET}')
print(f'PROBLEM TYPE: {problem_type}')

df, labels = get_pandas_dataframe(book_col, TARGET)

NUM_LABELS = len(labels)

label2id = dict(zip(labels, range(NUM_LABELS)))
id2label = dict(zip(range(NUM_LABELS), labels))

if problem_type == 'multilabel':
    multilabel = True
    p_t = "multi_label_classification"
    logit_func = torch.nn.Sigmoid()
    acc_metric = Accuracy(subset_accuracy=True)
    
else:
    multilabel = False
    p_t = "single_label_classification"
    logit_func = torch.nn.Softmax()
    acc_metric = MulticlassAccuracy(num_classes=NUM_LABELS)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metrics = {
        "accuracy": acc_metric(logit_func(torch.tensor(logits)), torch.tensor(labels).int()),
    }
    return metrics


dataset = Dataset.from_pandas(df)
token_dataset = dataset.map(tokenize_function, batched=True)

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
all_splits = [k for k in kf.split(token_dataset)]

for k in range(NUM_SPLITS):
    
    print(f'\nTRAINING CV {k+1}/{NUM_SPLITS} - {TARGET}')
    
    train_idx, val_idx = all_splits[k]
    train_dataset = token_dataset.select(train_idx)
    val_dataset   = token_dataset.select(val_idx)

    model = AutoModelForSequenceClassification.from_pretrained("Maltehb/danish-bert-botxo", 
                                                            num_labels=NUM_LABELS, 
                                                            problem_type=p_t,
                                                            label2id=label2id,
                                                            id2label=id2label)
    
    training_args = TrainingArguments(
        output_dir="test_trainer",
        seed=SEED,
        save_strategy='no',
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        evaluation_strategy='epoch',
        report_to='tensorboard',
        logging_strategy='steps',
        logging_steps=1,
        logging_dir=f'huggingface_logs/{TARGET}/BERT-{TARGET}-CV_{k+1}-batch_size_{BATCH_SIZE}-epochs_{NUM_EPOCHS}-seed_{SEED}',
        num_train_epochs=NUM_EPOCHS
    )

    trainer = Trainer(
        model=model,    
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    trainer.train()
