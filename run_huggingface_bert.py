import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import evaluate
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelAccuracy
from sklearn.model_selection import KFold
from data_clean import *
from atel.data import BookCollection

SEED = 42
NUM_SPLITS = 10
set_seed(SEED)

book_col = BookCollection(data_file="./data/book_col_271120.pkl")
df, labels = get_pandas_dataframe(book_col, 'Semantisk univers')

NUM_LABELS = len(labels)

metric = evaluate.load("accuracy")
acc_metric = Accuracy(subset_accuracy=True)
multilabel_acc = MultilabelAccuracy(num_labels=NUM_LABELS)
multilabel_acc_sample = MultilabelAccuracy(num_labels=NUM_LABELS, average=None)

tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics_multiclass(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics_multilabel(eval_pred):
    logits, labels = eval_pred
    
    metrics = {
        "accuracy": acc_metric(torch.tensor(logits), torch.tensor(labels).int()),
        "ml_acc": multilabel_acc(torch.tensor(logits), torch.tensor(labels).int()),
        "sample_acc": multilabel_acc_sample(torch.tensor(logits), torch.tensor(labels).int())
    }
    
    return metrics


dataset = Dataset.from_pandas(df)
token_dataset = dataset.map(tokenize_function, batched=True)

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
all_splits = [k for k in kf.split(token_dataset)]

for k in range(NUM_SPLITS):
    train_idx, val_idx = all_splits[k]
    train_dataset = token_dataset.select(train_idx)
    val_dataset   = token_dataset.select(val_idx)

    model = AutoModelForSequenceClassification.from_pretrained("Maltehb/danish-bert-botxo", 
                                                               num_labels=NUM_LABELS, 
                                                               problem_type="multi_label_classification")
    
    training_args = TrainingArguments(
        output_dir="test_trainer", 
        save_strategy='no',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy='epoch',
        report_to='tensorboard',
        logging_dir='huggingface_logs',
        logging_steps=5,
        num_train_epochs=1
    )

    trainer = Trainer(
        model=model,    
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_multilabel,
    )

    trainer.train()
    break

