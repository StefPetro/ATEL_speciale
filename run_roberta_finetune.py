from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import Dataset
from torchmetrics.functional.classification import multilabel_exact_match
from torchmetrics.functional.classification import (
    multilabel_accuracy,
    multilabel_f1_score,
)
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)
from sklearn.model_selection import KFold
from data_clean import *
from atel.data import BookCollection
import argparse
import yaml
from yaml import CLoader
import os
import torch
from transformers import RobertaTokenizer
import shutil

parser = argparse.ArgumentParser(
    description="Arguments for running the BERT finetuning"
)
parser.add_argument(
    "--target_col", help="The target column to train the BERT model on.", default=None
)
args = parser.parse_args()

TARGET = args.target_col

SEED = 42
NUM_SPLITS = 10
BATCH_SIZE = 16
BATCH_ACCUMALATION = 4
NUM_EPOCHS = 1000
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
set_seed(SEED)

with open("target_info.yaml", "r", encoding="utf-8") as f:
    target_info = yaml.load(f, Loader=CLoader)

# checks if targets is part of the actual problem columns
assert TARGET in target_info.keys()

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

tokenizer = RobertaTokenizer.from_pretrained("./tokenizers/BPEtokenizer_121022")


def tokenize_function(examples):
    return tokenizer(
        examples["text"], max_length=128, padding="max_length", truncation=True
    )


problem_type = target_info[TARGET]["problem_type"]
NUM_LABELS = target_info[TARGET]["num_labels"]

print(f"STARTED TRAINING FOR: {TARGET}")
print(f"PROBLEM TYPE: {problem_type}")

df, labels = get_pandas_dataframe(book_col, TARGET)

label2id = dict(zip(labels, range(NUM_LABELS)))
id2label = dict(zip(range(NUM_LABELS), labels))

if problem_type == "multilabel":
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

    if problem_type == "multilabel":
        acc_exact = multilabel_exact_match(preds, labels, num_labels=NUM_LABELS)
        acc_macro = multilabel_accuracy(preds, labels, num_labels=NUM_LABELS)
        f1_macro = multilabel_f1_score(preds, labels, num_labels=NUM_LABELS)

        metrics = {
            "accuracy_exact": acc_exact,
            "accuracy_macro": acc_macro,
            "f1_macro": f1_macro,
        }
    else:
        acc_micro = multiclass_accuracy(
            preds, labels, num_classes=NUM_LABELS, average="micro"
        )
        acc_macro = multiclass_accuracy(
            preds, labels, num_classes=NUM_LABELS, average="macro"
        )
        f1_macro = multiclass_f1_score(preds, labels, num_classes=NUM_LABELS)

        metrics = {
            "accuracy_micro": acc_micro,
            "accuracy_macro": acc_macro,
            "f1_macro": f1_macro,
        }

    return metrics


dataset = Dataset.from_pandas(df)
token_dataset = dataset.map(tokenize_function, batched=True)

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
all_splits = [k for k in kf.split(token_dataset)]

for k in range(NUM_SPLITS):

    print(f"\nTRAINING CV {k+1}/{NUM_SPLITS} - {TARGET}")

    train_idx, val_idx = all_splits[k]
    train_dataset = token_dataset.select(train_idx)
    val_dataset = token_dataset.select(val_idx)

    model = AutoModelForSequenceClassification.from_pretrained(
        "models/BabyBERTa_241022",
        num_labels=NUM_LABELS,
        problem_type=p_t,
        label2id=label2id,
        id2label=id2label,
    )

    # optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer = optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=100
    # )

    logging_name = (
        f"huggingface_logs"
        + f"/{TARGET}"
        + f"/BERT-BS{BATCH_SIZE}"
        + f"-BA{BATCH_ACCUMALATION}"
        + f"-ep{NUM_EPOCHS}"
        + f"-seed{SEED}"
        + f"-WD{WEIGHT_DECAY}"
        + f"-LR{LEARNING_RATE}"
        + f"/CV_{k+1}"
    )

    training_args = TrainingArguments(
        output_dir=f"huggingface_saves/{TARGET}",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="eval_f1_macro",  # f1-score for now
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        logging_dir=logging_name,
        report_to="tensorboard",
        evaluation_strategy="epoch",
        seed=SEED,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model("BEST-RoBERTa")

    trainer.model.eval()
    outputs = trainer.model(
        input_ids=torch.tensor(val_dataset["input_ids"]).to('cuda'),
        labels=torch.tensor(val_dataset["labels"]).to('cuda'),
        attention_mask=torch.tensor(val_dataset["attention_mask"]).to('cuda')
    )

    torch.save(outputs.logits, f"{logging_name}/{TARGET}_CV{k+1}_best_model_logits.pt")

    # ## Removes the saved checkpoints, as they take too much space
    # for f in os.listdir(f"huggingface_saves/{TARGET}"):
    #     shutil.rmtree(os.path.join(f"huggingface_saves/{TARGET}", f))

    break
