import warnings

import fasttext
import fasttext.util
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import Trainer
from yaml import CLoader
import argparse

from atel.data import BookCollection
from data_clean import set_seed
from lstm_model import lstm_data, lstm_text

warnings.filterwarnings("ignore", ".*does not have many workers.*")

parser = argparse.ArgumentParser(description="Arguments for training the LSTM model")
parser.add_argument(
    "--target_col", help="The target column to train the LSTM model on.", default=None
)
parser.add_argument(
    "--cv", help="Which cross-validation fold to use. Can be 1-10", default=1, type=int
)
args = parser.parse_args()

TARGET = args.target_col
CV = args.cv - 1  # minus 1 as we want the --cv argument to be 1-10

SEED = 42
NUM_FOLDS = 10
NUM_EPOCHS = 10000
EMBEDDING_SIZE = 100
set_seed(SEED)

with open("target_info.yaml", "r", encoding="utf-8") as f:
    target_info = yaml.load(f, Loader=CLoader)

problem_type = target_info[TARGET]["problem_type"]
NUM_LABELS = target_info[TARGET]["num_labels"]

## Load the data
book_col = BookCollection(data_file="./data/book_col_271120.pkl")

## Load fastText model
# https://fasttext.cc/docs/en/crawl-vectors.html
print("Loading fastText model...")
ft = fasttext.load_model(
    "fasttext_model/cc.da.300.bin"
)  # Download from fastTexts website
fasttext.util.reduce_model(ft, 100)
print("Loading complete!")

settings = {
    "multi_label": True if problem_type == "multilabel" else False,
    "n_features": EMBEDDING_SIZE,
    "hidden_size": 256,
    "num_layers": 4,
    "dropout": 0.2,
    "batch_size": 128,
    "learning_rate": 1e-5,
    "output_size": NUM_LABELS,
}

print(f"RUNNING CV K = {CV+1}/{NUM_FOLDS}")

model = lstm_text(**settings)
data = lstm_data(
    book_col=book_col,
    target_col=TARGET,
    ft=ft,
    batch_size=settings["batch_size"],
    seq_len=128,
    seed=SEED,
    k=CV,
)
logger_name = f'{TARGET.replace(" ", "_")}-cv{k}-max_epoch_{NUM_EPOCHS}'
logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs/", name=logger_name)

trainer = Trainer(
    max_epochs=NUM_EPOCHS,
    gpus=1 if torch.cuda.is_available() else 0,
    log_every_n_steps=1,
    enable_checkpointing=False,
    logger=logger,
)
trainer.fit(model, data)

print("Done Training!")

best_epoch = model.best_epoch
best_preds = model.best_model_logits
torch.save(
    best_preds, f"lightning_logs/{logger_name}/{TARGET}_best_epoch_{best_epoch}.pt"
)
print("Saved model logits for best F1-score")
