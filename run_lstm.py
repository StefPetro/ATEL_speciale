import numpy as np
from atel.data import BookCollection
from data_clean import set_seed
from lstm_model import lstm_data, lstm_text
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import fasttext
import fasttext.util

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

SEED = 42
NUM_EPOCHS = 100
set_seed(SEED)

## Load the data
book_col = BookCollection(data_file="./data/book_col_271120.pkl")

## Load fastText model
# https://fasttext.cc/docs/en/crawl-vectors.html
print('Loading fastText model...')
ft = fasttext.load_model('fasttext_model/cc.da.300.bin')  # Download from fastTexts website
fasttext.util.reduce_model(ft, 100)
print('Loading complete!')

settings = {
    'multi_label': True,
    'n_features': 100, 
    "hidden_size": 128,
    "num_layers": 4,
    "dropout": 0.2,
    "batch_size": 64,
    "learning_rate" : 1e-5,
    "output_size": 5
}

num_folds = 10
results1 = []
results2 = []
target_col = 'Semantisk univers'

for k in range(num_folds):
    print(f'STARTING CV K = {k+1}/{num_folds}')
    
    data = lstm_data(
        book_col=book_col, 
        target_col=target_col, 
        ft=ft, 
        batch_size=settings['batch_size'], 
        seq_len=171,
        seed=SEED,
        k=k
    )
    model = lstm_text(**settings)
    logger_name = f'{target_col.replace(" ", "_")}-cv{k}-max_epoch_{NUM_EPOCHS}-num_lstm_layers_{settings["num_layers"]}'
    logger = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=logger_name)
    
    trainer = Trainer(
        max_epochs = NUM_EPOCHS,
        gpus = 1 if torch.cuda.is_available() else 0,
        log_every_n_steps = 1,
        enable_checkpointing = False,
        logger = logger
    )
    trainer.fit(model, data)
    
    val_scores = trainer.validate(model, data)[0]
    score1 = val_scores['avg_val_acc']
    results1.append(score1)
    
    score2 = val_scores['val_acc_step']
    results2.append(score2)
    print('Done!')
    break

print(np.mean(results1), np.mean(results2))