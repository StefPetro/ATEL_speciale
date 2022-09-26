import numpy as np
from atel.data import BookCollection
from data_clean import set_seed
from lstm_model import lstm_data, lstm_text
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import fasttext

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

SEED = 42
NUM_EPOCHS = 10
set_seed(SEED)

## Load the data
book_col = BookCollection(data_file="./data/book_col_271120.pkl")

## Load fastText model
# https://fasttext.cc/docs/en/crawl-vectors.html
print('Loading fastText model...')
ft = fasttext.load_model('fasttext_model/cc.da.300.bin')  # Download from fastTexts website
print('Loading complete!')

settings = {
    'multi_label': True,
    'n_features': 300, 
    "hidden_size": 256*2, 
    "num_layers": 2, 
    "dropout": 0.2, 
    "batch_size": 16, 
    "learning_rate" : 5e-5,
    "output_size": 15
}

num_folds = 10
results1 = []
results2 = []
target_col = 'Genre'

for k in range(num_folds):
    print(f'STARTING CV K = {k+1}/{num_folds}')
    
    model = lstm_text(**settings)
    data = lstm_data(
        book_col=book_col, 
        target_col=target_col, 
        ft=ft, batch_size=settings['batch_size'], 
        seed=SEED, 
        k=k
    )
    logger = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=f'{target_col}-cv{k}-max_epoch_{NUM_EPOCHS}')
    
    trainer = Trainer(
        max_epochs = NUM_EPOCHS,
        gpus = 1 if torch.cuda.is_available() else 0,
        log_every_n_steps = 1,
        logger = logger
    )
    trainer.fit(model, data)
    
    val_scores = trainer.validate(model, data)[0]
    score1 = val_scores['avg_val_acc']
    results1.append(score1)
    
    score2 = val_scores['val_acc_step']
    results2.append(score2)
    break

print(np.mean(results1), np.mean(results2))