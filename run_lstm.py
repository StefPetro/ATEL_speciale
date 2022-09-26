import numpy as np
from atel.data import BookCollection
from data_clean import set_seed
from lstm_model import lstm_data, lstm_text
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import fasttext

SEED = 42
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
    "hidden_size": 256, 
    "num_layers": 1, 
    "dropout": 0.2, 
    "batch_size": 8, 
    "learning_rate" : 1e-4,
    "output_size": 15
}

num_folds = 10
results = []
target_col = 'Genre'

for k in range(num_folds):
    model = lstm_text(**settings)
    data = lstm_data(
        book_col=book_col, 
        target_col=target_col, 
        ft=ft, batch_size=settings['batch_size'], 
        seed=SEED, 
        k=k
    )
    logger = pl.loggers.TensorBoardLogger(save_dir='lightning_logs')
    
    trainer = Trainer(
        max_epochs = 1,
        gpus = 1 if torch.cuda.is_available() else 0,
        log_every_n_steps = 5,
        logger = logger
    )
    trainer.fit(model, data)
    
    val_scores = trainer.validate(model, data)
    print(val_scores)
    
    break
    # results.append(score)

np.mean(results)