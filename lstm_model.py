from typing import Union, Tuple, Optional

from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from data_clean import *

settings = {
    'multi_label': False,
    'n_features': 300, 
    "hidden_size": 256, 
    "num_layers": 1, 
    "dropout": 0.2, 
    "batch_size": 32, 
    "learning_rate" : 1e-4,
    "output_size": 1
}


class lstm_text(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.multi_label   = kwargs.get('multi_label',   False)
        self.n_features    = kwargs.get('n_features',    300)
        self.hidden_size   = kwargs.get('hidden_size',   256)
        self.num_layers    = kwargs.get("num_layers",    1)
        self.dropout       = kwargs.get("dropout",       0.2)
        self.batch_size    = kwargs.get("batch_size",    32)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.output_size   = kwargs.get('output_size',   1)

        self.lstm = nn.LSTM(input_size    = self.n_features,
                            hidden_size   = self.hidden_size,
                            num_layers    = self.num_layers,
                            dropout       = self.dropout,
                            bidirectional = True,
                            batch_first   = True)

        self.dropout = nn.Dropout(p=0.2)
        
        self.out_layer = nn.Linear(in_features  = self.hidden_size*2,  # times 2 because bidirectional
                                   out_features = self.output_size)
        
        if self.multi_label:  # if we are trying to solve a multi label problem
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
            
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optim
    
    def forward(self, x):
        # lstm input: (N, L, H_in) = (batch_size, seq_len, embedding_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.dropout(lstm_out)
        out = F.gelu(out)
        out = self.out_layer(out[:, -1])
        return out
    
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        return loss
    
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        return loss


## TODO: implement K-fold Cross validation - make sure it's the same folds for all tasks
class lstm_data(pl.LightningDataModule):
    def __init__(self, book_col, target_col: str, seq_len: int = 256, 
                       k: int = 0, batch_size: int = 32):
        super().__init__()
        self.book_col   = book_col
        self.k          = k
        self.batch_size = batch_size
        self.target_col = target_col
        self.seq_len    = seq_len
    
    
    def setup(self, stage: Optional[str] = None):
        
        book_ids, X = get_fasttext_embeddings(self.book_col, self.seq_len)
        target_ids, targets, labels = get_labels(self.book_col, self.target_col)

        mask = torch.isin(torch.from_numpy(target_ids), torch.from_numpy(book_ids))
        y = torch.from_numpy(targets[mask]).float()
        
        train_size = int(len(y)*0.9)
        val_size = len(y) - train_size 
        
        print(X.shape, y.shape)
        full_data = TensorDataset(X, y)
        self.train_data, self.val_data = random_split(full_data, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)





