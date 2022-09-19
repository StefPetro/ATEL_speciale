from typing import Union, Tuple, Optional

from sklearn.model_selection import KFold
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer

settings = {
    'n_features': 32, 
    "hidden_size": 128, 
    "num_layers": 4, 
    "dropout": 0.1, 
    "batch_size": 64, 
    "learning_rate" : 1e-4, 
    "output_size": 1
}


class lstm_text(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        ## TODO: Add default values to .get method == .get('...', 32)
        self.n_features    = kwargs.get('n_features')
        self.hidden_size   = kwargs.get('hidden_size')
        self.num_layers    = kwargs.get("num_layers")
        self.dropout       = kwargs.get("dropout")
        self.batch_size    = kwargs.get("batch_size")
        self.learning_rate = kwargs.get('learning_rate')
        self.output_size   = kwargs.get('output_size')

        self.lstm = nn.LSTM(input_size = self.n_features,
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            dropout = self.dropout)

        self.linear = nn.Linear(in_features = self.hidden_size, 
                                out_features = self.output_size)
    
    def forward(self, x):
        pass
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optim
     
    def training_step(self, train_batch):
        
        ## TODO: Define a loss for the training step
        return None # needs to be loss
    
    def validation_epoch_end(self, val_batch):
        
        ## TODO: Define a loss for the validation step
        return None # needs to be loss


## TODO: implement K-fold Cross validation - make sure it's the same folds for all tasks
class lstm_data(pl.LightningDataModule):
    def __init__(self, data, k: int = 0):
        super().__init__()
        
    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None
