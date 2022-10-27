from typing import Union, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, dataset
from sklearn.model_selection import KFold
import fasttext
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.functional.classification import multilabel_exact_match
from torchmetrics.functional.classification import multilabel_accuracy, multilabel_f1_score
from torchmetrics.functional.classification import multilabel_recall, multilabel_precision
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score
from torchmetrics.functional.classification import multiclass_recall, multiclass_precision
from torchmetrics.functional.classification import multiclass_auroc, multilabel_auroc
from data_clean import *


settings = {
    'multi_label': True,
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
        self.l1_size       = kwargs.get('l1_size',       512)
        self.l2_size       = kwargs.get('l2_size',       256)
        self.dropout       = kwargs.get("dropout",       0.2)
        self.batch_size    = kwargs.get("batch_size",    32)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.output_size   = kwargs.get('output_size',   1)
        
        self.save_hyperparameters()
        
        self.best_model_logits = None
        self.best_f1 = 0
        self.best_epoch = 0
        self.epoch = 0
        
        self.lstm = nn.LSTM(input_size    = self.n_features,
                            hidden_size   = self.hidden_size,
                            num_layers    = self.num_layers,
                            dropout       = self.dropout,
                            bidirectional = True,
                            batch_first   = True)

        self.l1        = nn.Linear(in_features  = self.hidden_size*2,  # multiplied by 2 if bidirectional,
                                   out_features = self.l1_size)
        
        self.l2        = nn.Linear(in_features  = self.l1_size,
                                   out_features = self.l2_size)
        
        self.out_layer = nn.Linear(in_features  = self.l2_size,
                                   out_features = self.output_size)
        
        
        self.dropout = nn.Dropout(p=0.2)
        
        if self.multi_label:  # if we are trying to solve a multi label problem
            print('Set to multi label classification')
            self.loss_func  = nn.BCEWithLogitsLoss()
            self.logit_func = nn.Sigmoid()            
            
        else:
            print('Set to multi class classification')
            self.loss_func  = nn.CrossEntropyLoss()
            self.logit_func = nn.Softmax()
            
    
    def compute_metrics(self, preds, targets, logit_func, multi_label, current):
        """ Function that compute relevant metrics to log """
        
        preds = logit_func(preds)
        
        if multi_label:
            acc_exact = multilabel_exact_match(preds, targets, num_labels=self.output_size)
            acc_macro = multilabel_accuracy(preds, targets, num_labels=self.output_size) 
            # precision_macro = multilabel_precision(preds, targets, num_labels=self.output_size)
            # recall_macro = multilabel_recall(preds, targets, num_labels=self.output_size)
            f1_macro = multilabel_f1_score(preds, targets, num_labels=self.output_size)
            auroc_macro = multilabel_auroc(preds, targets, num_labels=self.output_size, average="macro", thresholds=None)
            
            metrics = {
                f'{current}_acc_exact':       acc_exact,
                f'{current}_acc_macro':       acc_macro,
                # f'{current}_step_precision_macro': precision_macro,
                # f'{current}_step_recall_macro':    recall_macro,
                f'{current}_f1_macro':        f1_macro,
                f'{current}_AUROC_macro':     auroc_macro
            }

        else:
            acc_micro = multiclass_accuracy(preds, targets, num_classes=self.output_size, average='micro')
            acc_macro = multiclass_accuracy(preds, targets, num_classes=self.output_size, average='macro')
            # precision_macro = multiclass_precision(preds, targets, num_classes=self.output_size)
            # recall_macro = multiclass_recall(preds, targets, num_classes=self.output_size)
            f1_macro = multiclass_f1_score(preds, targets, num_classes=self.output_size)
            auroc_macro = multiclass_auroc(preds, targets, num_classes=self.output_size, average="macro", thresholds=None)
            
            metrics = {
                f'{current}_acc_micro':       acc_micro,
                f'{current}_acc_macro':       acc_macro,
                # f'{current}_step_precision_macro': precision_macro,
                # f'{current}_step_recall_macro':    recall_macro,
                f'{current}_f1_macro':        f1_macro,
                f'{current}_AUROC_macro':     auroc_macro
            }
            
        return metrics
                
            
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        return optim
    
    
    def forward(self, x):
        # lstm input:
        # (N, L, H_in) = (batch_size, seq_len, embedding_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1])
        
        l1  = F.gelu(self.dropout(self.l1(lstm_out)))
        l2  = F.gelu(self.dropout(self.l2(l1)))
        out = self.out_layer(l2)
        return out
    
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        preds = self(x)
        loss = self.loss_func(preds, y)

        # metrics = self.compute_metrics(preds, y, self.logit_func, self.multi_label, 'train_step')
        self.log('train_step_loss', loss)
        # self.log_dict(metrics)
        return {'loss': loss}
    
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        preds = self(x)
        loss = self.loss_func(preds, y)
        
        self.log('val_step_loss', loss)
        return {'preds': preds, 'target': y}

    
    def validation_epoch_end(self, outputs) -> None:
        all_preds = torch.cat([out['preds'] for out in outputs])
        y         = torch.cat([out['target'] for out in outputs])
                
        loss = self.loss_func(all_preds, y).mean()
        self.log("val_epoch_loss", loss)
        
        metrics = self.compute_metrics(all_preds, y, self.logit_func, self.multi_label, 'val_epoch')
        self.log_dict(metrics)
        
        if metrics['val_epoch_f1_macro'] > self.best_f1:
            self.best_model_logits = all_preds
            self.best_f1 = metrics['val_epoch_f1_macro']
            self.best_epoch = self.epoch
        
        self.epoch += 1
        # acc = torch.stack([out['acc'] for out in outputs]).mean()
        # self.log("avg_val_acc", acc)


class lstm_data(pl.LightningDataModule):
    def __init__(
            self,
            book_col: atel.data.BookCollection,
            target_col: str,
            ft: fasttext.FastText,
            seq_len: int = 256,
            batch_size: int = 32,
            k: int = 0,
            seed: int = 42,
            num_splits: int = 10,
            problem_type: str = 'multilabel'
        ):
        super().__init__()
        
        self.book_col       = book_col
        self.k              = k
        self.batch_size     = batch_size
        self.target_col     = target_col
        self.ft             = ft
        self.seq_len        = seq_len
        self.seed           = seed
        self.num_splits     = num_splits
        self.problem_type   = problem_type
    
    def setup(self, stage: Optional[str] = None):
        
        book_ids, X = get_fasttext_embeddings(self.book_col, self.ft, self.seq_len)
        target_ids, targets, labels = get_labels(self.book_col, self.target_col)

        mask = torch.isin(torch.from_numpy(target_ids), torch.from_numpy(book_ids))
        if self.problem_type == 'multilabel':
            y = torch.from_numpy(targets[mask]).float()
        elif self.problem_type == 'multiclass':
            y = torch.from_numpy(targets[mask]).long()
        
        full_data = TensorDataset(X, y)
        
        kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=self.seed)
        all_splits = [k for k in kf.split(full_data)]
        
        train_idx, val_idx = all_splits[self.k]
        train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
        
        self.train_data = dataset.Subset(full_data, train_idx)
        self.val_data   = dataset.Subset(full_data, val_idx)

    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

