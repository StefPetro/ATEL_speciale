import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import KFold
from yaml import CLoader

from atel.data import BookCollection
from compute_metrics import compute_metrics
from data_clean import *

with open("target_info.yaml", "r", encoding="utf-8") as f:
    target_info = yaml.load(f, Loader=CLoader)


# Get each target column
target_cols = list(target_info.keys())

## Initial load and clean
book_col = BookCollection(data_file="./data/book_col_271120.pkl")
book_ids, texts = clean_book_collection_texts(book_col)

# Get the splits
kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_splits = [k for k in kf.split(book_ids)]

scores = pd.DataFrame()
SEMs = pd.DataFrame()
for t in target_cols:

    # Get the target
    target_ids, targets, labels = get_labels(book_col, t)
    mask = np.isin(target_ids, book_ids)
    y = targets[mask]

    multi_label = True if target_info[t]["problem_type"] == "multilabel" else False
    num_labels = target_info[t]["num_labels"]

    metrics = []
    for k, (train, val) in enumerate(all_splits):
        target = y[val]

        # # For LSTM
        # preds_dir = glob.glob(
        #     f"./lightning_logs/{t.replace(' ', '_')}/*/bs_64/CV_{k+1}/*.pt"
        # )

        # For HuggingFace
        preds_dir = glob.glob(f"./huggingface_logs/{t}/*/CV_{k+1}/*.pt")

        assert len(preds_dir) == 1, "There must only be 1 set of predictions"

        preds = torch.load(preds_dir[0], map_location=torch.device("cpu"))

        if multi_label:
            logit_func = nn.Sigmoid()
            preds = logit_func(preds)
        else:
            # Take the Sigmoid of preds
            logit_func = nn.Softmax()
            preds = logit_func(preds)

        metrics.append(
            compute_metrics(preds, torch.Tensor(target), multi_label, num_labels)
        )

        break

    metrics_df = pd.DataFrame(metrics)
    k_scores = metrics_df.mean(axis=0)

    sem = np.std(metrics_df, axis=0) / np.sqrt(len(metrics_df))

    scores[t] = k_scores
    SEMs[t] = sem

print(
    ((scores * 100).round(1)).astype(str)
    + " \pm "
    + ((SEMs * 100).round(1)).astype(str)
)
print(
    (scores.mean(axis=1) * 100).round(1).astype(str)
    + " \pm "
    + (SEMs.mean(axis=1) * 100).round(1).astype(str)
)
