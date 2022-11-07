import glob

import torch
from sklearn.model_selection import KFold
from data_clean import *
from atel.data import BookCollection
import torch.nn as nn
import yaml
from yaml import CLoader
import numpy as np
import pandas as pd

from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multilabel_accuracy,
    multilabel_exact_match,
    multilabel_f1_score,
)

with open("target_info.yaml", "r", encoding="utf-8") as f:
    target_info = yaml.load(f, Loader=CLoader)


def compute_metrics(preds, targets, multi_label, output_size):
    """Function that compute relevant metrics"""

    if multi_label:
        # Take the Sigmoid of preds
        logit_func = nn.Sigmoid()
        preds = logit_func(preds)

        acc_exact = multilabel_exact_match(preds, targets, num_labels=output_size)
        acc_macro = multilabel_accuracy(preds, targets, num_labels=output_size)
        f1_macro = multilabel_f1_score(preds, targets, num_labels=output_size)

        metrics = {
            "acc_exact": acc_exact,
            "acc_micro": np.nan,
            "acc_macro": acc_macro,
            "f1_macro": f1_macro,
        }

    else:
        # Take the Sigmoid of preds
        logit_func = nn.Softmax()
        preds = logit_func(preds)

        acc_micro = multiclass_accuracy(
            preds, targets, num_classes=output_size, average="micro"
        )
        acc_macro = multiclass_accuracy(
            preds, targets, num_classes=output_size, average="macro"
        )
        f1_macro = multiclass_f1_score(preds, targets, num_classes=output_size)

        metrics = {
            "acc_exact": np.nan,
            "acc_micro": acc_micro,
            "acc_macro": acc_macro,
            "f1_macro": f1_macro,
        }

    return metrics


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

    if t != "Semantisk univers":
        continue

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

        metrics.append(
            compute_metrics(preds, torch.Tensor(target), multi_label, num_labels)
        )

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
