import numpy as np
import pandas as pd
import yaml
from yaml import CLoader

from atel.data import BookCollection
from compute_metrics import compute_metrics
from data_clean import *

SEED = 42
set_seed(SEED)

book_col = BookCollection(data_file="./data/book_col_271120.pkl")
book_ids, texts = clean_book_collection_texts(book_col, lowercase=False)

with open("target_info.yaml", "r", encoding="utf-8") as f:
    target_problems = yaml.load(f, Loader=CLoader)


def largest_multilabel(targets):
    N, M = targets.shape
    preds = np.zeros((N, M))
    for c in range(M):
        counts = np.bincount(targets[:, c])
        largest_class = np.argmax(counts)
        preds[:, c].fill(largest_class)
    return preds


def largest_multiclass(targets):
    N = len(targets)
    counts = np.bincount(targets)
    largest_class = np.argmax(counts)
    preds = np.ones(N) * largest_class
    return preds


scores = pd.DataFrame()
for target_col, info in target_problems.items():
    target_ids, targets, labels = get_labels(book_col, target_col)
    problem_type = info["problem_type"]
    multi_label = True if problem_type == 'multilabel' else False

    num_labels = info["num_labels"]

    mask = np.isin(target_ids, book_ids)
    targets = targets[mask]

    if problem_type == "multilabel":
        preds = largest_multilabel(targets)

        preds = torch.Tensor(preds)
        targets = torch.tensor(targets)

        metrics = compute_metrics(preds, targets, multi_label, num_labels)
        scores[target_col] = pd.Series(metrics, name=target_col, dtype="float")

    elif problem_type == "multiclass":
        preds = largest_multiclass(targets)
        
        preds = torch.Tensor(preds)
        targets = torch.tensor(targets)

        metrics = compute_metrics(preds, targets, multi_label, num_labels)
        scores[target_col] = pd.Series(metrics, name=target_col, dtype="float")

print(((scores * 100).round(1)).astype(str))
print((scores.mean(axis=1) * 100).round(1).astype(str))
