import numpy as np
import pandas as pd
import yaml
from yaml import CLoader
from sklearn.model_selection import KFold

from atel.data import BookCollection
from compute_metrics import compute_metrics
from data_clean import *

SEED = 42
set_seed(SEED)

book_col = BookCollection(data_file="./data/book_col_271120.pkl")
book_ids, texts = clean_book_collection_texts(book_col, lowercase=False)

with open("target_info.yaml", "r", encoding="utf-8") as f:
    target_problems = yaml.load(f, Loader=CLoader)


def largest_multilabel(val_targets, train_targets):
    N, M = val_targets.shape
    preds = np.zeros((N, M))
    for c in range(M):
        counts = np.bincount(train_targets[:, c])
        largest_class = np.argmax(counts)
        preds[:, c].fill(largest_class)
    return preds


def largest_multiclass(val_targets, train_targets):
    N = len(val_targets)
    counts = np.bincount(train_targets)
    largest_class = np.argmax(counts)
    preds = np.ones(N) * largest_class
    return preds


# Get the splits
kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_splits = [k for k in kf.split(book_ids)]

scores = pd.DataFrame()
SEMs = pd.DataFrame()

for target_col, info in target_problems.items():
    target_ids, targets, labels = get_labels(book_col, target_col)
    problem_type = info["problem_type"]
    multi_label = True if problem_type == "multilabel" else False

    num_labels = info["num_labels"]

    mask = np.isin(target_ids, book_ids)
    targets = targets[mask]

    metrics = []
    for k, (train, val) in enumerate(all_splits):
        y = targets[val]
        y_train = targets[train]
        if problem_type == "multilabel":
            preds = largest_multilabel(y, y_train)

            preds = torch.Tensor(preds)
            y = torch.tensor(y)

        elif problem_type == "multiclass":
            preds = largest_multiclass(y, y_train)

            preds = torch.Tensor(preds)
            y = torch.tensor(y)

        metrics.append(compute_metrics(preds, y, multi_label, num_labels))

    metrics_df = pd.DataFrame(metrics)
    k_scores = metrics_df.mean(axis=0)

    sem = np.std(metrics_df, axis=0) / np.sqrt(len(metrics_df))

    scores[target_col] = k_scores
    SEMs[target_col] = sem

print(
    (
        ((scores * 100).round(1)).astype(str)
        + " \pm "
        + ((SEMs * 100).round(1)).astype(str)
    )
)
# print(
#     (scores.mean(axis=1) * 100).round(1).astype(str)
#     + " \pm "
#     + (SEMs.mean(axis=1) * 100).round(1).astype(str)
# )

print(
    (scores * 100).round(1).mean(axis=1).round(1).astype(str)
    + " \pm "
    + (SEMs * 100).round(1).mean(axis=1).round(1).astype(str)
)
