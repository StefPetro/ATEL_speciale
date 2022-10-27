import numpy as np
import pandas as pd
from data_clean import *
from atel.data import BookCollection
import yaml
from yaml import CLoader
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
    multilabel_accuracy,
    multilabel_auroc,
    multilabel_exact_match,
    multilabel_f1_score,
    multilabel_precision,
    multilabel_recall,
)


SEED = 42
set_seed(SEED)

book_col = BookCollection(data_file="./data/book_col_271120.pkl")
book_ids, texts = clean_book_collection_texts(book_col, lowercase=False)

with open("target_info.yaml", "r", encoding="utf-8") as f:
    target_problems = yaml.load(f, Loader=CLoader)


def compute_metrics(preds, labels, problem_type, num_labels):

    preds = torch.Tensor(preds)
    labels = torch.tensor(labels)

    if problem_type == "multilabel":
        acc_exact = multilabel_exact_match(preds, labels, num_labels=num_labels)
        acc_macro = multilabel_accuracy(preds, labels, num_labels=num_labels)

        # How are they calculated?:
        # The metrics are calculated for each label.
        # So if there is 4 labels, then 4 recalls are calculated.
        # These 4 values are then averaged, which is the end score that is logged.
        # The default average applied is 'macro'
        # precision_macro = multilabel_precision(preds, labels, num_labels=NUM_LABELS)
        # recall_macro = multilabel_recall(preds, labels, num_labels=NUM_LABELS)
        f1_macro = multilabel_f1_score(preds, labels, num_labels=num_labels)

        # AUROC score of 1 is a perfect score
        # AUROC score of 0.5 corresponds to random guessing.
        # auroc_macro = multilabel_auroc(
        #     preds, labels, num_labels=num_labels, average="macro", thresholds=None
        # )

        metrics = {
            "accuracy_exact/micro": acc_exact,
            "accuracy_macro": acc_macro,
            # 'precision_macro': precision_macro,
            # 'recall_macro':    recall_macro,
            "f1_macro": f1_macro,
            # "AUROC_macro": auroc_macro,
        }
    else:

        acc_micro = multiclass_accuracy(
            preds, labels, num_classes=num_labels, average="micro"
        )
        acc_macro = multiclass_accuracy(
            preds, labels, num_classes=num_labels, average="macro"
        )
        # precision_macro = multiclass_precision(preds, labels, num_classes=NUM_LABELS)
        # recall_macro = multiclass_recall(preds, labels, num_classes=NUM_LABELS)
        f1_macro = multiclass_f1_score(preds, labels, num_classes=num_labels)

        # auroc_macro = multiclass_auroc(
        #     preds, labels, num_classes=num_labels, average="macro", thresholds=None
        # )

        metrics = {
            "accuracy_exact/micro": acc_micro,
            "accuracy_macro": acc_macro,
            # 'precision_macro': precision_macro,
            # 'recall_macro':    recall_macro,
            "f1_macro": f1_macro,
            # "AUROC_macro": auroc_macro,
        }

    return metrics


def largest_multilabel(targets):
    N, M = targets.shape
    preds = np.zeros((N, M))
    for c in range(M):
        counts = np.bincount(targets[:, c])
        largest_class = np.argmax(counts)
        preds[:, c].fill(largest_class)
    return preds


def largest_multiclass(targets):
    N, M = targets.shape
    preds = np.zeros((N, M))

    largest_class = targets.sum(axis=0).argmax()
    preds[:, largest_class].fill(1)
    return preds


scores = pd.DataFrame()
for target_col, info in target_problems.items():
    target_ids, targets, labels = get_labels(book_col, target_col)
    problem_type = info["problem_type"]
    num_labels = info["num_labels"]

    mask = np.isin(target_ids, book_ids)
    targets = targets[mask]

    if problem_type == "multilabel":
        preds = largest_multilabel(targets)
        metrics = compute_metrics(preds, targets, problem_type, num_labels)
        scores[target_col] = pd.Series(metrics, name=target_col, dtype="float")

    elif problem_type == "multiclass":
        preds = largest_multiclass(targets)
        metrics = compute_metrics(
            preds.argmax(axis=1), targets.argmax(axis=1), problem_type, num_labels
        )
        scores[target_col] = pd.Series(metrics, name=target_col, dtype="float")

print(scores)
print(scores.mean(axis=1))
