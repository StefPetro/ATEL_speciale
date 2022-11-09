from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multilabel_accuracy,
    multilabel_exact_match,
    multilabel_f1_score,
)

import numpy as np

def compute_metrics(preds, targets, multi_label, output_size):
    """Function that compute relevant metrics"""

    if multi_label:
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