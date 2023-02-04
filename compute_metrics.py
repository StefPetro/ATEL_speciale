from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
    multilabel_stat_scores,
    multilabel_accuracy,
    multilabel_exact_match,
    multilabel_f1_score,
    multilabel_precision,
    multilabel_recall,
    multiclass_stat_scores,
    multilabel_specificity,
    multiclass_specificity
)

import numpy as np

def compute_metrics(preds, targets, multi_label, output_size):
    """Function that compute relevant metrics"""

    if multi_label:
        acc_exact = multilabel_exact_match(preds, targets, num_labels=output_size)
        acc_macro = multilabel_accuracy(preds, targets, num_labels=output_size)
        f1_macro = multilabel_f1_score(preds, targets, num_labels=output_size)
        precision_macro = multilabel_precision(preds, targets, num_labels=output_size)
        recall_macro = multilabel_recall(preds, targets, num_labels=output_size)
        specificity_macro = multilabel_specificity(preds, targets, num_labels=output_size)

        # print(specificity_macro, (tn/(fp+tn)))

        metrics = {
            "acc_exact": acc_exact,
            "acc_micro": np.nan,
            "acc_macro": acc_macro,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "fpr": 1-specificity_macro # fpr = fp/(fp+tn) = 1-tn/(fp+tn) = 1-specificity
        }

    else:
        acc_micro = multiclass_accuracy(
            preds, targets, num_classes=output_size, average="micro"
        )
        acc_macro = multiclass_accuracy(
            preds, targets, num_classes=output_size, average="macro"
        )
        f1_macro = multiclass_f1_score(preds, targets, num_classes=output_size)
        precision_macro = multiclass_precision(preds, targets, num_classes=output_size)
        recall_macro = multiclass_recall(preds, targets, num_classes=output_size)
        specificity_macro = multiclass_specificity(preds, targets, num_classes=output_size)

        metrics = {
            "acc_exact": np.nan,
            "acc_micro": acc_micro,
            "acc_macro": acc_macro,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "fpr": 1-specificity_macro
        }

    return metrics