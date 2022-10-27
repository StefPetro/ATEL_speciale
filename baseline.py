import lemmy
import numpy as np
import pandas as pd
import spacy
import yaml
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
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
from yaml import CLoader
import torch
import torch.nn as nn


from atel.data import BookCollection
from data_clean import *

## Set seed
set_seed(42)

with open("target_info.yaml", "r", encoding="utf-8") as f:
    target_info = yaml.load(f, Loader=CLoader)


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


def evaluator(book_col, X, target_col, clf):
    assert (
        target_col in target_info.keys()
    )  # checks if targets is part of the actual problem columns

    problem_type = target_info[target_col].get("problem_type")
    num_labels = target_info[target_col].get("num_labels")

    target_ids, targets, labels = get_labels(book_col, target_col)

    mask = np.isin(target_ids, book_ids)
    y = targets[mask]

    if problem_type == "multilabel":
        # Multi label classification
        model = MultiOutputClassifier(clf)
    else:
        y = y.argmax(axis=1)
        model = clf

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_splits = [k for k in kf.split(X)]

    metrics = []
    for k in range(10):
        train_idx, val_idx = all_splits[k]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics.append(compute_metrics(y_pred, y_val, problem_type, num_labels))

    metrics_df = pd.DataFrame(metrics)
    cv_score = metrics_df.mean(axis=0)

    # sem = np.std(acc) / np.sqrt(len(acc))
    return cv_score


## Initial load and clean
book_col = BookCollection(data_file="./data/book_col_271120.pkl")
book_ids, texts = clean_book_collection_texts(book_col)

## Lemmetize, remove stopwords, etc.
lemmatizer = lemmy.load("da")

nlp = spacy.load("da_core_news_sm")

documents = []  # all the texts lemmetized
for t in texts:
    doc = nlp(t)
    texts_ = [lemmatizer.lemmatize(token.pos_, token.text)[0] for token in doc]
    documents.append(" ".join(texts_))


## Create bag of words (CountVectorizer)
tfidfvectorizer = TfidfVectorizer(
    max_features=1500, min_df=3, max_df=0.75, stop_words=stopwords.words("danish")
)


X = tfidfvectorizer.fit_transform(documents).toarray()

target_cols = [
    "Genre",
    "Perspektiv",
    "Tekstbånd",
    "Fremstillingsform",
    "Semantisk univers",
    "Holistisk vurdering",
    "Stemmer",
]

classifiers = [
    RidgeClassifier(),
    GaussianNB(),
    RandomForestClassifier(n_estimators=1000, random_state=42),
]
# clf = MultinomialNB()
# clf = RandomForestClassifier(n_estimators=1000, random_state=42)
# clf = RidgeClassifier()
# clf = GaussianNB()

for clf in classifiers:
    scores = pd.DataFrame()
    for t in target_cols:
        cv = evaluator(book_col, X, t, clf)
        scores[t] = cv

    print(f"Classifier: {clf}")
    print(scores)
    print(scores.mean(axis=1))
