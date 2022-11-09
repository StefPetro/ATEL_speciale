import lemmy
import numpy as np
import pandas as pd
import spacy
import torch
import yaml
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from yaml import CLoader

from atel.data import BookCollection
from compute_metrics import compute_metrics
from data_clean import *

## Set seed
set_seed(42)

with open("target_info.yaml", "r", encoding="utf-8") as f:
    target_info = yaml.load(f, Loader=CLoader)


def evaluator(book_col, X, target_col, clf):
    assert (
        target_col in target_info.keys()
    )  # checks if targets is part of the actual problem columns

    problem_type = target_info[target_col].get("problem_type")
    multi_label = True if problem_type == 'multilabel' else False
    num_labels = target_info[target_col].get("num_labels")

    target_ids, targets, labels = get_labels(book_col, target_col)

    mask = np.isin(target_ids, book_ids)
    y = targets[mask]

    if multi_label:
        # Multi label classification
        model = MultiOutputClassifier(clf)
    else:
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

        y_pred = torch.Tensor(y_pred)
        y_val = torch.tensor(y_val)

        metrics.append(compute_metrics(y_pred, y_val, multi_label, num_labels))

    metrics_df = pd.DataFrame(metrics)
    cv_score = metrics_df.mean(axis=0)

    sem = np.std(metrics_df, axis=0) / np.sqrt(len(metrics_df))
    return cv_score, sem


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

for clf in classifiers:
    scores = pd.DataFrame()
    SEMs = pd.DataFrame()
    for t in target_cols:
        cv, sem = evaluator(book_col, X, t, clf)
        scores[t] = cv
        SEMs[t] = sem

    print(f"Classifier: {clf}")
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

    # print(SEMs)
