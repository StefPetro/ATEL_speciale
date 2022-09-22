import gc
import re

import lemmy
import numpy as np
import spacy
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from atel.data import BookCollection
from data_clean import *

## Set seed
set_seed(42)


def evaluator(X, target_col, clf):

    target_ids, targets, labels = get_labels(book_col, target_col)

    mask = np.isin(target_ids, book_ids)
    y = targets[mask]

    print(f"Number of samples: {y.shape[0]}")
    print(f"Number of labels: {y.shape[1]}")
    print(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = MultiOutputClassifier(clf)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # print(classification_report(y_test, y_pred))
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
    print(f"CV score: {np.mean(cross_val_score(model, X_train, y_train, cv=5))}")

    return accuracy_score(y_test, y_pred), np.mean(
        cross_val_score(model, X_train, y_train, cv=5)
    )


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


# clf = MultinomialNB()
# clf = RandomForestClassifier(n_estimators=1000, random_state=42)
clf = RidgeClassifier()
# clf = GaussianNB()

accuracy = []
cv_score = []
for t in target_cols:
    acc, cv = evaluator(X, t, clf)
    accuracy.append(acc)
    cv_score.append(cv)

print(np.mean(accuracy))
print(np.mean(cv_score))
