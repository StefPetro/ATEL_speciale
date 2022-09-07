import re
import gc
import numpy as np
from nltk.corpus import stopwords
import lemmy
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from atel.data import BookCollection
from data_clean import *

## Initial load and clean
book_col = BookCollection(data_file='./data/book_col_271120.pkl')  
book_ids, texts = clean_book_collection_texts(book_col)

## Lemmetize, remove stopwords, etc.
lemmatizer = lemmy.load("da")

nlp = spacy.load("da_core_news_sm")

documents = []  # all the texts lemmetized
for t in texts:
    doc = nlp(t)
    texts_ = [lemmatizer.lemmatize(token.pos_, token.text)[0] for token in doc]
    documents.append(' '.join(texts_))


## Create bag of words (CountVectorizer)
tfidfvectorizer = TfidfVectorizer(max_features=1500, min_df=3, max_df=0.8, 
                                  stop_words=stopwords.words('danish'))


X = tfidfvectorizer.fit_transform(documents).toarray()

print(X.shape)
print(X)


model = RandomForestClassifier(n_estimators=1000, random_state=42)

