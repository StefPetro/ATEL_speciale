import re
import random
from typing import Tuple
import numpy as np
import pandas as pd
import fasttext
import torch
from torch.nn.utils.rnn import pad_sequence
import atel.data
from sklearn.model_selection import KFold


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed has been set to {seed}")


""" Initial Text Cleaning """


def print_book_sentences(book_col: atel.data.BookCollection, book_id: int):
    print(book_col[book_id].get_fulltext())


def clean_book_text_lowercase(book: atel.data.Book) -> str:
    s = book.get_fulltext()
    s = s.replace("\t", " ").replace("\n", " ")
    s = re.sub("[^[a-zA-Z0-9æøåÆØÅ\s]", " ", s)
    s = re.sub("\s+", " ", s)  # removes trailing whitespaces
    s = s.lower().strip()

    return s


def clean_book_text(book: atel.data.Book) -> str:
    s = book.get_fulltext()
    s = s.replace("\t", " ").replace("\n", " ")
    s = re.sub("[^[a-zA-Z0-9æøåÆØÅ\s.!?-]", " ", s)
    s = re.sub("\s+", " ", s)  # removes trailing whitespaces
    s = s.strip()

    return s


def clean_book_collection_texts(
    book_col: atel.data.BookCollection,
    include_empty_texts: bool = False,
    lowercase: bool = True,
) -> Tuple[list, list]:

    book_ids = []
    texts = []

    for i, book in enumerate(book_col):
        if lowercase:
            s = clean_book_text_lowercase(book)
        else:
            s = clean_book_text(book)

        if not include_empty_texts and s != "":
            texts.append(s)
            book_ids.append(i)

    book_ids = np.array(book_ids)
    return book_ids, texts


""" Getting labels from Text level """


def get_text_level_data(book_col: atel.data.BookCollection) -> list:
    data = []

    for i, book in enumerate(book_col):
        if book.code_dict is None:
            book.code_dict = {"book_id": i}
            data.append(book.code_dict)
            continue
        book.code_dict["book_id"] = i
        data.append(book.code_dict)

    return data


def get_labels(
    book_col: atel.data.BookCollection, target_col: str
) -> Tuple[np.ndarray, np.ndarray, list]:
    list_cols = [  # columns that consists of lists
        "Genre",
        "Attitude",
        "Stavning",
        "Perspektiv",
        "Tekstbånd",
        "Fremstillingsform",
        "Semantisk univers",
        "Stemmer",
        "Forbindere",
        "Interjektioner",
    ]

    data = get_text_level_data(book_col)
    book_df = pd.DataFrame(data)

    if target_col not in book_df.columns:
        raise f"Column {target_col} does not exis in the data."

    ## Replace empty strings with NaN values
    book_df = book_df.replace("", np.NaN)

    ## Explode all columns
    ex_book_df = book_df.copy(deep=True)  # ex = exploded
    for col in list_cols:
        ex_book_df = ex_book_df.explode(col)

    # Replace "Vilde dyr " with "Vilde dyr" (space difference)
    ex_book_df.loc[
        ex_book_df["Semantisk univers"] == "Vilde dyr\xa0", "Semantisk univers"
    ] = "Vilde dyr"

    col_df = ex_book_df[["book_id", target_col]].drop_duplicates()

    one_hot_df = pd.get_dummies(col_df, prefix=target_col).groupby("book_id").sum()

    labels = [c[len(target_col) + 1 :] for c in one_hot_df.columns]

    targets = one_hot_df.values
    target_ids = np.array(one_hot_df.index)

    return target_ids, targets, labels


def get_fasttext_embeddings(
        book_col: atel.data.BookCollection, 
        ft: fasttext.FastText,
        seq_len: int
    
    ) -> torch.Tensor:
    book_ids, texts = clean_book_collection_texts(book_col, lowercase=True)

    el = [torch.Tensor(np.array([ft.get_word_vector(w) for w in t.split(' ')])) for t in texts]
    el_padded = pad_sequence(el, batch_first=True)[:, :seq_len, :]
    
    assert el_padded.shape[1] == seq_len, 'Sequence length is not equal max length'

    return book_ids, el_padded

