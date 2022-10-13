import numpy as np
import pandas as pd
from data_clean import *
from atel.data import BookCollection

SEED = 42
set_seed(SEED)

book_col = BookCollection(data_file="./data/book_col_271120.pkl")
book_ids, texts = clean_book_collection_texts(book_col, lowercase=False)

with open('target_problem_type.yaml', 'r', encoding='utf-8') as f:
    target_problems = yaml.load(f, Loader=CLoader)


def random_multilabel(targets, num_repeats: int = 10_000):
    N, M = targets.shape
    p = np.round(targets.sum(axis=0)/targets.sum(), 4)
    random_guess = np.random.binomial(1, p, (num_repeats, N, M))

    pred_check = np.all(random_guess==targets, axis=2)
    avg = pred_check.mean()
    return avg
    

def random_multiclass(targets):
    # get the probabilities of each class in the dataset
    unique, counts = np.unique(targets, return_counts=True)
    p = np.round(counts/counts.sum(), 4)
    n = np.argmax(p)
    return p[n]


for target_col, problem_type in target_problems.items():
    
    target_ids, targets, labels = get_labels(book_col, target_col)
        
    mask = np.isin(target_ids, book_ids)
    targets = targets[mask]

    if problem_type == 'multilabel':
        avg = random_multilabel(targets)
        print(target_col, avg)
    
    elif problem_type == 'multiclass':
        p = random_multiclass(targets)
        print(target_col, p)
