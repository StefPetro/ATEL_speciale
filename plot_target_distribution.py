import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from atel.data import BookCollection
import yaml
from yaml import CLoader
sns.set_style('whitegrid')

# load data
book_col = BookCollection(data_file='./data/book_col_271120.pkl')

# Total number of books
print(f'Number of books: {book_col.num_books}')


data = []

for i, book in enumerate(book_col):
    if book.code_dict is None:
        print(f'Book with index {i} is None')
        continue
    book.code_dict['book_id'] = i 
    data.append(book.code_dict)
    
    
book_df = pd.DataFrame(data)

list_cols = [
    'Genre', 'Attitude', 'Stavning', 'Perspektiv', 'Tekstbånd', 'Fremstillingsform', 
    'Semantisk univers', 'Stemmer', 'Forbindere', 'Interjektioner'
]

## Replace empty strings with NaN values
book_df = book_df.replace('', np.NaN)

## Explode all columns
# Exploded books
ex_book_df = book_df.copy(deep=True)
for col in list_cols:
    ex_book_df = ex_book_df.explode(col)


# Replace "Vilde dyr " with "Vilde dyr" (space difference)
ex_book_df.loc[
        ex_book_df["Semantisk univers"] == "Vilde dyr\xa0", "Semantisk univers"
    ] = "Vilde dyr"


with open('category_groupings.yaml', 'r', encoding='utf-8') as f:
     groupings = yaml.load(f, Loader=CLoader)

for group in groupings['Semantisk univers']:
    ex_book_df["Semantisk univers"] = ex_book_df["Semantisk univers"].replace(
        groupings['Semantisk univers'][group], group
    )


def plot_distribution(category: str, **kwargs):
    
    xlim = kwargs.get('xlim', None)
    
    counts = ex_book_df[['book_id', category]].drop_duplicates().value_counts(category)
    idx = counts.index
    y = counts.values
    
    if category == 'Holistisk vurdering':
        idx, y = [y for y, x in sorted(zip(idx, y))], [x for y, x in sorted(zip(idx, y))]
    
    plt.figure(figsize=(7, 5), dpi=300)
    plt.barh(idx, y)
    plt.title(f'Distribution of labels: {category}', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Labels', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if xlim:
        plt.xlim(0, xlim)
    
    plt.savefig(f'imgs/label_dist/dist_{category}.png', bbox_inches="tight")

    print(f'{category} plot saved!')


all_cats = {
    'Genre':               {'xlim': None},
    'Tekstbånd':           {'xlim': 400},
    'Fremstillingsform':   {'xlim': None},
    'Semantisk univers':   {'xlim': None},
    'Stemmer':             {'xlim': 600},
    'Perspektiv':          {'xlim': 500},
    'Holistisk vurdering': {'xlim': 300}
}

for k, v in all_cats.items():
    plot_distribution(k, **v)

print('Done!')
