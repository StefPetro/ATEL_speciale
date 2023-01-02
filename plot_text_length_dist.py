from data_clean import *
from atel.data import BookCollection
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

book_ids, texts = clean_book_collection_texts(book_col, False, False)

lns = []
for t in texts:
    lns.append(len(t.split(' ')))

plt.figure(figsize=(7, 5), dpi=300)
plt.hist(lns, bins='auto')
plt.axvline(np.percentile(lns, 95), color='k', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()

plt.text(np.percentile(lns, 95)*1.1, max_ylim*0.9, f'95% percentile: {np.percentile(lns, 95)}')
plt.title('Distribution of number of words in texts', fontsize=16)
plt.xlabel('Number of words', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')

plt.savefig(f'imgs/text_length_dist/text_length_dist.png', bbox_inches="tight")
# plt.show()

print(np.min(lns), np.max(lns))
print(np.percentile(lns, 95))
