from atel.data import BookCollection

# load all books
book_col = BookCollection(data_file='./book_col_271120.pkl')

# number of books
print(f'Number of books: {book_col.num_books}')

# get 20th book and print summary
book = book_col[20-1]
print('Printing summary of 11th book')
print(book)

# each book is structured in a set of pages
print(f'Book has {len(book.pages)} pages')

# each page has child and teacher/adult texts
print(f'Child text for first page: {book.pages[0].child_text}')
print(f'Teacher text for first page: {book.pages[1].adult_text}')

# full (teacher) text for a given book can be extracted as list conviniently
print('All teacher text for book:\n' , book.get_fulltext(False))

# each page is a also structured as a set of sentences
page = book.pages[0]
print(f'Page contains {len(page.sentences)} sentences:')
for i, sentence in enumerate(page.sentences):
    print(f'{i}\t{sentence}')
print('\n')

# each sentence is labelled
sentence = page.sentences[0]
print(f'Labels for sentence: {sentence.text}\n')
for key, value in sentence.code_dict.items():
    print(f'\t{key:25s}\t{value}')
print('\n')

# each book also contains labels at book-level
print(f'Labels for entire book')
for key, value in book.code_dict.items():
    print(f'\t{key:25s}\t{value}')

