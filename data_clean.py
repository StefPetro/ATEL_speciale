import re
import atel
from atel.data import BookCollection

def print_book_sentences(book_col: atel.data.BookCollection, book_id: int):
    print(book_col[book_id].get_fulltext())


def clean_book_text(book: atel.data.Book):
    s = book.get_fulltext() 
    s = s.replace('\t', ' ').replace('\n', ' ')
    s = re.sub('[^[a-zA-Z0-9æøåÆØÅ\s]', ' ', s)
    s = re.sub('\s+', ' ', s)  # removes trailing whitespaces
    s = s.lower().strip()
    
    return s


def clean_book_collection_texts(book_col: atel.data.BookCollection, include_empty_texts: bool=False):
    book_ids = []
    texts    = []
    
    for i, book in enumerate(book_col):
        s = clean_book_text(book)
        
        if not include_empty_texts and s != '':
            texts.append(s)
            book_ids.append(i)
    
    return book_ids, texts

