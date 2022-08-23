import numpy as np
import pandas as pd
import dill
import json
import string
import os
import re
from collections import OrderedDict



########################################################################################################################
# Define data structures
########################################################################################################################

class Sentence(object):

    def __init__(self, start=None, all_code_dicts=None, text=None, is_balloon=False):
        self.start = start
        self.all_code_dicts = all_code_dicts
        self.text = text
        self.is_balloon = is_balloon

        if self.all_code_dicts:
            # for now: use first entry
            author_key = list(self.all_code_dicts)[0]
            self.code_dict = self.all_code_dicts[author_key]
        else:
            self.code_dict = None

        # Not coded?
        self.not_coded = False
        if self.all_code_dicts:
            for key, code_dict in self.all_code_dicts.items():
                if code_dict and 'Ikke kodet' in code_dict:
                    self.not_coded = True

    def __repr__(self):
        if type(self.text) is str and len(self.text) > 0:
            return '"' + self.text + '"'
        else:
            return '""'



class Page(object):

    def __init__(self, id, child_text, adult_text, adult_text_mod, balloons, image=None):
        self.child_text = child_text
        self.adult_text = adult_text
        self.adult_text_mod = adult_text_mod
        self.balloons = balloons
        self.image = image

        # use modified adult text whenever it exists
        if len(adult_text_mod) > 0:
            self.text = self.adult_text_mod
        else:
            self.text = self.adult_text

        self.id = id
        self.sentences = []
        self.balloon_sentences = []
        self.cutoffs = []

    def add_sentence(self, sentence):

        if sentence.is_balloon:
            if len(self.balloons) >= sentence.start:
                sentence.text = self.balloons[sentence.start-1]
            else:
                sentence.text = ''

            self.balloon_sentences.append(sentence)

        else:

            self.sentences.append(sentence)
            self.cutoffs.append(sentence.start)

            # TODO: Better way to do this?
            if len(self.cutoffs) == 1:
                self.sentences[0].text = self.text
            else:
                text_parts = [self.text[i:j] for i,j in zip(self.cutoffs, self.cutoffs[1:]+[None])]

                for sentence, text_part in zip(self.sentences, text_parts):
                    sentence.text = text_part.strip()

    def __repr__(self):
        return self.text


    def __iter__(self):
        return iter(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]




class Book(object):

    def __init__(self, id, all_code_dicts=None, elev=None, skole=None, klassetrin="", startaar="", title='', date=None, import_id=None):
        self.id = id
        self.all_code_dicts = all_code_dicts
        self.pages = []
        self.num_pages = len(self.pages)
        self.import_id = import_id
        self.title = title
        self.elev = elev
        self.skole = skole
        self.klassetrin = klassetrin
        self.startaar = startaar
        self.date = date

        if self.all_code_dicts:
            # for now: use first entry
            author_key = list(self.all_code_dicts)[0]
            self.code_dict = self.all_code_dicts[author_key]
        else:
            self.code_dict = None

    def add_page(self, page):
        self.pages.append(page)
        self.num_pages += 1

    def __repr__(self):
        print('\nBook ID:\t{}'.format(self.id))
        print('Elev: {}'.format(self.elev))
        print('School: {}'.format(self.skole))
        print('Klassetrin: {}'.format(self.klassetrin))
        print('Num pages:\t{}'.format(self.num_pages))

        if self.num_pages > 0:
            print('')
            for page in self.pages:
                print('Page {}: {}'.format(page.id, page.child_text))
                print('Page {}: {}'.format(page.id, page.text))
                print('')

        return ''#'' '.join([page.text for page in self.pages])

    def get_fulltext(self, join=True):
        full_text_ = [page.text for page in self.pages]

        if join:
            return ' '.join(full_text_)
        else:
            return full_text_

    def get_fulltext_child(self, join=True):
        full_text_ = [page.child_text for page in self.pages]

        if join:
            return ' '.join(full_text_)
        else:
            return full_text_

    def get_page_by_id(self, id):
        for page in self.pages:
            if page.id == id:
                return page

        return None

    def get_num_words(self):
        return len([i.strip(string.punctuation) for i in self.fulltext.split()])

    def __iter__(self):
        return iter(self.pages)

    def __getitem__(self, item):
        return self.pages[item]


    fulltext = property(fget=get_fulltext)
    fulltext_child = property(fget=get_fulltext_child)
    num_words = property(fget=get_num_words)


class BookCollection(object):

    def __init__(self, name=None, data_file=None):

        self.name = name
        self.books = []

        if data_file:
            self.load_from_file(data_file)


    def load_from_file(self, data_file):
        f = open(data_file, 'rb')
        book_col_obj = dill.load(f)
        self.books = book_col_obj.books

        print('Loaded from disk: %s' % data_file)
        return self


    def save_to_disk(self, data_file):
        f = open(data_file, 'wb')
        dill.dump(self, f)
        f.close()

        print('Saved to disk: %s' % data_file)
        return self

    def add_book(self, book):
        self.books.append(book)
        return self

    def export_all_text(self, include_balloons=False, remove_empty=True, use_adult_text_mod=True):

        if include_balloons:
            raise NotImplementedError("This feature (include_balloons=True) is not implemented.")

        child_text = []
        adult_text = []

        for book in self.books:

            page_child = []
            page_adult = []



            for page in book.pages:

                if remove_empty is False or len(page.child_text) > 0:
                    page_child.append(page.child_text)

                a_text = page.adult_text
                if use_adult_text_mod and len(page.adult_text_mod):
                    a_text = page.adult_text_mod

                if remove_empty is False or len(a_text) > 0:
                    page_adult.append(a_text)

            child_text.append(page_child)
            adult_text.append(page_adult)

        return child_text, adult_text

    def __iter__(self):
        return iter(self.books)

    def __getitem__(self, item):
        return self.books[item]






    @property
    def num_books(self):
        return len(self.books)

    @property
    def num_pages(self):
        return sum(book.num_pages for book in self.books)




########################################################################################################################
# Clean text
########################################################################################################################
def clean_text(my_text):

    # remove everything else than letters, numbers, fullstops
    my_text = re.sub(r'[^ a-zA-ZæøåÆØÅ0-9.,?]', '', my_text)

    # replace several fullstop with a single fullstop
    my_text = re.sub(r'\.+', ".", my_text)

    return my_text


########################################################################################################################
# Read all books
########################################################################################################################


def read_books_from_file(data_dir, imports, filename, book_id_offset=False, verbose=False):

    book_lines = []
    for imp in imports:
        book_fullpath = os.path.join(data_dir, imp, filename)

        print('\tReading from: %s...' % book_fullpath, end='')

        book_file = open(book_fullpath, 'r',  encoding='utf-8')
        import_lines = book_file.readlines()

        # skip first line
        import_lines.pop(0)

        book_lines.extend(import_lines)
        book_file.close()

        print('%d lines.' % len(import_lines))

    book_dict = OrderedDict()

    num = []

    for idx, line in enumerate(book_lines):

        components = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', line)

#        if idx == 737:


        # extract page id from each line
        import_id = int(components[0])

        # elev, skole, klasse, startaar
        elev_id = components[1]
        skole = components[3]
        klassetrin = components[5]
        startaar = components[6]
        date = components[9]


        # extract page id from each line
        code_book_id = int(components[7])

        # extract title
        book_title = components[8].lstrip('"').rstrip('"')


        # correct for offset?
        if book_id_offset:
            code_book_id = code_book_id - 513

        # remove first part until 9th comma
#        for i in range(10):
#            comma_idx = line.index(',', ) + 1
#            line = line[comma_idx:]

        # keep code part
        line = components[10]

        line_dict = {}
        if line.count('"') > 0:
            # locate first and last quotation mark
            begin_idx = line.index('"') + 1
            end_idx = line.rindex('"')

            # extract json string
            json_str = line[begin_idx:end_idx]

            # parse json string
            json_acceptable_string = json_str.replace('""', '"')
            line_dict = json.loads(json_acceptable_string)

            num.append(len(line_dict))


            # clean dict: remove list of empty strings
            for author, code_dict in line_dict.items():
                if code_dict and len(code_dict) > 0:
                    line_dict[author] = {key: val for key, val in code_dict.items() if not (val and type(val) is list and len(''.join(val)) == 0)}

        else:
            if verbose:
                print('No JSON information in line %d: %s' % (idx, line))

        # store
        book_dict[code_book_id] = Book(id=code_book_id, all_code_dicts=line_dict, import_id=import_id, title=book_title, elev=elev_id, skole=skole, klassetrin=klassetrin, startaar=startaar, date=date)

    return book_dict

#######################################################################################################################
# Load sidexport
#######################################################################################################################

def replace_float_with_empty_string(my_list):
    new_list = []
    for element in my_list:
        if type(element) is float:
            new_list.append('')
        else:
            new_list.append(element)
    return new_list


def read_pages_from_file(data_dir, imports, filename, book_id_offset=False):

    book_id = []
    page_id = []
    child_text = []
    adult_text = []
    adult_mod_text = []
    speech_balloons = []


    for imp in imports:
        sentence_filename = os.path.join(data_dir, imp, filename)

        print('\tReading from: %s...' % sentence_filename, end='')

        sentence_data = pd.read_csv(sentence_filename)

        book_id.extend(list(sentence_data['bogid'].values))
        page_id.extend(list(sentence_data['sideid'].values))
        child_text.extend(list(sentence_data['elev'].values))
        adult_text.extend(list(sentence_data['voksen'].values))
        adult_mod_text.extend(list(sentence_data['voksenred'].values))

        # clean missing data
        child_text = replace_float_with_empty_string(child_text)
        adult_text = replace_float_with_empty_string(adult_text)
        adult_mod_text = replace_float_with_empty_string(adult_mod_text)

        # handle speech balloon if present
        balloon_data = list(sentence_data['taleboble'])

        # extract text from balloons
        import_balloons = []
        for balloon in balloon_data:
            page_balloons = []
            if type(balloon) is str and balloon != '[]':

                balloon_dicts = json.loads(balloon)
                for balloon_dict in balloon_dicts:

                    if 'Text' in balloon_dict:
                        page_balloons.append(balloon_dict['Text'])
                    else:
                        page_balloons.append(None)

            import_balloons.append(page_balloons)
        speech_balloons.extend(import_balloons)

        print('')


    # correct for offset
    if book_id_offset:
        book_id = np.array(book_id) - 513


    # construct map from page ID to book ID
    page2book = {pid: bid for pid, bid in zip(page_id, book_id)}

    # make sure we have child text and one of (adult, adult_mod)
    entries = list(zip(book_id, child_text, adult_text, adult_mod_text, page_id, speech_balloons))

    #entries = [(bid, child, adult, adult_mod, pid, balloons) for bid, child, adult, adult_mod, pid, balloons in entries if
    #           len(child) > 0 and (len(adult) > 0 or len(adult_mod) > 0)]

    # sort
    entries = sorted(entries, key=lambda x: x[3])


    # return
    return entries, page2book




########################################################################################################################
# Sort by page id  and split
########################################################################################################################



def match_book_and_pages(book_dict, entries):

    unmatched_pages = []

    for bid, child, adult, adult_mod, pid, balloons in entries:

        # add to book
        if bid in book_dict:

            page = book_dict[bid].get_page_by_id(pid)

            # check if we have that specific page already
            if page is None:
                book_dict[bid].add_page(Page(id=pid, child_text=child, adult_text=adult, adult_text_mod=adult_mod, balloons=balloons))
        else:
            print('Did not find corresponding book for page id = {}'.format(pid))
            unmatched_pages.append((pid, child, adult, adult_mod, balloons))

    return book_dict, unmatched_pages



#######################################################################################################################
# Load codes
#######################################################################################################################




def read_period_codes_from_file(data_dir, imports, filename, book_dict):
    period_lines = []


    for imp in imports:
        period_filename = os.path.join(data_dir, imp, filename)

        print('\tReading from: %s/%s...' % (data_dir, imp), end='')

        f = open(period_filename, 'r')
        lines = f.readlines()

        print('%d lines.' % len(lines))

        # skip first line
        lines.pop(0)

        period_lines.extend(lines)
        f.close()


    code_dict = OrderedDict()


    for idx, line in enumerate(period_lines):

        # extract page id from each line
        code_page_id = int(line.split(',')[0])
        start_id = int(line.split(',')[1])
        is_balloon = int(line.split(',')[2])
        bid = int(line.split(',')[-1])

        if not bid in book_dict:
            import ipdb
            ipdb.set_trace()
            print('Found page with page ID = {}, but did not find corresponding book id. Skipping'.format(code_page_id))
            continue

        book = book_dict[bid]


        # # do we have a corresponding book?
        # if code_page_id in page2book:
        #     bid = page2book[code_page_id]
        #
        #     if bid in book_dict:
        #         book = book_dict[bid]
        #     else:
        #         print('Found page with page ID = {} and book ID = {}, but dit not found corresponding book. Skipping...'.format(code_page_id, bid))
        #         continue
        # else:
        #     print('Found page with page ID = {}, but did not find corresponding book id. Skipping'.format(code_page_id))
        #     continue

        if line.count('"') > 0:

            # locate first and last quotation mark
            begin_idx = line.index('"')+1
            end_idx = line.rindex('"')

            # extract json string
            json_str = line[begin_idx:end_idx]

            # parse json string
            json_acceptable_string = json_str.replace('""', '"')
            line_dict = json.loads(json_acceptable_string)

            # clean dict: remove fields that only contains two citation marks
            for author, code_dict in line_dict.items():
                if code_dict and len(code_dict) > 0:
                    line_dict[author] = {key: val for key, val in code_dict.items() if val != '""'}

            # clean dict: remove empty strings
            for author, code_dict in line_dict.items():
                if code_dict and len(code_dict) > 0:
                    line_dict[author] = {key: val for key, val in code_dict.items() if not (val and type(val) is str and len(''.join(val)) == 0)}

            # clean dict: remove fields equal to None
            for author, code_dict in line_dict.items():
                if code_dict and len(code_dict) > 0:
                    line_dict[author] = {key: val for key, val in code_dict.items() if val is not None}

            # clean dict: remove list of empty strings
            for author, code_dict in line_dict.items():
                if code_dict and len(code_dict) > 0:
                    line_dict[author] = {key: val for key, val in code_dict.items() if not (val and type(val) is list and len(''.join(val)) == 0)}



        else:
            line_dict = {}

        # Get page
        page = book.get_page_by_id(code_page_id)
        if page:
            page.add_sentence(Sentence(start_id, all_code_dicts=line_dict, is_balloon=is_balloon))
        else:
            print('Did not find page with id {}. Skipping'.format(code_page_id))

    return book_dict



########################################################################################################################
# Construct page id to page map
########################################################################################################################

def contruct_page_id_to_page_map(book_dict):
    id2page = {}

    for bid, book in book_dict.items():
        for page in book.pages:
            if page.id in id2page:
                raise ValueError('Conflict! Page ID already exists')
            else:
                id2page[page.id] = page

    return id2page


##################################################################
# filter - remove all books containing un-coded sentences
##################################################################

def filter_books(book_dict):
    book_filtered = {}

    for bid, book in book_dict.items():
        not_coded = False
        for page in book.pages:
            for sentence in page.sentences:
                if sentence.not_coded == True:
                    not_coded = True
                else:
                    for author, code_dict in sentence.all_code_dicts.items():
                        if code_dict and len(code_dict) > 0 and 'Syntaksfejl' in code_dict:
                            not_coded = True
                            # print('Syntax error in book %d page %d, skipping book.' % (bid, pid))
                            break

        if not_coded == False:
            book_filtered[bid] = book

    return book_filtered

########################################################################################################################
# Inspect data
########################################################################################################################

def inspect_data(book_filtered, keep_running):
  for i, (bid, book) in enumerate(book_filtered.items()):

        print(80*'-')
        print('Book #%d: %s' % (bid, book.title))
        print(80 * '-')

        for j, page in enumerate(book.pages):

            print('Page %2d (%4d): %s' % (j+1, page.id, page.text.strip()))

            for k, sentence in enumerate(page.sentences):

                if sentence.not_coded or len(sentence.all_code_dicts) == 0:
                    coded_str = " (Not coded, balloon=%d)" % sentence.is_balloon
                else:
                    coded_str = " (start=%d, %2d codes, balloon=%d)" % (sentence.start, len(next(iter(sentence.all_code_dicts.values()))), sentence.is_balloon)

                print('\tSentence %2d: %s' % (k+1, sentence.text.strip() + coded_str))

                if print_period_codes:
                    for author, author_dict in sentence.all_code_dicts.items():
                        print('\t\tAuthor id #%2d' % int(author))
                        for key, code in author_dict.items():
                            print('\t\t\t%-30s: ' % key, code)

            print('')
            for k, sentence in enumerate(page.balloon_sentences):

                if sentence.not_coded or len(sentence.all_code_dicts) == 0:
                    coded_str = " (Not coded, balloon=%d)" % sentence.is_balloon
                else:
                    coded_str = " (start=%d, %2d codes, balloon=%d)" % (sentence.start, len(next(iter(sentence.all_code_dicts.values()))), sentence.is_balloon)

                print('\tBalloon sentence %2d: %s' % (k+1, sentence.text.strip() + coded_str))

                if print_period_codes:
                    for author, author_dict in sentence.all_code_dicts.items():
                        print('\t\tAuthor id #%2d' % int(author))
                        for key, code in author_dict.items():
                            print('\t\t\t%-30s: ' % key, code)

            print('')

            inp = input()
            if len(inp) > 0 and inp[0] == 'q':
                keep_running = False
                break

        if not keep_running:
            break

        print('\n')




def parse_DPU_data(data_dir, book_col_file=None, output_dir='', verbose=False, imports=['import2', 'import3']):

    book_filename = 'bogexport.csv'
    page_filename = 'sideexport.csv'
    period_filename = 'periodeexport.csv'

    if book_col_file is None:
        book_col_file = './book_col.pkl'

    book_col_file = output_dir + book_col_file

    print(80 * '-')
    print('Importing data...')
    print(80 * '-')
    print('')

    print('Reading books...')
    book_dict = read_books_from_file(data_dir, imports, book_filename)
    print('\tDone.\n')

    print('Reading pages...')
    entries, page2book = read_pages_from_file(data_dir, imports, page_filename)
    print('\tDone.\n')

    print('Matching pages and books...')
    book_dict, unmatched_pages = match_book_and_pages(book_dict, entries)
    print('\tDone.\n')

    print('Load and matching period codes...')
    book_dict = read_period_codes_from_file(data_dir, imports, period_filename, book_dict)
    print('\tDone.\n')

    print('Construct page id to page map...')
    id2page = contruct_page_id_to_page_map(book_dict)
    print('\tDone.\n')
    print('\n')

    print('Constructing book collections...')
    book_collection = BookCollection(name="Data: %s" % data_dir)
    for bid, book in book_dict.items():
        book_collection.add_book(book)

    print('\tNum books: %d' % book_collection.num_books)
    print('\tNum pages: %d' % book_collection.num_pages)
    print('')


    book_collection.save_to_disk(book_col_file)

    return book_collection




