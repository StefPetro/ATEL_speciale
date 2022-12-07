"""
Prepares dataset to be line-by-line
"""
from transformers import (
    RobertaTokenizer,
    LineByLineTextDataset,
)
import glob
from tqdm import tqdm
import os
import pickle
from datasets import load_dataset

import sys
sys.setrecursionlimit(1000000)

tokenizer = RobertaTokenizer.from_pretrained(
    "./tokenizers/BPEtokenizer_121022", max_length=128
)

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)


all_files = glob.glob("./data/dagw/sektioner/*/*")
keep_files = []

for idx, f in enumerate(tqdm(all_files)):
    # check if file is empty:
    if os.path.isdir(f): # If it is just a directory
        continue
    elif f.endswith('.jsonl') or f.endswith('.json'): # if it is a .jsonl file
        continue
    elif f.split('/')[-1] == 'LICENSE': # if it is a license file
        continue
    elif os.stat(f).st_size == 0 or open(f, "r").read().isspace():
        continue
    else:
        keep_files.append(f)

data = load_dataset("text", data_files=keep_files, sample_by='line')

data.save_to_disk("linebyline_gw.hf")

# dataset = data.map(tokenize_function, batched=True)

# # Storing the dataset
# with open('tokenized_gw_linebyline.pkl', 'wb') as f:
#    pickle.dump(dataset, f)
