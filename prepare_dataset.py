from transformers import (
    RobertaTokenizer,
    LineByLineTextDataset,
)
import glob
from tqdm import tqdm
import os
import pickle

import sys
sys.setrecursionlimit(1000000)

tokenizer = RobertaTokenizer.from_pretrained(
    "./tokenizers/BPEtokenizer_121022", max_length=128
)

all_files = glob.glob("./data/dagw/sektioner/*/*")

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

    if idx == 0:
        dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=f,
        block_size=128,
        )
    else:
        d = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=f,
        block_size=128,
        )
        
        dataset += d

# Storing the dataset
with open('tokenized_gw_linebyline.pkl', 'wb') as f:
   pickle.dump(dataset, f)
