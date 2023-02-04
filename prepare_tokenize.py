"""
Tokenizes prepared dataset
"""

from transformers import (
    RobertaTokenizer,
)
from datasets import load_from_disk

import sys
sys.setrecursionlimit(1000000)

data = load_from_disk("linebyline_gw.hf")

tokenizer = RobertaTokenizer.from_pretrained(
    "./tokenizers/BPEtokenizer_121022", max_length=128
)

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

dataset = data.map(tokenize_function, batched=True, num_proc=4)

dataset.save_to_disk("tokenized_gw.hf")