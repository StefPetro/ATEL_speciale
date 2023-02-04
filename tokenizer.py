from datasets import load_dataset
import os
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path


dataset = load_dataset("./data/Gyldendal_child_books", split="train")

paths = [str(x) for x in Path("./data/Gyldendal_child_books").glob("**/*.txt")]

# RoBERTa uses the BPE Tokenizer
tokenizer = ByteLevelBPETokenizer()
# and train
tokenizer.train(
    files=paths,
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

os.mkdir("./tokenizers/BPEtokenizer_121022")

tokenizer.save_model("./tokenizers/BPEtokenizer_121022")
