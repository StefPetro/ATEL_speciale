from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import glob
from tqdm import tqdm
import os

tokenizer = RobertaTokenizer.from_pretrained(
    "./tokenizers/BPEtokenizer_121022", max_length=128
)

all_files = glob.glob("./data/dagw/sektioner/*/*")

for idx, f in enumerate(tqdm(all_files)):
    # check if file is empty:
    if os.stat(f).st_size == 0:
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


config = RobertaConfig(
    vocab_size=5_000,
    max_position_embeddings=130,
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=8,
    num_hidden_layers=8,
    type_vocab_size=1,
)

model = RobertaForMaskedLM.from_pretrained("./models/BabyBERTa_091122").cuda()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./models/RoBERTa_with_warmup_211122",
    overwrite_output_dir=True,
    num_train_epochs=12,
    per_device_train_batch_size=64,
    do_train=True,
    do_eval=False,
    do_predict=False,
    save_steps=10_000,
    save_total_limit=2,
    # warmup_ratio=0.2, # WARMUP
    logging_strategy="steps",
    logging_steps=100,
    logging_dir="roberta_logs_gw",
    report_to="tensorboard",
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()

trainer.save_model("./models/BabyBERTa_131022_GW")