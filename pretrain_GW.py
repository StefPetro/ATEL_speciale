from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk

import sys
sys.setrecursionlimit(1000000)

dataset = load_from_disk("tokenized_gw.hf")

tokenizer = RobertaTokenizer.from_pretrained(
    "./tokenizers/BPEtokenizer_121022", max_length=128
)

config = RobertaConfig(
    vocab_size=5_000,
    max_position_embeddings=130,
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=8,
    num_hidden_layers=8,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config).cuda()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./models/RoBERTa-GWstart-warmup",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    do_train=True,
    do_eval=False,
    do_predict=False,
    save_steps=10_000,
    save_total_limit=2,
    # warmup_ratio=0.2, # WARMUP
    logging_strategy="steps",
    logging_steps=100,
    logging_dir="roberta_logs",
    report_to="tensorboard",
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train'],
)
trainer.train(resume_from_checkpoint = True)

trainer.save_model("./models/BabyBERTa-GWstart-warmup")
