from transformers import (
    LineByLineTextDataset,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

tokenizer = RobertaTokenizer.from_pretrained(
    "./tokenizers/BPEtokenizer_121022", max_length=128
)

model = RobertaForMaskedLM.from_pretrained("./models/BabyBERTa-GWstart").cuda()

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/Gyldendal_child_books/gyldendalbooks.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./models/RoBERTa-GWstart-gyldendal",
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
    train_dataset=dataset,
)
trainer.train()

trainer.save_model("./models/BabyBERTa-GWstart-gyldendal")
