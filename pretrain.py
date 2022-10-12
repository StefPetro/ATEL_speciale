from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


config = RobertaConfig(
    vocab_size=5_000,
    max_position_embeddings=514,
    num_attention_heads=6,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizer.from_pretrained("/BPETtokenizer_121022", max_length=512)

model = RobertaForMaskedLM(config=config).cuda()

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../data/Gyldendal_child_books/gyldendalbooks.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="/MiniBERTa_121022",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()

trainer.save_model("/MiniBERTa_121022")
