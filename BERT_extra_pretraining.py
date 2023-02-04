from transformers import (
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForMaskedLM,
    AutoTokenizer
)

tokenizer = AutoTokenizer.from_pretrained('Maltehb/danish-bert-botxo')

model = AutoModelForMaskedLM.from_pretrained('Maltehb/danish-bert-botxo').cuda()

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/Gyldendal_child_books/gyldendalbooks.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


logging_name = f'bertMLM_logs'\
               +f'/BERT_extra_pretraining'

training_args = TrainingArguments(
    output_dir="bertMLMsaves/BERT_mlm_gyldendal",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    do_train=True,
    do_eval=False,
    do_predict=False,
    save_steps=10_000,
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=100,
    logging_dir=logging_name,
    report_to="tensorboard",
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train(resume_from_checkpoint = True)

trainer.save_model("./models/BERT_mlm_gyldendal")