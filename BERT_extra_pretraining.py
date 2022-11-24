from transformers import (
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForMaskedLM,
    AutoTokenizer
)
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('Maltehb/danish-bert-botxo')

model = AutoModelForMaskedLM.from_pretrained('Maltehb/danish-bert-botxo')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


data = load_dataset("text", data_files='data/gyldendalbooks.txt', sample_by='line')
dataset = data.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


logging_name = f'huggingface_logs'\
               +f'/BERT_extra_pretraining'

training_args = TrainingArguments(
    output_dir="../../../../../work3/s173991/huggingface_saves/BERT_gyldendal_241122",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
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
    train_dataset=dataset['train'],
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("../../../../../work3/s173991/huggingface_saves/BERT_gyldendal_241122")
