from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from torchmetrics.functional.classification import (
    multilabel_exact_match,
    multilabel_accuracy, multiclass_accuracy,
    multilabel_f1_score, multiclass_f1_score,
    multilabel_recall, multiclass_recall,
    multilabel_precision, multiclass_precision,
    multilabel_auroc, multiclass_auroc
)
from sklearn.model_selection import KFold
from data_clean import *
from atel.data import BookCollection
import argparse
import yaml
from yaml import CLoader

parser = argparse.ArgumentParser(description='Arguments for running the BERT finetuning')
parser.add_argument(
    '--target_col',
    help='The target column to train the BERT model on.', 
    default=None
)
args = parser.parse_args()

TARGET = args.target_col

SEED = 42
NUM_SPLITS = 10
BATCH_SIZE = 16
BATCH_ACCUMALATION = 4
NUM_EPOCHS = 100
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
OUTPUT_ATTENTION = False
set_seed(SEED)

with open('target_info.yaml', 'r', encoding='utf-8') as f:
    target_info = yaml.load(f, Loader=CLoader)

assert TARGET in target_info.keys()  # checks if targets is part of the actual problem columns

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

tokenizer = AutoTokenizer.from_pretrained("../../../../../work3/s173991/huggingface_models/BERT_mlm_gyldendal")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


problem_type = target_info[TARGET]['problem_type']
NUM_LABELS   = target_info[TARGET]['num_labels']

print(f'STARTED TRAINING FOR: {TARGET}')
print(f'PROBLEM TYPE: {problem_type}')

df, labels = get_pandas_dataframe(book_col, TARGET)

label2id = dict(zip(labels, range(NUM_LABELS)))
id2label = dict(zip(range(NUM_LABELS), labels))

if problem_type == 'multilabel':
    multilabel = True
    p_t = "multi_label_classification"
    logit_func = torch.nn.Sigmoid()
    
else:
    multilabel = False
    p_t = "single_label_classification"
    logit_func = torch.nn.Softmax()


def compute_metrics(eval_pred):
    output, labels = eval_pred
    
    if OUTPUT_ATTENTION:
        logits = output[0]
        attention = output[1]
    else:
        logits = output

    preds = logit_func(torch.tensor(logits))
    labels = torch.tensor(labels).int()
    
    if problem_type == 'multilabel':
        acc_exact = multilabel_exact_match(preds, labels, num_labels=NUM_LABELS)
        acc_macro = multilabel_accuracy(preds, labels, num_labels=NUM_LABELS)
        
        # How are they calculated?:
        # The metrics are calculated for each label. 
        # So if there is 4 labels, then 4 recalls are calculated.
        # These 4 values are then averaged, which is the end score that is logged.
        # The default average applied is 'macro' 
        precision_macro = multilabel_precision(preds, labels, num_labels=NUM_LABELS)
        recall_macro = multilabel_recall(preds, labels, num_labels=NUM_LABELS)
        f1_macro = multilabel_f1_score(preds, labels, num_labels=NUM_LABELS)
        
        # AUROC score of 1 is a perfect score
        # AUROC score of 0.5 corresponds to random guessing.
        auroc_macro = multilabel_auroc(preds, labels, num_labels=NUM_LABELS, average="macro", thresholds=None)
        
        metrics = {
            'accuracy_exact':  acc_exact,
            'accuracy_macro':  acc_macro,
            'precision_macro': precision_macro,
            'recall_macro':    recall_macro,
            'f1_macro':        f1_macro,
            'AUROC_macro':     auroc_macro
        }
    else:
        acc_micro = multiclass_accuracy(preds, labels, num_classes=NUM_LABELS, average='micro')
        acc_macro = multiclass_accuracy(preds, labels, num_classes=NUM_LABELS, average='macro')
        precision_macro = multiclass_precision(preds, labels, num_classes=NUM_LABELS)
        recall_macro = multiclass_recall(preds, labels, num_classes=NUM_LABELS)
        f1_macro = multiclass_f1_score(preds, labels, num_classes=NUM_LABELS)
        auroc_macro = multiclass_auroc(preds, labels, num_classes=NUM_LABELS, average="macro", thresholds=None)
        
        metrics = {
            'accuracy_micro':  acc_micro,
            'accuracy_macro':  acc_macro,
            'precision_macro': precision_macro,
            'recall_macro':    recall_macro,
            'f1_macro':        f1_macro,
            'AUROC_macro':     auroc_macro
        }
        
    return metrics
    

dataset = Dataset.from_pandas(df.reset_index())
token_dataset = dataset.map(tokenize_function, batched=True)

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
all_splits = [k for k in kf.split(token_dataset)]

for k in range(NUM_SPLITS):
    
    print(f'\nTRAINING CV {k+1}/{NUM_SPLITS} - {TARGET}')
    
    train_idx, val_idx = all_splits[k]
    train_dataset = token_dataset.select(train_idx)
    val_dataset   = token_dataset.select(val_idx)

    model = AutoModelForSequenceClassification.from_pretrained("../../../../../work3/s173991/huggingface_models/BERT_mlm_gyldendal", 
                                                               num_labels=NUM_LABELS, 
                                                               problem_type=p_t,
                                                               label2id=label2id,
                                                               id2label=id2label,
                                                               output_attentions=OUTPUT_ATTENTION)
    
    only_cls_layer = False
    if only_cls_layer:
        for param in list(model.bert.embeddings.parameters()):
            param.requires_grad = False
            print("Froze Embedding Layer")
            
        for idx in range(len(model.bert.encoder.layer)):
            for param in list(model.bert.encoder.layer[idx].parameters()):
                param.requires_grad = False
            print('Froze Encoder layer')
    
    logging_name = f'huggingface_logs'\
                   +f'/BERT_mlm_gyldendal'\
                   +f'/version_2'\
                   +f'/{TARGET.replace(" ", "_")}'\
                   +f'/BERT-BS{BATCH_SIZE}'\
                   +f'-BA{BATCH_ACCUMALATION}'\
                   +f'-ep{NUM_EPOCHS}'\
                   +f'-seed{SEED}'\
                   +f'-WD{WEIGHT_DECAY}'\
                   +f'-LR{LEARNING_RATE}'\
                   +f'/CV_{k+1}'
    
    training_args = TrainingArguments(
        output_dir=f'../../../../../work3/s173991/huggingface_saves/{logging_name[17:]}',  # [17:] removes 'huggingface_logs'
        save_strategy='epoch',
        save_total_limit=1,
        # metric_for_best_model='f1_macro',  # f1-score for now
        # greater_is_better=True,
        # load_best_model_at_end=True,
        logging_strategy='epoch',
        logging_dir=logging_name,
        report_to='tensorboard',
        evaluation_strategy='epoch',
        seed=SEED,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    trainer = Trainer(
        model=model,    
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    print('Evulate model:')
    eval_dict = trainer.evaluate()
    print(eval_dict)
    with open(f'{logging_name}/eval_{TARGET}_CV{k+1}.yml', 'w') as outfile:
        yaml.dump(eval_dict, outfile, default_flow_style=False)

    print('Make predictions:')
    test_dataset = val_dataset.remove_columns("labels")
    outputs = trainer.predict(test_dataset)
    
    if OUTPUT_ATTENTION:
        logits = outputs.predictions[0]
        attention = outputs.predictions[-1]
    else:
        logits = outputs.predictions
    
    # print(compute_metrics((outputs.predictions, val_dataset['labels'])))

    torch.save(logits, f"{logging_name}/{TARGET}_CV{k+1}_logits.pt")
    ## Last garbage collection
    del model
    del trainer
