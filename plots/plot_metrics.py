from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

## https://stackoverflow.com/questions/52756152/tensorboard-extract-scalar-by-a-script
def get_run(path: str) -> dict:
    event_acc = event_accumulator.EventAccumulator(path)
    event_acc.Reload()
    data = {}

    for tag in sorted(event_acc.Tags()["scalars"]):
        x, y = [], []

        for scalar_event in event_acc.Scalars(tag):
            x.append(scalar_event.step)
            y.append(scalar_event.value)

        data[tag] = (np.asarray(x), np.asarray(y))
    return data


# filepath = f'./huggingface_logs'\
#           +f'/active_learning'\
#           +f'/BERT_mlm_gyldendal'\
#           +f'/entropy'\
#           +f'/Genre'\
#           +f'/BS16-BA4-MS3300-seed42-WD0.01-LR2e-05'\
#           +f'/CV_1'\
#           +f'/num_samples_350'


targets = ['Genre', 'Tekstbaand', 'Fremstillingsform', 'Semantisk_univers', 'Stemmer', 'Perspektiv', 'Holistisk_vurdering']

# lstm_filepath = f'./lightning_logs/{targets[0]}/num_epoch_20000-embedding_size_100-lstm_layers_4-lstm_size_256-l1_size_256-l2_size_128'
hf_filepath = f'./huggingface_logs/{targets[0]}/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05/CV_3'

## available tags
# ['eval/AUROC_macro', 'eval/accuracy_exact', 
#  'eval/accuracy_macro', 'eval/f1_macro', 
#  'eval/recall_macro', 'eval/precision_macro',   # -- NOT FOR REGULAR BERT RIGHT NOW
#  'eval/loss', 'eval/runtime', 
#  'eval/samples_per_second', 'eval/steps_per_second', 
#  'train/epoch', 'train/learning_rate', 
#  'train/loss', 'train/total_flos', 
#  'train/train_loss', 'train/train_runtime', 
#  'train/train_samples_per_second', 'train/train_steps_per_second']

data = get_run(hf_filepath)
print(data.keys())

def hf_get_all_cv():
    pass

def plot_hf_metrics():
    pass



for i in range(1, 11):
    hf_filepath = f'./huggingface_logs/{targets[0]}/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05/CV_{i}'
    metric = 'eval/accuracy_exact'
    steps, values = get_run(hf_filepath)[metric] # the value is a tuple, with the first value being the steps
    
