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


targets = ['Genre', 'Tekstbaand', 
           'Fremstillingsform', 'Semantisk_univers', 
           'Stemmer', 'Perspektiv', 
           'Holistisk_vurdering']

# lstm_filepath = f'./lightning_logs/{targets[0]}/num_epoch_20000-embedding_size_100-lstm_layers_4-lstm_size_256-l1_size_256-l2_size_128'

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

def hf_get_all_cv(metric: str='eval/f1_macro', target: str='Genre', model: str='BERT'):
    
    filepath_dict = {
        'BERT': f'./huggingface_logs/{target}/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05',
        'BERT gyldendal': f'./huggingface_logs/BERT_mlm_gyldendal/{target}/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05'
    }
    
    all_val = np.array([])
    for i in range(1, 11):
        hf_filepath = filepath_dict[model] + f'/CV_{i}'
        steps, val = get_run(hf_filepath)[metric] # the value is a tuple, with the first value being the steps
        all_val = np.vstack([all_val, val]) if all_val.size else val

    sem =  np.std(all_val, axis=0)/np.sqrt(all_val.shape[0])  # Standard error of the mean
    mean = np.mean(all_val, axis=0)
    return mean, sem, steps


def plot_hf_metric(metric: str='eval/f1_macro', target: str='Genre', model: str='BERT'):
    mean, sem, steps = hf_get_all_cv(metric, target, model)
    
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(steps, mean)
    plt.fill_between(steps, mean-sem, mean+sem, alpha=0.33)
    plt.title(f"{model}: {target} - {metric.split('/')[0]} {metric.split('/')[1].replace('_', ' ')}", fontsize=16)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel(f"{metric.split('/')[0]} {metric.split('/')[1].replace('_', ' ')}", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# plot_hf_metric()


# ['epoch', 'hp_metric', 
# 'train_step_loss', 'val_epoch_AUROC_macro', 
# 'val_epoch_acc_exact', 'val_epoch_acc_macro', 'val_epoch_acc_micro',
# 'val_epoch_f1_macro', 'val_epoch_loss', 'val_step_loss']

# Function takes a long time to run
# We want to get all metrics at the same time, to reduce runtime
def lstm_get_all_cv(target: str):
    
    metric_dict = {
        'train_step_loss':       'train loss',            
        'val_epoch_AUROC_macro': 'AUROC macro',           
        'val_epoch_acc_exact':   'accuracy exact',       
        'val_epoch_acc_micro':   'accuracy micro',       
        'val_epoch_acc_macro':   'accuracy macro',        
        'val_epoch_f1_macro':    'f1 macro',       
        'val_epoch_loss':        'validation epoch loss',
        'val_step_loss':         'validation step loss'
    }
    
    all_metrics = {}
    for i in range(1, 11):
        lstm_filepath = f'./lightning_logs'\
                        +f'/{target}'\
                        +f'/num_epoch_20000-embedding_size_100-lstm_layers_4-lstm_size_256-l1_size_256-l2_size_128'\
                        +f'/bs_64'\
                        +f'/CV_{i}'\
                        +f'/version_0'
        data = get_run(lstm_filepath) 
        for metric, val in data.items():  # the value is a tuple, with the first value being the steps
            if metric not in all_metrics.keys() and metric in metric_dict.keys():
                all_metrics[metric] = val[1]
            elif metric in metric_dict.keys():
                all_metrics[metric] = np.vstack([all_metrics[metric], val[1]])

    steps = data['train_step_loss'][0]
    all_sem = {}
    all_mean = {}
    for metric, vals in all_metrics.items():
        all_sem[metric]  = np.std(vals, axis=0)/np.sqrt(vals.shape[0])  # Standard error of the mean
        all_mean[metric] = np.mean(vals, axis=0)
    return all_mean, all_sem, steps


def plot_lstm_metric(target: str):
    all_mean, all_sem, steps = lstm_get_all_cv(target)
    
    metric_dict = {
        'train_step_loss':       'train loss',            
        'val_epoch_AUROC_macro': 'AUROC macro',           
        'val_epoch_acc_exact':   'accuracy exact',       
        'val_epoch_acc_micro':   'accuracy micro',       
        'val_epoch_acc_macro':   'accuracy macro',        
        'val_epoch_f1_macro':    'f1 macro',       
        'val_epoch_loss':        'validation epoch loss',
        'val_step_loss':         'validation step loss'
    }
    
    get_n_element = lambda x, n: x[0:len(x):n]
    steps = get_n_element(steps, 40)
    
    for metric in all_mean.keys():
        mean = get_n_element(all_mean[metric], 40)
        sem = get_n_element(all_sem[metric], 40)
    
        plt.figure(figsize=(7, 5), dpi=200)
        plt.plot(steps, mean)
        plt.fill_between(steps, mean-sem, mean+sem, alpha=0.33)
        plt.title(targets[0], fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel(metric_dict[metric], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.savefig(f'imgs/metric/LSTM/{target}/{metric}.png', bbox_inches="tight")
        plt.show()


plot_lstm_metric('Genre')
# lstm_get_all_cv('Genre')