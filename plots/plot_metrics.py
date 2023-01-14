from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from yaml import CLoader
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

with open('translation.yaml', 'r', encoding='utf-8') as f:
    translation = yaml.load(f, Loader=CLoader)

targets = [
    'Genre',
    'Tekstbaand',
    'Fremstillingsform',
    'Semantisk_univers', 
    'Stemmer',
    'Perspektiv', 
    'Holistisk_vurdering'
]

hf_metric_dict = {
        'eval/AUROC_macro':    'Val. AUROC Macro',
        'eval/accuracy_exact': 'Val. Subset Acc.',
        'eval/accuracy_micro': 'Val. Acc. Micro',
        'eval/accuracy_macro': 'Val. Acc. Macro',
        'eval/f1_macro':       'Val. F1 Macro',
        'eval/loss':           'Val Loss'
    }

lstm_metric_dict = {
        'train_step_loss':       'Train Loss',
        'val_epoch_AUROC_macro': 'Val. AUROC Macro',
        'val_epoch_acc_exact':   'Val. Subset Acc.',
        'val_epoch_acc_micro':   'Val. Acc. Micro',
        'val_epoch_acc_macro':   'Val. Acc. Macro',
        'val_epoch_f1_macro':    'Val. F1 Macro',
        'val_epoch_loss':        'Val. Epoch Loss',
        'val_step_loss':         'Val. Step Loss'
    }

model_title_dict = {
    'BERT': 'Danish BERT',
    'BERT_Gyldendal': 'Danish BERT + Gyldendal',
    'BabyBERTa_WU': 'BørneBÆRTa $\div$ WU',
    'BabyBERTa_WU_GW': 'BørneBÆRTa $+$ WU',
    'BabyBERTa_noWU': 'BørneBÆRTa $\div$ WU $+$ GW',
    'BabyBERTa_noWU_GW': 'BørneBÆRTa $+$ WU $+$ GW',
    'BabyBERTa_GWstart_Gyldendal': 'BørneBÆRTa (GW $+$ Gyl)',
}


def hf_get_all_cv(target: str='Genre', model: str='BERT'):
    
    filepath_dict = {
        'BERT': f'./huggingface_logs/{target}/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05',
        'BERT_Gyldendal': f'./huggingface_logs/BERT_mlm_gyldendal/{target}/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05',
        'BabyBERTa_WU': f'./babyberta_logs/models/BabyBERTa_091122/{target}/BERT-BS16-BA4-ep500-seed42-WD0.01-LR2e-05',
        'BabyBERTa_WU_GW': f'./babyberta_logs/models/BabyBERTa_091122_GW/{target}/BERT-BS16-BA4-ep500-seed42-WD0.01-LR2e-05',
        'BabyBERTa_noWU': f'./babyberta_logs/models/BabyBERTa_131022/{target}/BERT-BS16-BA4-ep500-seed42-WD0.01-LR2e-05',
        'BabyBERTa_noWU_GW': f'./babyberta_logs/models/BabyBERTa_131022_GW/{target}/BERT-BS16-BA4-ep500-seed42-WD0.01-LR2e-05',
        'BabyBERTa_GWstart_Gyldendal': f'./babyberta_logs/models/BabyBERTa-GWstart-gyldendal/{target}/BERT-BS16-BA4-ep500-seed42-WD0.01-LR2e-05'
    }
    
    all_metrics = {}
    for i in range(1, 11):
        hf_filepath = filepath_dict[model] + f'/CV_{i}'
        data = get_run(hf_filepath) 
        for metric, val in data.items():  # the value is a tuple, with the first value being the steps
            if metric not in all_metrics.keys() and metric in hf_metric_dict.keys():
                all_metrics[metric] = val[1]
            elif metric in hf_metric_dict.keys():
                all_metrics[metric] = np.vstack([all_metrics[metric], val[1]])
    
    steps = data['eval/loss'][0]
    all_sem = {}
    all_mean = {}
    for metric, vals in all_metrics.items():
        all_sem[metric]  = np.std(vals, axis=0)/np.sqrt(vals.shape[0])  # Standard error of the mean
        all_mean[metric] = np.mean(vals, axis=0)
    return all_mean, all_sem, steps


def plot_hf_metric(target: str='Genre', model: str='BERT'):
    all_mean, all_sem, steps = hf_get_all_cv(target, model)
    
    for metric in all_mean.keys():
        mean = all_mean[metric]
        sem = all_sem[metric]
        
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(steps, mean)
        plt.fill_between(steps, mean-sem, mean+sem, alpha=0.33)
        plt.title(f"{model_title_dict[model]}\n{translation['features'][target.replace('aa', 'å').replace('_', ' ')]} - {hf_metric_dict[metric]}", fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel(f"{hf_metric_dict[metric]}", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"imgs/metrics/{model}/{target}/{metric.replace('/', '_')}.png", bbox_inches="tight")
        plt.close()
        # plt.show()

model_names = [
    'BERT', 'BERT_Gyldendal', 
    'BabyBERTa_WU', 'BabyBERTa_WU_GW', 
    'BabyBERTa_noWU', 'BabyBERTa_noWU_GW', 
    'BabyBERTa_GWstart_Gyldendal'
]

for model in model_names:
    print(f'Starting plotting for {model}...')
    for target in targets:
        plot_hf_metric(target, model)
        print(f'Finished with {target}')
    print(f'Done with {model}!')
    
# Function takes a long time to run
# We want to get all metrics at the same time, to reduce runtime
def lstm_get_all_cv(target: str):
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
            if metric not in all_metrics.keys() and metric in lstm_metric_dict.keys():
                all_metrics[metric] = val[1]
            elif metric in lstm_metric_dict.keys():
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
    
    get_n_element = lambda x, n: x[0:len(x):n]
    steps = get_n_element(steps, 40)
    
    for metric in all_mean.keys():
        mean = get_n_element(all_mean[metric], 40)
        sem = get_n_element(all_sem[metric], 40)
    
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(steps, mean)
        plt.fill_between(steps, mean-sem, mean+sem, alpha=0.33)
        plt.title(f'LSTM\n{translation["features"][target.replace("aa", "å").replace("_", " ")]} - {lstm_metric_dict[metric]}', fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel(lstm_metric_dict[metric], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f'imgs/metrics/LSTM/{target}/{metric}.png', bbox_inches="tight")
        plt.close()
        # plt.show()

# print('Starting plotting for LSTM...')
# for target in targets:
#     plot_lstm_metric(target)
#     print(f'Finished plot for {target}')
# print('Done with LSTM!')
