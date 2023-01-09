import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from atel.data import BookCollection
from data_clean import *
import yaml
from yaml import CLoader
sns.set_style('whitegrid')

NUM_SPLITS = 10
SEED = 42
set_seed(SEED)

# shape = rows, cols
# figsize = width, height
plot_dict = {
    'Genre':               {'shape': (5, 3), 'figsize': (12, 12.5)},
    'Tekstbaand':          {'shape': (2, 2), 'figsize': (8, 5)},
    'Fremstillingsform':   {'shape': (4, 2), 'figsize': (8, 10)},
    'Semantisk_univers':   {'shape': (3, 2), 'figsize': (8, 7.5)},
    'Stemmer':             {'shape': (1, 3), 'figsize': (12, 2.5)},
    'Perspektiv':          {'figsize': (5, 3)},
    'Holistisk_vurdering': {'figsize': (7, 5)},
}

def plot_multiclass_confusion_matrix(y_true, y_preds):
    cm = confusion_matrix(y_true.reshape(-1), np.argmax(y_preds, axis=1))
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]

    labels = [f'{x}\n{round(y, 4)}%' for x, y in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.asarray(labels).reshape(cm.shape)

    plt.figure(figsize=plot_dict[TARGET.replace(" ", "_").replace("å", "aa")]['figsize'], dpi=300)
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'{TARGET}', fontsize=16)
    plt.savefig(f'imgs/confusion_matrix/{TARGET.replace(" ", "_").replace("å", "aa")}/{TARGET.replace(" ", "_").replace("å", "aa")}_cm.png', bbox_inches="tight")
    # plt.show()


def plot_multilabel_confusion_matrix(y_true, y_preds):
    cms = multilabel_confusion_matrix(y_true, y_preds)

    rows, cols = plot_dict[TARGET.replace(" ", "_").replace("å", "aa")]['shape']

    fig, axes = plt.subplots(rows, cols, figsize=plot_dict[TARGET.replace(" ", "_").replace("å", "aa")]['figsize'], dpi=300)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'Confusion matrices: {TARGET}', fontsize=16)
    
    for i, (cm, ax) in enumerate(zip(cms, axes.flatten())):
        cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
        
        labels = [f'{x}\n{round(y, 4)}%' for x, y in zip(cm.flatten(), cm_norm.flatten())]
        labels = np.asarray(labels).reshape(2, 2)
        
        sns.heatmap(cm, ax=ax, annot=labels, fmt='', cmap='Blues')
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.set_title(label_names[i], fontsize=14)
    
    plt.savefig(f'imgs/confusion_matrix/{TARGET.replace(" ", "_").replace("å", "aa")}/{TARGET.replace(" ", "_").replace("å", "aa")}_cm.png', bbox_inches="tight")
    # plt.close()
    # plt.show()


with open('target_info.yaml', 'r', encoding='utf-8') as f:
    target_info = yaml.load(f, Loader=CLoader)

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

for TARGET in target_info.keys():
    print(f'Plotting for {TARGET}')
    problem_type = target_info[TARGET]['problem_type']

    df, label_names = get_pandas_dataframe(book_col, TARGET)
    labels = df.labels.values
    labels = np.array([a for a in labels])
    print(label_names)

    kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
    all_splits = [k for k in kf.split(labels)]

    if problem_type == 'multilabel':
        logit_func = torch.nn.Sigmoid()
    else:
        logit_func = torch.nn.Softmax()


    all_preds = np.array([])
    y_true    = np.array([])
    for i in range(10):
        filepath = f'./huggingface_logs'\
                +f'/BERT_mlm_gyldendal'\
                +f'/{TARGET.replace(" ", "_").replace("å", "aa")}'\
                +f'/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05'\
                +f'/CV_{i+1}'\
                +f'/{TARGET.replace("å", "aa")}_CV{i+1}_logits.pt'
                
        
        logits = torch.from_numpy(torch.load(filepath))
        preds  = torch.round(logit_func(logits))  # rounds down 0.5, which is the behaviour in torchmetrics too (as default)
        
        train_idx, val_idx = all_splits[i]

        if y_true.shape[0] == 0 and all_preds.shape[0] == 0:
            all_preds = preds.numpy()
            y_true    = labels[val_idx] if len(labels.shape) > 1 else labels[val_idx, None]
        else:
            all_preds = np.vstack([all_preds, preds.numpy()])
            y_true    = np.vstack([y_true, labels[val_idx] if len(labels.shape) > 1 else labels[val_idx, None]])


    if problem_type == 'multilabel':
        plot_multilabel_confusion_matrix(y_true, all_preds)
    else:
        plot_multiclass_confusion_matrix(y_true, all_preds)

print('Done!')
