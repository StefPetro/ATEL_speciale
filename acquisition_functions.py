import torch

def calc_entropy(pk, problem_type='multi_class'):
    idx = pk > 0
    if problem_type == 'multi_class':
        H = -torch.sum(pk[idx] * torch.log2(pk[idx]))
    elif problem_type == 'multi_label':
        H = None
    return H