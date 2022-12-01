import torch

def calc_entropy(logits, problem_type='multi_class'):
    if problem_type == 'multi_class':
        logit_func = torch.nn.Sigmoid()
        probs = logit_func(logits)
        idx = probs > 0
        H = -torch.sum(probs[idx] * torch.log2(probs[idx]))
    elif problem_type == 'multi_label':
        logit_func = torch.nn.Softmax()
        probs = logit_func(logits)
        idx = probs > 0
        H = None
    return H