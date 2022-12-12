import torch

def calc_entropy(logits, problem_type='multiclass'):
    logits = torch.from_numpy(logits)
    if problem_type == 'multiclass':
        logit_func = torch.nn.Sigmoid()
        probs = logit_func(logits)
        H = -torch.sum(probs * torch.log2(probs))
        return H
        
    elif problem_type == 'multilabel':
        logit_func = torch.nn.Softmax()
        probs = logit_func(logits)
        H0 = -torch.sum(probs * torch.log2(probs), dim=1)
        H1 = -torch.sum((1.-probs) * torch.log2((1.-probs)), dim=1)
        H = H0 + H1
        return H