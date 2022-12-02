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
        idx0 = probs > 0
        idx1 = (1.-probs) > 0
        H0 = -torch.sum(probs[idx0] * torch.log2(probs[idx0]))
        H1 = -torch.sum((1.-probs)[idx1] * torch.log2((1.-probs)[idx1]))
        H = H0 + H1
        
    return H