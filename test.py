import torch

filepath = 'huggingface_logs/Fremstillingsform/BERT-BS16-BA4-ep100-seed42-WD0.01-LR2e-05/CV_1/Fremstillingsform_CV1_best_model_logits.pt'

test = torch.load(filepath)

print(test)
