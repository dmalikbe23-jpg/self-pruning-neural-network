import torch

def sparsity_loss(model):
    loss = 0.0
    for gates in model.get_all_gates():
        loss += torch.sum(gates)
    return loss