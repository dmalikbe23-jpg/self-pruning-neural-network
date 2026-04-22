import torch

def calculate_sparsity(model, threshold=1e-2):
    total = 0
    zero = 0

    for gates in model.get_all_gates():
        total += gates.numel()
        zero += torch.sum(gates < threshold).item()

    return 100 * zero / total