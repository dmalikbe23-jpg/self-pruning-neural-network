import torch
from utils.metrics import calculate_sparsity
from utils.plot import plot_gate_distribution

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = 100 * correct / total
    sparsity = calculate_sparsity(model)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    plot_gate_distribution(model)

    return accuracy, sparsity