import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import csv

from models.network import PrunableCNN
from utils.loss import sparsity_loss
from utils.metrics import calculate_sparsity
from utils.plot import plot_gate_distribution
from config import *

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (IMPORTANT for better accuracy)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=False
)


# 🔥 TRAIN FUNCTION
def train(lambda_val):
    model = PrunableCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_model = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Lambda {lambda_val} | Epoch [{epoch+1}/{EPOCHS}]")

        for data, target in loop:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward
            output = model(data)

            # Loss
            ce_loss = criterion(output, target)
            sp_loss = sparsity_loss(model)
            loss = ce_loss + lambda_val * sp_loss

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 🔥 Sparsity after epoch
        sparsity = calculate_sparsity(model)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Sparsity: {sparsity:.2f}%")

        # 🔥 Validation
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

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_model = model.state_dict()

    # 🔥 Save best model
    torch.save(best_model, f"best_model_lambda_{lambda_val}.pth")
    print(f"\n✅ Best Accuracy for lambda {lambda_val}: {best_acc:.2f}% (model saved)\n")

    return model


# 🔥 EVALUATION FUNCTION
def evaluate(model):
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

    print("✅ Final Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    # Save plot
    plot_gate_distribution(model)

    return accuracy, sparsity


# 🚀 MAIN
if __name__ == "__main__":
    results = []

    for lambda_val in LAMBDA_VALUES:
        print("\n" + "="*60)
        print(f"🔥 Training with lambda = {lambda_val}")
        print("="*60)

        model = train(lambda_val)

        print("Evaluating model...")
        acc, sparsity = evaluate(model)

        results.append((lambda_val, acc, sparsity))

    # 📊 Print summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*60)

    for res in results:
        print(f"Lambda: {res[0]} | Accuracy: {res[1]:.2f}% | Sparsity: {res[2]:.2f}%")

    # 💾 Save results to CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lambda", "Accuracy", "Sparsity"])
        for res in results:
            writer.writerow(res)

    print("\n✅ Results saved to results.csv")