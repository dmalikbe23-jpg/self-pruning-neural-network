import matplotlib.pyplot as plt

def plot_gate_distribution(model):
    all_gates = []

    for gates in model.get_all_gates():
        all_gates.extend(gates.detach().cpu().numpy())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.savefig("gate_distribution.png")
    plt.close()