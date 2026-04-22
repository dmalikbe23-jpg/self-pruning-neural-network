import torch.nn as nn
from models.prunable_linear import PrunableLinear

class PrunableCNN(nn.Module):
    def __init__(self):
        super(PrunableCNN, self).__init__()

        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32x16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64x8x8
        )

        # FC layers with pruning
        self.fc1 = PrunableLinear(64 * 8 * 8, 512)
        self.fc2 = PrunableLinear(512, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_all_gates(self):
        gates = []
        for module in self.modules():
            if hasattr(module, "get_gates"):
                gates.append(module.get_gates().view(-1))
        return gates