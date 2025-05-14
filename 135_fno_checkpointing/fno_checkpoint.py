import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, x):
        return checkpoint(self.block, x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device)
    x = torch.randn(1024, 32, device=device)
    y = model(x).mean()
    y.backward()
    print("checkpointed backward done")


if __name__ == "__main__":
    main()
