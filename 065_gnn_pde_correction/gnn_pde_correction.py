import torch
import torch.nn as nn

class DummyGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
    def forward(self, x):
        return self.fc(x)


def main():
    gnn = DummyGNN()
    u = torch.randn(64, 64)
    lap = torch.roll(u, 1, 0) + torch.roll(u, -1, 0) + torch.roll(u, 1, 1) + torch.roll(u, -1, 1) - 4 * u
    corr = gnn(u.unsqueeze(-1)).squeeze(-1)
    u_next = u + 0.1 * lap + 0.01 * corr
    print("mean", u_next.mean().item())


if __name__ == "__main__":
    main()
