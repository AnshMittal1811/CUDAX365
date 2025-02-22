import torch
import torch.nn as nn

class DummySurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    surrogate = DummySurrogate().to(device)
    rho = torch.randn(1, 64, 64, 1, device=device)

    rho_next = rho + 0.01 * torch.randn_like(rho)
    corr = surrogate(rho)
    rho_corr = rho_next + 0.1 * corr
    print("rho", rho_corr.mean().item())


if __name__ == "__main__":
    main()
