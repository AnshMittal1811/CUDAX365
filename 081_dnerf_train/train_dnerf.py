import torch
import torch.nn as nn

class DNeRF(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)


def synthetic_batch(batch=8192):
    x = torch.rand(batch, 3) * 2 - 1
    t = torch.rand(batch, 1)
    center = torch.stack([0.3 * torch.sin(2 * 3.1415 * t), 0.0 * t, 0.0 * t], dim=2).squeeze(1)
    dist = ((x - center) ** 2).sum(dim=1, keepdim=True)
    sigma = torch.exp(-10.0 * dist)
    return torch.cat([x, t], dim=1), sigma


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DNeRF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(200):
        x, y = synthetic_batch()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print("step", step, "loss", float(loss))


if __name__ == "__main__":
    main()
