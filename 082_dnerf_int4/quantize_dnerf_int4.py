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


def fake_int4_quant(t):
    scale = t.abs().max() / 7.0
    return torch.clamp((t / scale).round(), -8, 7) * scale


def main():
    model = DNeRF()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(fake_int4_quant(p))
    x = torch.randn(1024, 4)
    y = model(x)
    print("quantized output mean", float(y.mean()))


if __name__ == "__main__":
    main()
