import time
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    def forward(self, x):
        return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyMLP().to(device).eval()
    x = torch.randn(1024, 64, device=device)

    # fake INT4 quantization via per-tensor scaling
    with torch.no_grad():
        w = model.net[0].weight
        scale = w.abs().max() / 7.0
        w_q = torch.clamp((w / scale).round(), -8, 7) * scale
        model.net[0].weight.copy_(w_q)

    t0 = time.time()
    for _ in range(200):
        _ = model(x)
    torch.cuda.synchronize()
    t1 = time.time()
    print("fps", 200 / (t1 - t0))


if __name__ == "__main__":
    main()
