import os
from pathlib import Path
import torch
import torch.nn as nn

os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torchinductor_cache"

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=3, num_bands=6):
        super().__init__()
        self.freq = 2 ** torch.arange(num_bands).float()
    def forward(self, x):
        freqs = self.freq.to(x.device)[None, None, :]
        x = x.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(x.shape[0], -1)

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ff = FourierFeatures().to(device)
    mlp = MLP(3 * 6 * 2).to(device)

    def render(x):
        return mlp(ff(x))

    render = torch.compile(render, fullgraph=True)
    x = torch.randn(1024, 3, device=device)
    y = render(x)
    torch.cuda.synchronize()
    print(y.shape)

    ptx = list(Path("./torchinductor_cache").rglob("*.ptx"))
    print("ptx files", len(ptx))
    for p in ptx[:5]:
        print(p)


if __name__ == "__main__":
    main()
