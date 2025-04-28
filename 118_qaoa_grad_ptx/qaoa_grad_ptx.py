import os
from pathlib import Path
import torch

torch.manual_seed(0)
os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torchinductor_cache"


def cost(x):
    return (torch.sin(x) * torch.cos(x)).sum()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1024, device=device, requires_grad=True)

    compiled = torch.compile(cost, fullgraph=True)
    y = compiled(x)
    y.backward()
    torch.cuda.synchronize()

    ptx = list(Path("./torchinductor_cache").rglob("*.ptx"))
    print("ptx files", len(ptx))
    for p in ptx[:5]:
        print(p)


if __name__ == "__main__":
    main()
