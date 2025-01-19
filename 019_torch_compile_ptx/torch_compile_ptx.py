import os
import torch

os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torchinductor_cache"

torch.manual_seed(0)
device = "cuda"


@torch.compile
def kernel(x, y):
    z = torch.sin(x) * torch.cos(y)
    return z @ z.transpose(-1, -2)


def main():
    x = torch.randn(256, 256, device=device)
    y = torch.randn(256, 256, device=device)
    out = kernel(x, y)
    torch.cuda.synchronize()
    print(out.shape)


if __name__ == "__main__":
    main()
