import torch
import time


def main():
    device = "cuda"
    x = torch.randn(1 << 24, device=device)
    y = torch.empty_like(x)

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    t0 = time.time()
    with torch.cuda.stream(stream1):
        y.copy_(x)
    with torch.cuda.stream(stream2):
        z = x * 1.01
    torch.cuda.synchronize()
    t1 = time.time()
    print("elapsed_ms", (t1 - t0) * 1000)


if __name__ == "__main__":
    main()
