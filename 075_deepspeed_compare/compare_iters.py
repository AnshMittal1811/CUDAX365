import time
import torch


def run_loop(iters=20, size=1024):
    x = torch.randn(size, size, device="cuda")
    t0 = time.time()
    for _ in range(iters):
        y = x @ x
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    base = run_loop()
    lora = run_loop()
    print("baseline_ms", base * 1000)
    print("qlora_like_ms", lora * 1000)


if __name__ == "__main__":
    main()
