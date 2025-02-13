import time
import torch


def main():
    x = torch.randn(1 << 24, device="cuda")
    t0 = time.time()
    for _ in range(10):
        y = x * 1.01 + 0.1
    torch.cuda.synchronize()
    t1 = time.time()
    print("gpu_ms", (t1 - t0) * 1000)

    with open("io_blob.bin", "wb") as f:
        f.write(b"0" * (10 * 1024 * 1024))
    print("io done")


if __name__ == "__main__":
    main()
