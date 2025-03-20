import torch
import time

def main():
    x = torch.randn(1024, 1024, device="cuda")
    for _ in range(200):
        x = x @ x
    torch.cuda.synchronize()
    print("worker done")


if __name__ == "__main__":
    main()
