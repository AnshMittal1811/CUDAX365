import time
import json
import torch


def gemm_gflops(m, n, k, iters=50):
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    t1 = time.time()
    flops = 2 * m * n * k * iters
    return flops / (t1 - t0) / 1e9


def main():
    report = {
        "gemm_1024": gemm_gflops(1024, 1024, 1024),
        "gemm_2048": gemm_gflops(2048, 2048, 2048, iters=10),
    }
    with open("gflops_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(report)


if __name__ == "__main__":
    main()
