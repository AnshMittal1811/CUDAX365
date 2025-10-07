import os
import time

try:
    import torch
    import triton
    import triton.language as tl
except Exception:
    torch = None
    triton = None
    tl = None


def run_kernel():
    @triton.jit
    def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask, other=0.0)
        y = tl.load(Y + offs, mask=mask, other=0.0)
        tl.store(Z + offs, x + y, mask=mask)

    N = 1024 * 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    z = torch.empty_like(x)

    grid = (triton.cdiv(N, 1024),)
    add_kernel[grid](x, y, z, N, BLOCK=1024)
    torch.cuda.synchronize()


def main():
    os.environ["TRITON_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = "./triton3_dump"

    if torch is None or triton is None:
        print("Triton not available")
        return

    start = time.time()
    run_kernel()
    elapsed = time.time() - start

    with open("triton3_log.txt", "w", encoding="utf-8") as f:
        f.write(f"elapsed_sec={elapsed:.4f}\n")

    print("Wrote triton3_log.txt")


if __name__ == "__main__":
    main()
