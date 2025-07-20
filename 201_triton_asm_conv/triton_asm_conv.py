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


def run_triton():
    @triton.jit
    def conv_kernel(X, W, Y, stride_h, stride_w, H, Wd, K, BLOCK: tl.constexpr):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)
        offs = tl.arange(0, BLOCK)
        x = pid0 * BLOCK + offs
        y = pid1
        mask = x < Wd

        acc = tl.zeros([BLOCK], tl.float32)
        for ky in range(K):
            for kx in range(K):
                ix = x + kx
                iy = y + ky
                idx = iy * stride_h + ix
                w_idx = ky * K + kx
                val = tl.load(X + idx, mask=mask, other=0.0)
                w = tl.load(W + w_idx)
                acc += val * w
        tl.store(Y + y * stride_w + x, acc, mask=mask)

    H = 64
    Wd = 64
    K = 3
    x = torch.randn((H + K - 1, Wd + K - 1), device="cuda", dtype=torch.float32)
    w = torch.randn((K, K), device="cuda", dtype=torch.float32)
    y = torch.empty((H, Wd), device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(Wd, 128), H)
    conv_kernel[grid](x, w, y, x.stride(0), y.stride(0), H, Wd, K, BLOCK=128)
    torch.cuda.synchronize()
    return y


def main():
    os.environ["TRITON_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = "./triton_dump"

    if torch is None or triton is None:
        print("Triton not available; skipped kernel build")
        return

    start = time.time()
    _ = run_triton()
    elapsed = time.time() - start

    with open("triton_conv_log.txt", "w", encoding="utf-8") as f:
        f.write(f"elapsed_sec={elapsed:.4f}\n")

    print("Wrote triton_conv_log.txt")


if __name__ == "__main__":
    main()
