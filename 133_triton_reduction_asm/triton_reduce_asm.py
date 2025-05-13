import os
import torch
import triton
import triton.language as tl

os.environ["TRITON_CACHE_DIR"] = "./triton_cache"

@triton.jit
def reduce_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # optional inline asm if supported
    if hasattr(tl, "inline_asm_elementwise"):
        x = tl.inline_asm_elementwise("add.rn.f32 $0, $1, 0f00000000;", [x], dtype=tl.float32)
    s = tl.sum(x, axis=0)
    if tl.program_id(0) == 0:
        tl.store(out_ptr, s)


def main():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty(1, device="cuda", dtype=torch.float32)
    BLOCK = 1024
    reduce_kernel[(1,)](x, out, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    print("sum", float(out.cpu()))

    try:
        asm = reduce_kernel.asm["ptx"]
        with open("reduce_kernel.ptx", "w", encoding="utf-8") as f:
            f.write(asm)
        print("wrote reduce_kernel.ptx")
    except Exception as e:
        print("ptx unavailable", e)


if __name__ == "__main__":
    main()
