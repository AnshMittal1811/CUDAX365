import os
import torch
import triton
import triton.language as tl

os.environ["TRITON_CACHE_DIR"] = "./triton_cache"


@triton.jit
def bias_relu_kernel(x_ptr, b_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    if hasattr(tl, "inline_asm_elementwise"):
        y = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2; max.f32 $0, $0, 0f00000000;",
            [x, b],
            dtype=tl.float32,
        )
    else:
        y = tl.maximum(x + b, 0)
    tl.store(y_ptr + offs, y, mask=mask)


def main():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    bias_relu_kernel[grid](x, b, y, n, BLOCK=BLOCK)
    torch.cuda.synchronize()

    # dump PTX if available
    kernel = bias_relu_kernel
    try:
        asm = kernel.asm["ptx"]
        with open("bias_relu_kernel.ptx", "w", encoding="utf-8") as f:
            f.write(asm)
        print("wrote bias_relu_kernel.ptx")
    except Exception as e:
        print("PTX not available from triton runtime:", e)


if __name__ == "__main__":
    main()
