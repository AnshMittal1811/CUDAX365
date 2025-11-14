
import argparse
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='INT4 matmul with Triton (dequant sim)')
    parser.add_argument('--m', type=int, default=256)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--k', type=int, default=256)
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()

    try:
        import torch
        import triton
        import triton.language as tl
        triton_ok = True
    except Exception:
        triton_ok = False
        torch = None

    if torch is None:
        print('Torch not available; using NumPy fallback')
        x = np.random.standard_normal((args.m, args.k)).astype(np.float32)
        w = np.random.randint(-8, 8, size=(args.k, args.n)).astype(np.float32) * 0.05
        t0 = time.perf_counter()
        for _ in range(args.iters):
            x @ w
        t1 = time.perf_counter()
        print(f'NumPy matmul avg ms: {(t1 - t0) * 1000 / args.iters:.3f}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    w_int4 = torch.randint(-8, 8, (args.k, args.n), device=device, dtype=torch.int8)
    scale = torch.tensor(0.05, device=device, dtype=torch.float16)

    if triton_ok and device.type == 'cuda':
        @triton.jit
        def dequant_kernel(inp, out, scale_ptr, n_elements, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements
            val = tl.load(inp + offs, mask=mask, other=0).to(tl.float16)
            scale = tl.load(scale_ptr)
            val = val * scale
            tl.store(out + offs, val, mask=mask)

        w_deq = torch.empty((args.k, args.n), device=device, dtype=torch.float16)
        n_elements = w_deq.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK']),)
        dequant_kernel[grid](w_int4, w_deq, scale, n_elements, BLOCK=256)
    else:
        w_deq = w_int4.float() * scale

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(args.iters):
        y = x @ w_deq
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f'Avg matmul ms: {(t1 - t0) * 1000 / args.iters:.3f}')
    print(f'Output shape: {y.shape}')

if __name__ == '__main__':
    main()
