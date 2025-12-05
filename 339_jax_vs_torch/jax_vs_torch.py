
import time
import numpy as np

def main():
    n = 256
    a = np.random.standard_normal((n, n)).astype(np.float32)
    b = np.random.standard_normal((n, n)).astype(np.float32)

    t0 = time.perf_counter()
    _ = a @ b
    t1 = time.perf_counter()
    print(f'NumPy matmul ms: {(t1 - t0) * 1000:.2f}')

    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ta = torch.from_numpy(a).to(device)
        tb = torch.from_numpy(b).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        _ = ta @ tb
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        print(f'Torch matmul ms: {(t3 - t2) * 1000:.2f}')
    except Exception:
        print('Torch not available')

    try:
        import jax
        import jax.numpy as jnp
        ja = jnp.array(a)
        jb = jnp.array(b)
        t4 = time.perf_counter()
        _ = ja @ jb
        _ = jax.block_until_ready(_)
        t5 = time.perf_counter()
        print(f'JAX matmul ms: {(t5 - t4) * 1000:.2f}')
    except Exception:
        print('JAX not available')

if __name__ == '__main__':
    main()
