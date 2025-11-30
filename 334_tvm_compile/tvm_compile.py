
import time
import numpy as np

def main():
    try:
        import tvm
        import tvm.relay as relay
        import torch
        tvm_ok = True
    except Exception:
        tvm_ok = False

    if not tvm_ok:
        a = np.random.standard_normal((1, 128)).astype(np.float32)
        w = np.random.standard_normal((128, 64)).astype(np.float32)
        t0 = time.perf_counter()
        _ = a @ w
        t1 = time.perf_counter()
        print(f'NumPy fallback ms: {(t1 - t0)*1000:.3f}')
        return

    import torch.nn as nn
    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
    input_shape = (1, 128)
    scripted = torch.jit.trace(model, torch.randn(*input_shape))
    mod, params = relay.frontend.from_pytorch(scripted, [('input0', input_shape)])
    target = 'cuda' if tvm.cuda().exist else 'llvm'
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    print('TVM build finished:', lib)

if __name__ == '__main__':
    main()
