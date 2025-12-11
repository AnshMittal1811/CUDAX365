
import time
import numpy as np

def main():
    try:
        import nvidia.dali  # noqa: F401
        dali_ok = True
    except Exception:
        dali_ok = False

    if not dali_ok:
        # CPU fallback: simple augmentation
        imgs = np.random.randint(0, 255, size=(64, 224, 224, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        _ = imgs[:, ::-1, :, :]
        t1 = time.perf_counter()
        print(f'CPU flip ms: {(t1 - t0) * 1000:.2f}')
        return

    print('DALI available; plug in DALI pipeline here.')

if __name__ == '__main__':
    main()
