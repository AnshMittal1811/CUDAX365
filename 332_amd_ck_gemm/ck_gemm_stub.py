
import numpy as np

def main():
    a = np.random.standard_normal((128, 128)).astype(np.float32)
    b = np.random.standard_normal((128, 128)).astype(np.float32)
    c = a @ b
    print('GEMM fallback (NumPy) output mean:', float(c.mean()))
    print('If ROCm/CK is available, replace this with CK GEMM example.')

if __name__ == '__main__':
    main()
