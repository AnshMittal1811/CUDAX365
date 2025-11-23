
import numpy as np

def main():
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a)
    out = a + b
    print('CPU fallback result:', out.tolist())

if __name__ == '__main__':
    main()
