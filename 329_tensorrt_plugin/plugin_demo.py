
import numpy as np

def swish(x):
    return x / (1.0 + np.exp(-x))

def main():
    x = np.linspace(-3, 3, 8, dtype=np.float32)
    y = swish(x)
    print('Swish activation output:', y.tolist())
    print('TensorRT plugin skeleton is in plugin_stub.cpp')

if __name__ == '__main__':
    main()
