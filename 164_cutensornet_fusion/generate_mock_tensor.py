import numpy as np

rng = np.random.RandomState(0)

x = rng.randn(64, 64, 64).astype(np.float32)
np.save("input_tensor.npy", x)
print("Wrote input_tensor.npy")
