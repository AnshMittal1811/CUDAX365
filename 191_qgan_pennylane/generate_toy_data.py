import numpy as np

rng = np.random.RandomState(0)

real = rng.normal(loc=0.5, scale=0.1, size=(512, 1)).astype(np.float32)
np.save("real_samples.npy", real)
print("Wrote real_samples.npy")
