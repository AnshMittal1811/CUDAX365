import numpy as np

rng = np.random.RandomState(0)

real = rng.normal(loc=0.5, scale=0.1, size=(512, 16)).astype(np.float32)
fake = rng.normal(loc=0.45, scale=0.15, size=(512, 16)).astype(np.float32)

np.save("real_features.npy", real)
np.save("fake_features.npy", fake)
print("Wrote real_features.npy and fake_features.npy")
