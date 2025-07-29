import numpy as np

rng = np.random.RandomState(0)

images = rng.rand(16, 64, 64, 3).astype(np.float32)
depths = rng.rand(16, 64, 64).astype(np.float32)

np.save("images.npy", images)
np.save("depths.npy", depths)
print("Wrote images.npy and depths.npy")
