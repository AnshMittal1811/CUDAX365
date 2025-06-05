import numpy as np

rng = np.random.RandomState(0)

h, w = 256, 256
base = np.zeros((h, w), dtype=np.float32)
base[64:192, 64:192] = 1.0
noise = rng.normal(scale=0.25, size=(h, w)).astype(np.float32)
noisy = np.clip(base + noise, 0.0, 1.0)

np.save("noisy_image.npy", noisy)
print("Wrote noisy_image.npy")
