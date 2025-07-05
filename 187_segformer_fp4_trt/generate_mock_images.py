import numpy as np

rng = np.random.RandomState(0)

images = rng.rand(16, 3, 64, 64).astype(np.float32)
labels = rng.randint(0, 2, size=(16, 64, 64)).astype(np.int64)

np.save("mock_images.npy", images)
np.save("mock_labels.npy", labels)
print("Wrote mock_images.npy and mock_labels.npy")
