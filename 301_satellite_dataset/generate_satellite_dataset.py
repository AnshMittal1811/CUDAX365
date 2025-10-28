import numpy as np

rng = np.random.RandomState(0)

images = rng.randint(0, 255, size=(50, 3, 128, 128), dtype=np.uint8)
masks = rng.randint(0, 2, size=(50, 128, 128), dtype=np.uint8)

np.save("sat_images.npy", images)
np.save("sat_masks.npy", masks)

print("Wrote sat_images.npy and sat_masks.npy")
