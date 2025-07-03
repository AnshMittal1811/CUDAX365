import numpy as np

rng = np.random.RandomState(0)

h, w = 256, 256
background = rng.rand(h, w, 3).astype(np.float32)
segmask = (rng.rand(h, w) > 0.8).astype(np.float32)

lidar_points = rng.randint(0, min(h, w), size=(200, 2))

np.save("background.npy", background)
np.save("segmask.npy", segmask)
np.save("lidar_points.npy", lidar_points)

print("Wrote background.npy, segmask.npy, lidar_points.npy")
