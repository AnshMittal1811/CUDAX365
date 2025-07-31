import numpy as np

rng = np.random.RandomState(1)

images = np.load("../211_depth_aware_nerf/images.npy")

pseudo_lidar = images.mean(axis=-1) + rng.normal(scale=0.05, size=images.shape[:3])

np.save("pseudo_lidar.npy", pseudo_lidar.astype(np.float32))
print("Wrote pseudo_lidar.npy")
