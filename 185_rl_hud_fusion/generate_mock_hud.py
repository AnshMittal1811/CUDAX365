import numpy as np

rng = np.random.RandomState(1)

segmask = (rng.rand(64, 64) > 0.7).astype(np.float32)
lidar = rng.randn(128, 2).astype(np.float32)

np.save("hud_segmask.npy", segmask)
np.save("hud_lidar.npy", lidar)
print("Wrote hud_segmask.npy and hud_lidar.npy")
