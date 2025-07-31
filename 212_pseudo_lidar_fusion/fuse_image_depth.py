import numpy as np

images = np.load("../211_depth_aware_nerf/images.npy")
pseudo = np.load("pseudo_lidar.npy")

fused = np.concatenate([images, pseudo[..., None]], axis=-1)
np.save("fused_inputs.npy", fused)
print("Wrote fused_inputs.npy")
