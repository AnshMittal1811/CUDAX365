import numpy as np

rng = np.random.RandomState(0)
frames = 8
h, w = 128, 128

for i in range(frames):
    volume = rng.rand(32, 32, 32).astype(np.float32)
    projection = volume.mean(axis=0)
    projection = (projection - projection.min()) / (projection.ptp() + 1e-6)
    np.save(f"nerf_frame_{i:03d}.npy", projection)

print("Wrote nerf_frame_*.npy")
