import numpy as np

frames = 30
h, w = 256, 256

for i in range(frames):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (i * 7) % 255
    img.tofile(f"frame_{i:03d}.raw")

print("Wrote raw frames")
