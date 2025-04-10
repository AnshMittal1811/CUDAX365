import numpy as np

w, h, frames = 320, 180, 60
for i in range(frames):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = (i * 4) % 255
    img[:, :, 1] = (np.linspace(0, 255, w).astype(np.uint8)[None, :])
    img[:, :, 2] = (np.linspace(0, 255, h).astype(np.uint8)[:, None])
    img.tofile(f"frame_{i:04d}.rgb")
print("wrote raw rgb frames")
