import numpy as np
import matplotlib.pyplot as plt

background = np.load("background.npy")
segmask = np.load("segmask.npy")
lidar = np.load("lidar_points.npy")

plt.figure(figsize=(4, 4))
plt.imshow(background)
plt.imshow(segmask, cmap="Reds", alpha=0.4)
plt.scatter(lidar[:, 1], lidar[:, 0], s=5, c="cyan")
plt.axis("off")
plt.tight_layout()
plt.savefig("hud_overlay.png", dpi=150)
print("Wrote hud_overlay.png")
