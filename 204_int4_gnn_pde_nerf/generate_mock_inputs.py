import numpy as np

rng = np.random.RandomState(0)

mesh = rng.randn(512, 8).astype(np.float32)
pde_state = rng.rand(128, 128).astype(np.float32)
volume = rng.rand(32, 32, 32).astype(np.float32)

np.save("gnn_input.npy", mesh)
np.save("pde_state.npy", pde_state)
np.save("nerf_volume.npy", volume)

print("Wrote gnn_input.npy, pde_state.npy, nerf_volume.npy")
