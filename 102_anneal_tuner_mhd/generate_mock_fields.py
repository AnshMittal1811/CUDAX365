import numpy as np

nx, ny = 128, 128
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
rho = 1.0 + 0.1 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
phi = 0.05 * np.cos(4 * np.pi * X)

rho.astype(np.float32).tofile("rho.bin")
phi.astype(np.float32).tofile("phi.bin")
print("wrote rho.bin and phi.bin")
