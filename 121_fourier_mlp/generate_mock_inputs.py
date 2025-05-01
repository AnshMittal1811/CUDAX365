import numpy as np

x = np.random.rand(1024, 3).astype(np.float32)
x.tofile("inputs.bin")
print("wrote inputs.bin")
