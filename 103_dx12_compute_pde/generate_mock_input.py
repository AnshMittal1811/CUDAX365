import numpy as np
w, h = 128, 128
x = np.random.rand(w * h).astype(np.float32)
x.tofile("input.bin")
print("wrote input.bin")
