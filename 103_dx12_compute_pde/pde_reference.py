import numpy as np
w, h = 128, 128
x = np.fromfile("input.bin", dtype=np.float32)
if x.size != w * h:
    raise SystemExit("input size mismatch")
out = x * 0.99
out.astype(np.float32).tofile("output_ref.bin")
print("wrote output_ref.bin")
