import numpy as np

weights = np.random.randn(1024).astype(np.float32)
scale = np.max(np.abs(weights)) / 7.0
q = np.clip(np.round(weights / scale), -8, 7) * scale
np.save("citex_q4.npy", q)

with open("citex_quant_log.txt", "w", encoding="utf-8") as f:
    f.write("quantized to 4-bit\n")

print("Wrote citex_q4.npy and citex_quant_log.txt")
